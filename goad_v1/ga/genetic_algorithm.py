"""
Improved Genetic Algorithm for GOAD v1.0

Surface atoms are completely fixed during GA optimization.
Includes molecular torsions in the genome.
The molecule is kept inside the central unit cell so it doesn't
wrap to the edges via PBC.

Z-placement rationale
---------------------
The GA encodes the molecule's centre-of-mass (COM) position, not
individual atom positions.  For a molecule like glycerol that adsorbs
through an oxygen atom:

    O–metal equilibrium distance : ~2.3 Å  (experiment / DFT)
    Glycerol COM above bottom O  : ~2.0 Å  (molecular geometry, flat-lying)
    → COM equilibrium Z          : surface_z_max + 4.3 Å

We therefore initialise the COM in [surface_z_max + 2.0,
surface_z_max + 5.0] Å and clamp Z mutations to the same range.

History of surface_buffer changes:
  v1.0 original : 1.5 Å  (too low, atom clashes)
  fix 1         : 3.0 Å  (too high — bottom O still 3.4 Å from surface)
  fix 2 (this)  : 2.0 Å  (bottom O ~0.3 Å above surface at minimum,
                           strong repulsive gradient pulls toward 2.3 Å)
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from typing import Dict, List, Optional, Tuple
import logging

from ..calculator_manager import CalculatorManager
from ..utils.torsion_handler import TorsionHandler

logger = logging.getLogger(__name__)


_WORKER_SURFACE: Optional[Atoms] = None
_WORKER_MOLECULE: Optional[Atoms] = None
_WORKER_TORSION_HANDLER: Optional[TorsionHandler] = None
_WORKER_SURFACE_ENERGY: Optional[float] = None
_WORKER_MOLECULE_ENERGY: Optional[float] = None
_WORKER_CENTER_IN_CELL: bool = True
_WORKER_CALCULATOR = None


def _init_energy_worker(surface: Atoms, molecule: Atoms,
                        surface_energy: float, molecule_energy: float,
                        center_in_cell: bool, calculator_type: str):
    """Initialize per-process worker state and calculator."""
    global _WORKER_SURFACE, _WORKER_MOLECULE, _WORKER_TORSION_HANDLER
    global _WORKER_SURFACE_ENERGY, _WORKER_MOLECULE_ENERGY, _WORKER_CENTER_IN_CELL
    global _WORKER_CALCULATOR

    _WORKER_SURFACE = surface.copy()
    _WORKER_SURFACE.calc = None
    _WORKER_MOLECULE = molecule.copy()
    _WORKER_MOLECULE.calc = None
    _WORKER_TORSION_HANDLER = TorsionHandler(_WORKER_MOLECULE)
    _WORKER_SURFACE_ENERGY = surface_energy
    _WORKER_MOLECULE_ENERGY = molecule_energy
    _WORKER_CENTER_IN_CELL = center_in_cell
    _WORKER_CALCULATOR = CalculatorManager.get_calculator(calculator_type)


def _apply_rotation_worker(atoms: Atoms, euler_angles: np.ndarray):
    """Rotate atoms in place using ZYX Euler angles (degrees)."""
    angles_rad = np.deg2rad(euler_angles)
    alpha, beta, gamma = angles_rad

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
    Rz = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    com = atoms.get_center_of_mass()
    positions = atoms.get_positions()
    relative_pos = positions - com
    rotated_pos = relative_pos @ R.T
    atoms.set_positions(rotated_pos + com)


def _evaluate_individual_worker(task: Tuple[int, Dict]) -> Tuple[int, float, Optional[Atoms]]:
    """Evaluate one individual in a worker process."""
    idx, individual = task
    try:
        surface_copy = _WORKER_SURFACE.copy()
        molecule_copy = _WORKER_MOLECULE.copy()

        if _WORKER_TORSION_HANDLER.n_torsions > 0:
            molecule_copy = _WORKER_TORSION_HANDLER.apply_torsions(
                molecule_copy,
                individual['torsions']
            )

        molecule_copy.translate(individual['position'] - molecule_copy.get_center_of_mass())
        _apply_rotation_worker(molecule_copy, individual['orientation'])

        if _WORKER_CENTER_IN_CELL:
            cell = np.array(surface_copy.get_cell())
            A2 = cell[:2, :2]
            com = molecule_copy.get_center_of_mass()
            try:
                inv = np.linalg.inv(A2)
                frac_xy = inv @ com[:2]
                frac_xy = frac_xy % 1.0
                new_xy = A2 @ frac_xy
                shift = np.array([new_xy[0] - com[0],
                                  new_xy[1] - com[1],
                                  0.0])
                molecule_copy.translate(shift)
            except np.linalg.LinAlgError:
                pass

        system = surface_copy + molecule_copy
        fixed_indices = list(range(len(surface_copy)))
        system.set_constraint(FixAtoms(indices=fixed_indices))
        system.calc = _WORKER_CALCULATOR

        energy = system.get_potential_energy()
        e_ads = energy - (_WORKER_SURFACE_ENERGY + _WORKER_MOLECULE_ENERGY)
        return idx, e_ads, system
    except Exception as e:
        logger.warning(f"Energy calculation failed in worker: {e}")
        return idx, 1000.0, None


def _cuda_is_available() -> bool:
    """Return True if a CUDA device is visible to this process."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class GeneticAlgorithm:
    """
    Genetic Algorithm for molecular adsorption on surfaces.

    Key features for v1.0:
    - Surface atoms are COMPLETELY FIXED during GA search (will relax in post-GA BFGS)
    - Only molecule position and orientation vary
    - Support for molecular torsions
    - Molecule kept inside the central unit cell (no edge-wrapping)
    - Z placement anchored to physical O–metal distance (~2.3 Å)
    """

    def __init__(self, surface: Atoms, molecule: Atoms, calculator,
                 surface_energy: float, molecule_energy: float,
                 calculator_type: Optional[str] = None,
                 n_fixed_layers: int = 1,
                 generations: int = 50, population_size: int = 30,
                 mutation_rate: float = 0.3, crossover_rate: float = 0.7,
                 elite_size: int = 5, verbose: bool = True,
                 search_radius: Optional[float] = None,
                 center_in_cell: bool = True,
                 n_workers: Optional[int] = None):
        """
        Initialize GA.

        Args:
            surface: Surface structure (should be the POST-relaxation slab so
                     surface_z_max matches the actual top-layer Z used in GA).
            molecule: Molecule structure
            calculator: ASE calculator
            calculator_type: CalculatorManager calculator type string.
                            Required for parallel population evaluation workers.
            surface_energy: Reference energy of surface
            molecule_energy: Reference energy of molecule
            n_fixed_layers: Number of layers to keep fixed (info only)
            generations: Number of generations
            population_size: Population size
            mutation_rate: Mutation rate (0-1)
            crossover_rate: Crossover rate (0-1)
            elite_size: Number of elite individuals to preserve
            verbose: Print progress
            search_radius: Lateral half-width of the initial sampling box (Å).
                           If None, auto-set to 1/4 of the smaller in-plane
                           cell vector.
            center_in_cell: If True, snap the molecule's COM back into the
                            central unit cell after every placement.
            n_workers: Number of evaluation workers. If None, checks (in order):
                       GOAD_N_WORKERS env var, SLURM_CPUS_PER_TASK, os.cpu_count().
                       NOTE: always forced to 1 when a CUDA device is present to
                       avoid ProcessPoolExecutor / CUDA context deadlocks.
        """
        self.surface = surface.copy()
        self.surface.calc = None
        self.molecule = molecule.copy()
        self.molecule.calc = None
        self.calculator = calculator
        self.calculator_type = calculator_type
        self.surface_energy = surface_energy
        self.molecule_energy = molecule_energy
        self.n_fixed_layers = n_fixed_layers

        # GA parameters
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.verbose = verbose

        # Cell-centering controls
        self.center_in_cell = center_in_cell
        self._user_search_radius = search_radius
        self.n_workers = self._resolve_worker_count(n_workers)

        # Surface properties (resolves self.search_radius and self.surface_z_max)
        self._analyze_surface()

        # Torsion handling
        self.torsion_handler = TorsionHandler(molecule)
        self.n_torsions = self.torsion_handler.n_torsions

        if self.n_torsions > 0:
            logger.info(f"Molecule has {self.n_torsions} rotatable bonds")
        else:
            logger.info("Molecule has no rotatable bonds (rigid)")

        # Population and history
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_energy = float('inf')

        # ------------------------------------------------------------------
        # Vertical search-space parameters (COM above surface_z_max)
        # ------------------------------------------------------------------
        # Physical basis for glycerol / alcohols on transition metals:
        #   O–metal equilibrium distance : ~2.3 Å
        #   Glycerol COM above bottom O  : ~2.0 Å (flat-lying orientation)
        #   → COM equilibrium            : surface_z_max + 4.3 Å
        #
        # surface_buffer = 2.0 Å → bottom O at ~0.3 Å (repulsive, very
        #                           strong gradient toward 2.3 Å equilibrium)
        # max_height     = 5.0 Å → bottom O at ~3.0 Å (edge of MLFF range)
        #
        # Change history:
        #   original : 1.5 / 8.0  → bottom O 6+ Å away, flat landscape
        #   fix 1    : 3.0 / 5.0  → bottom O 3.4 Å away, still too far
        #   fix 2    : 2.0 / 5.0  → bottom O 0.3–3.0 Å, brackets 2.3 Å ✓
        self.surface_buffer = 2.0   # Å  COM minimum above surface_z_max
        self.max_height     = 5.0   # Å  COM maximum above surface_z_max

    # ------------------------------------------------------------------
    # Surface analysis & search-radius default
    # ------------------------------------------------------------------
    def _analyze_surface(self):
        """Analyze surface properties and resolve default search radius."""
        positions = self.surface.get_positions()
        z_coords = positions[:, 2]

        self.surface_z_min = z_coords.min()
        self.surface_z_max = z_coords.max()
        self.surface_center_xy = positions[:, :2].mean(axis=0)

        # In-plane cell vector lengths
        cell = np.array(self.surface.get_cell())
        ax = np.linalg.norm(cell[0, :2])
        ay = np.linalg.norm(cell[1, :2])

        if self._user_search_radius is None:
            self.search_radius = 0.25 * min(ax, ay)
            radius_src = "auto (1/4 of min in-plane cell)"
        else:
            self.search_radius = float(self._user_search_radius)
            radius_src = "user-specified"

        logger.info(f"Surface Z range: {self.surface_z_min:.2f} – {self.surface_z_max:.2f} Å")
        logger.info(f"Surface center (XY): "
                    f"({self.surface_center_xy[0]:.2f}, {self.surface_center_xy[1]:.2f})")
        logger.info(f"In-plane cell: |a|={ax:.2f} Å  |b|={ay:.2f} Å")
        logger.info(f"Search radius (lateral): {self.search_radius:.2f} Å  [{radius_src}]")
        logger.info(f"Center-in-cell enforcement: {self.center_in_cell}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """Run the genetic algorithm and return results."""
        z_min_abs = self.surface_z_max + self.surface_buffer
        z_max_abs = self.surface_z_max + self.max_height

        logger.info("=" * 60)
        logger.info("GENETIC ALGORITHM - GOAD v1.0")
        logger.info("=" * 60)
        logger.info(f"Population: {self.population_size}")
        logger.info(f"Generations: {self.generations}")
        logger.info("Surface: FIXED during GA search (will relax in post-GA BFGS)")
        logger.info("Molecule: FREE TO MOVE")
        logger.info(f"\nZ search window (COM above surface top layer):")
        logger.info(f"  surface_z_max : {self.surface_z_max:.2f} Å")
        logger.info(f"  COM min (Z)   : {z_min_abs:.2f} Å  "
                    f"(+{self.surface_buffer:.1f} Å → bottom O ~{self.surface_buffer - 2.0:.1f} Å above surface)")
        logger.info(f"  COM max (Z)   : {z_max_abs:.2f} Å  "
                    f"(+{self.max_height:.1f} Å → bottom O ~{self.max_height - 2.0:.1f} Å above surface)")
        logger.info(f"\nGenome composition:")
        logger.info(f"  Position (X, Y, Z):     3 genes")
        logger.info(f"  Orientation (α, β, γ): 3 genes")
        logger.info(f"  Torsions:               {self.n_torsions} genes")
        logger.info(f"  Total genes per individual: {6 + self.n_torsions}")
        logger.info("=" * 60 + "\n")

        self._initialize_population()

        for gen in range(self.generations):
            self._evaluate_population()

            if self.verbose:
                recent = self.fitness_history[-self.population_size:]
                best_gen = min(recent) if recent else float('inf')
                logger.info(f"Gen {gen+1}/{self.generations} | "
                            f"Best: {best_gen:.4f} eV | "
                            f"Overall best: {self.best_energy:.4f} eV")

            self._selection_crossover_mutation()

        logger.info("\n" + "=" * 60)
        logger.info("GA COMPLETED")
        logger.info("=" * 60)

        return self._get_results()

    # ------------------------------------------------------------------
    # Initial population
    # ------------------------------------------------------------------
    def _initialize_population(self):
        """
        Initialize random population with COM placed in the physical
        interaction window above the surface.

        Z range: [surface_z_max + surface_buffer, surface_z_max + max_height]
                 = [surface_z_max + 2.0, surface_z_max + 5.0]  Å

        For glycerol (COM ~2.0 Å above bottom O when flat-lying), this
        places the bottom oxygen 0.0–3.0 Å above the surface top layer —
        bracketing the ~2.3 Å O–metal equilibrium distance.
        """
        logger.info("Initializing population...")
        logger.info(f"  COM Z range: [{self.surface_z_max + self.surface_buffer:.2f}, "
                    f"{self.surface_z_max + self.max_height:.2f}] Å")

        for i in range(self.population_size):
            x = self.surface_center_xy[0] + np.random.uniform(
                -self.search_radius, self.search_radius)
            y = self.surface_center_xy[1] + np.random.uniform(
                -self.search_radius, self.search_radius)
            z = self.surface_z_max + np.random.uniform(
                self.surface_buffer, self.max_height)

            euler_angles   = np.random.uniform(0, 360, 3)
            torsion_angles = np.random.uniform(0, 360, self.n_torsions)

            individual = {
                'position':    np.array([x, y, z]),
                'orientation': np.array(euler_angles),
                'torsions':    np.array(torsion_angles),
                'energy':      None,
                'structure':   None,
            }
            self.population.append(individual)

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------
    def _evaluate_population(self):
        """Evaluate fitness of all individuals.

        Always runs serially when:
          - n_workers == 1, or
          - no calculator_type is set, or
          - a CUDA device is present (ProcessPoolExecutor + CUDA = deadlock).
        """
        pending = [(idx, individual) for idx, individual in enumerate(self.population)
                   if individual['energy'] is None]
        if not pending:
            return

        use_serial = (
            self.n_workers == 1
            or not self.calculator_type
            or _cuda_is_available()
        )

        if use_serial:
            self._evaluate_population_serial(pending)
            return

        try:
            with ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=_init_energy_worker,
                initargs=(
                    self.surface,
                    self.molecule,
                    self.surface_energy,
                    self.molecule_energy,
                    self.center_in_cell,
                    self.calculator_type,
                )
            ) as executor:
                for idx, energy, structure in executor.map(_evaluate_individual_worker, pending):
                    individual = self.population[idx]
                    individual['energy'] = energy
                    individual['structure'] = structure
                    self.fitness_history.append(energy)
                    if energy < self.best_energy:
                        self.best_energy = energy
                        self.best_individual = individual.copy()
        except Exception as e:
            logger.warning(f"Parallel evaluation failed, falling back to serial: {e}")
            self._evaluate_population_serial(pending)

    @staticmethod
    def _resolve_worker_count(n_workers: Optional[int]) -> int:
        """Resolve worker count from explicit value, env vars, or local CPU count."""
        if n_workers is not None:
            return max(1, int(n_workers))

        goad_workers = os.environ.get("GOAD_N_WORKERS")
        if goad_workers:
            try:
                count = max(1, int(goad_workers))
                logger.debug(f"n_workers={count} from GOAD_N_WORKERS")
                return count
            except ValueError:
                logger.warning(f"Invalid GOAD_N_WORKERS={goad_workers!r}, ignoring")

        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus:
            try:
                count = max(1, int(slurm_cpus))
                logger.debug(f"n_workers={count} from SLURM_CPUS_PER_TASK")
                return count
            except ValueError:
                logger.warning(f"Invalid SLURM_CPUS_PER_TASK={slurm_cpus!r}, using os.cpu_count()")

        return max(1, os.cpu_count() or 1)

    def _evaluate_population_serial(self, pending: List[Tuple[int, Dict]]):
        """Serial evaluation fallback."""
        for idx, individual in pending:
            energy = self._calculate_energy(individual)
            individual['energy'] = energy
            self.population[idx] = individual
            self.fitness_history.append(energy)
            if energy < self.best_energy:
                self.best_energy = energy
                self.best_individual = individual.copy()

    def _calculate_energy(self, individual: Dict) -> float:
        """Calculate adsorption energy of a single placement."""
        try:
            system = self._create_system(individual)

            surface_atoms_count = len(self.surface)
            fixed_indices = list(range(surface_atoms_count))
            system.set_constraint(FixAtoms(indices=fixed_indices))
            system.calc = self.calculator

            energy = system.get_potential_energy()
            e_ads = energy - (self.surface_energy + self.molecule_energy)

            individual['structure'] = system
            return e_ads

        except Exception as e:
            logger.warning(f"Energy calculation failed: {e}")
            return 1000.0

    # ------------------------------------------------------------------
    # System builder (with cell-centering)
    # ------------------------------------------------------------------
    def _create_system(self, individual: Dict) -> Atoms:
        """Build a (surface + positioned molecule) Atoms object."""
        surface_copy = self.surface.copy()
        molecule_copy = self.molecule.copy()

        if self.n_torsions > 0:
            molecule_copy = self.torsion_handler.apply_torsions(
                molecule_copy,
                individual['torsions']
            )

        molecule_copy.translate(
            individual['position'] - molecule_copy.get_center_of_mass()
        )

        self._apply_rotation(molecule_copy, individual['orientation'])

        if self.center_in_cell:
            cell = np.array(surface_copy.get_cell())
            A2 = cell[:2, :2]
            com = molecule_copy.get_center_of_mass()
            try:
                inv = np.linalg.inv(A2)
                frac_xy = inv @ com[:2]
                frac_xy = frac_xy % 1.0
                new_xy = A2 @ frac_xy
                shift = np.array([new_xy[0] - com[0],
                                  new_xy[1] - com[1],
                                  0.0])
                molecule_copy.translate(shift)
            except np.linalg.LinAlgError:
                pass

        system = surface_copy + molecule_copy
        return system

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------
    def _apply_rotation(self, atoms: Atoms, euler_angles: np.ndarray):
        """Rotate atoms in place using ZYX Euler angles (degrees)."""
        angles_rad = np.deg2rad(euler_angles)
        alpha, beta, gamma = angles_rad

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha),  np.cos(alpha)]
        ])
        Ry = np.array([
            [ np.cos(beta), 0, np.sin(beta)],
            [ 0,            1, 0           ],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        Rz = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma),  np.cos(gamma), 0],
            [0,              0,             1]
        ])
        R = Rz @ Ry @ Rx

        com = atoms.get_center_of_mass()
        positions = atoms.get_positions()
        relative_pos = positions - com
        rotated_pos = relative_pos @ R.T
        atoms.set_positions(rotated_pos + com)

    # ------------------------------------------------------------------
    # Evolution operators
    # ------------------------------------------------------------------
    def _selection_crossover_mutation(self):
        """Selection, crossover, and mutation step."""
        self.population.sort(key=lambda x: x['energy'])

        new_population = self.population[:self.elite_size].copy()

        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate:
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                child = self._crossover(parent1, parent2)
            else:
                parent = self._select_parent()
                child = self._mutate(parent.copy())
            new_population.append(child)

        self.population = new_population

    def _select_parent(self) -> Dict:
        """Tournament selection."""
        tournament_size = 5
        tournament = np.random.choice(len(self.population), tournament_size, replace=False)
        winner_idx = min(tournament, key=lambda i: self.population[i]['energy'])
        return self.population[winner_idx].copy()

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Single-point-ish crossover: position from p1, orientation from p2, torsions mixed."""
        child = {
            'position':    parent1['position'].copy(),
            'orientation': parent2['orientation'].copy(),
            'torsions':    np.zeros(self.n_torsions),
            'energy':      None,
            'structure':   None,
        }
        for i in range(self.n_torsions):
            child['torsions'][i] = (parent1['torsions'][i]
                                    if np.random.random() < 0.5
                                    else parent2['torsions'][i])
        return child

    def _mutate(self, individual: Dict) -> Dict:
        """
        Mutate position, orientation, or torsions.

        Z is clamped to [surface_z_max + surface_buffer,
                         surface_z_max + max_height]
        after every position mutation so molecules cannot drift above
        the MLFF interaction range over generations.
        """
        mutation_choice = np.random.random()

        if mutation_choice < 0.33:
            # Position mutation — Gaussian step, then clamp Z
            individual['position'] += np.random.normal(0, 0.5, 3)
            z_min = self.surface_z_max + self.surface_buffer
            z_max = self.surface_z_max + self.max_height
            individual['position'][2] = float(
                np.clip(individual['position'][2], z_min, z_max)
            )

        elif mutation_choice < 0.66:
            # Orientation mutation
            individual['orientation'] += np.random.normal(0, 10, 3)
            individual['orientation'] = individual['orientation'] % 360

        else:
            # Torsion mutation
            for i in range(self.n_torsions):
                if np.random.random() < 0.5:
                    individual['torsions'][i] += np.random.normal(0, 20)
                    individual['torsions'][i] = individual['torsions'][i] % 360

        individual['energy'] = None
        individual['structure'] = None
        return individual

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    def _get_results(self) -> Dict:
        results = {
            'best_individual': self.best_individual,
            'best_energy':     self.best_energy,
            'best_structure':  self.best_individual['structure'] if self.best_individual else None,
            'fitness_history': self.fitness_history,
            'generations':     self.generations,
            'population_size': self.population_size,
        }

        logger.info(f"Best E_ads found: {self.best_energy:.4f} eV")
        if self.best_individual:
            p = self.best_individual['position']
            o = self.best_individual['orientation']
            z_above = p[2] - self.surface_z_max
            logger.info(f"Best position (Å): X={p[0]:.2f}, Y={p[1]:.2f}, Z={p[2]:.2f} "
                        f"(+{z_above:.2f} Å above surface top)")
            logger.info(f"Best orientation (°): α={o[0]:.1f}, β={o[1]:.1f}, γ={o[2]:.1f}")
            if self.n_torsions > 0:
                logger.info(
                    "Best torsions (°): "
                    + " ".join(f"{t:.1f}" for t in self.best_individual['torsions'])
                )

        return results
