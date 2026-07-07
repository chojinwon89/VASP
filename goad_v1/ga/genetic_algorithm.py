"""
Improved Genetic Algorithm for GOAD v1.0

Surface atoms are completely fixed during GA optimization.
Includes molecular torsions in the genome.
The molecule is kept inside the central unit cell so it doesn't
wrap to the edges via PBC.

Z-placement rationale
---------------------
The GA encodes the molecule's centre-of-mass (COM) position, but
we bias the *initial* orientation so that the chemically relevant
surface-facing atom is placed at a physically motivated distance:

  - Molecules WITH oxygen  : lowest O atom placed at surface_z_max + o_target_z
                             (default 2.3 Å — known O–metal equilibrium distance)
  - Molecules WITHOUT oxygen but WITH carbon : lowest C atom placed at
                             surface_z_max + c_target_z
                             (default 2.1 Å — known C–metal equilibrium distance)
  - Molecules with neither O nor C (edge case): lowest atom used with c_target_z

This guarantees every generation-0 individual has the reactive atom close
to the surface, giving a real energy gradient from the very first evaluation
instead of wasting 30+ generations on a flat landscape.

Early stopping
--------------
The GA stops early if the best energy has not improved by more than
early_stop_tol eV in the last early_stop_patience generations.
Default: patience=30, tol=0.001 eV.

_create_system rotation order
------------------------------
IMPORTANT: rotation must happen BEFORE translation to the target COM.
  1. Centre molecule at origin
  2. Rotate around origin  (preserves the O/C-facing-down orientation)
  3. Translate COM to target position

Doing rotation AFTER translation would spin the molecule around its
displaced COM, randomising which atom ends up lowest and defeating
the O/C-bias encoded in _initialize_population.
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
    """Rotate atoms in place using ZYX Euler angles (degrees).

    Rotation is always performed around the current COM, so call this
    only when the molecule is already centred at the origin (or wherever
    the rotation pivot should be).
    """
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

        # ── correct order: centre → rotate → translate ──────────────────
        molecule_copy.translate(-molecule_copy.get_center_of_mass())
        _apply_rotation_worker(molecule_copy, individual['orientation'])
        molecule_copy.translate(individual['position'])
        # ────────────────────────────────────────────────────────────────

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
    - Initial population biased so the reactive atom faces the surface:
        * Has O atoms  -> lowest O at o_target_z = 2.3 Å above surface
        * No O, has C  -> lowest C at c_target_z = 2.1 Å above surface
        * Neither      -> lowest atom at c_target_z
    - Early stopping when best energy has not improved by > early_stop_tol eV
      in the last early_stop_patience generations (default: 30 gens, 0.001 eV)
    """

    def __init__(self, surface: Atoms, molecule: Atoms, calculator,
                 surface_energy: float, molecule_energy: float,
                 calculator_type: Optional[str] = None,
                 n_fixed_layers: int = 1,
                 generations: int = 200, population_size: int = 30,
                 mutation_rate: float = 0.3, crossover_rate: float = 0.7,
                 elite_size: int = 5, verbose: bool = True,
                 search_radius: Optional[float] = None,
                 center_in_cell: bool = True,
                 n_workers: Optional[int] = None,
                 o_target_z: float = 2.3,
                 c_target_z: float = 2.1,
                 early_stop_patience: int = 30,
                 early_stop_tol: float = 0.001):
        """
        Initialize GA.

        Args:
            surface: Surface structure
            molecule: Molecule structure
            calculator: ASE calculator
            calculator_type: CalculatorManager calculator type string.
                            Required for parallel population evaluation workers.
            surface_energy: Reference energy of surface
            molecule_energy: Reference energy of molecule
            n_fixed_layers: Number of layers to keep fixed (info only)
            generations: Maximum number of generations (default 200).
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
            o_target_z: Target distance (Å) between the lowest O atom and the
                        surface top layer in the initial population.
                        Default 2.3 Å — known O–metal equilibrium distance.
                        Used when molecule HAS oxygen atoms.
            c_target_z: Target distance (Å) between the lowest C atom and the
                        surface top layer in the initial population.
                        Default 2.1 Å — known C–metal equilibrium distance.
                        Used when molecule has NO oxygen atoms (pure hydrocarbons).
            early_stop_patience: Stop if best energy has not improved by more
                                 than early_stop_tol eV in this many consecutive
                                 generations. Default 30.
            early_stop_tol: Minimum improvement threshold (eV) for early
                            stopping. Default 0.001 eV.
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

        # Z-placement target distances
        self.o_target_z = o_target_z   # O–metal: 2.3 Å
        self.c_target_z = c_target_z   # C–metal: 2.1 Å

        # Early stopping
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol

        # Surface properties (resolves self.search_radius and self.surface_z_max)
        self._analyze_surface()

        # Torsion handling
        self.torsion_handler = TorsionHandler(molecule)
        self.n_torsions = self.torsion_handler.n_torsions

        if self.n_torsions > 0:
            logger.info(f"Molecule has {self.n_torsions} rotatable bonds")
        else:
            logger.info("Molecule has no rotatable bonds (rigid)")

        # Determine placement strategy once at init time and log it
        symbols = self.molecule.get_chemical_symbols()
        self._o_idx = [i for i, s in enumerate(symbols) if s == 'O']
        self._c_idx = [i for i, s in enumerate(symbols) if s == 'C']

        if self._o_idx:
            self._placement_mode = 'O'
            self._placement_target_z = self.o_target_z
            logger.info(f"Placement mode: O-facing-surface  "
                        f"(target {self.o_target_z:.2f} Å, {len(self._o_idx)} O atoms)")
        elif self._c_idx:
            self._placement_mode = 'C'
            self._placement_target_z = self.c_target_z
            logger.info(f"Placement mode: C-facing-surface  "
                        f"(target {self.c_target_z:.2f} Å, {len(self._c_idx)} C atoms, no O)")
        else:
            self._placement_mode = 'any'
            self._placement_target_z = self.c_target_z
            logger.warning("No O or C atoms found — using lowest atom with "
                           f"c_target_z={self.c_target_z:.2f} Å as fallback")

        # Population and history
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_energy = float('inf')

        # Vertical search-space parameters for mutation clamping
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
        logger.info("=" * 60)
        logger.info("GENETIC ALGORITHM - GOAD v1.0")
        logger.info("=" * 60)
        logger.info(f"Population: {self.population_size}")
        logger.info(f"Max generations: {self.generations}")
        logger.info(f"Early stopping: patience={self.early_stop_patience} gens, "
                    f"tol={self.early_stop_tol} eV")
        logger.info("Surface: FIXED during GA search (will relax in post-GA BFGS)")
        logger.info("Molecule: FREE TO MOVE")
        logger.info(f"\nInitial placement strategy: {self._placement_mode}-facing-surface")
        logger.info(f"  Target distance : {self._placement_target_z:.2f} Å")
        logger.info(f"  Target Z abs    : "
                    f"{self.surface_z_max + self._placement_target_z:.2f} Å")
        logger.info(f"\nMutation Z clamp (COM above surface_z_max):")
        logger.info(f"  [{self.surface_z_max + self.surface_buffer:.2f}, "
                    f"{self.surface_z_max + self.max_height:.2f}] Å")
        logger.info(f"\nGenome composition:")
        logger.info(f"  Position (X, Y, Z):     3 genes")
        logger.info(f"  Orientation (α, β, γ): 3 genes")
        logger.info(f"  Torsions:               {self.n_torsions} genes")
        logger.info(f"  Total genes per individual: {6 + self.n_torsions}")
        logger.info("=" * 60 + "\n")

        self._initialize_population()

        no_improve_count = 0
        best_energy_at_last_check = float('inf')

        for gen in range(self.generations):
            self._evaluate_population()

            if self.verbose:
                recent = self.fitness_history[-self.population_size:]
                best_gen = min(recent) if recent else float('inf')
                logger.info(f"Gen {gen+1}/{self.generations} | "
                            f"Best: {best_gen:.4f} eV | "
                            f"Overall best: {self.best_energy:.4f} eV | "
                            f"No-improve: {no_improve_count}/{self.early_stop_patience}")

            # Early stopping check
            if self.best_energy < best_energy_at_last_check - self.early_stop_tol:
                best_energy_at_last_check = self.best_energy
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.early_stop_patience:
                logger.info(
                    f"\nEarly stopping at gen {gen+1}: no improvement > "
                    f"{self.early_stop_tol} eV in {self.early_stop_patience} generations."
                )
                break

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
        Initialize population with chemically informed surface-facing bias.

        Placement priority:
          1. Molecule has O atoms  -> lowest O placed at surface_z_max + o_target_z (2.3 Å)
          2. No O, has C atoms     -> lowest C placed at surface_z_max + c_target_z (2.1 Å)
          3. Neither               -> lowest atom placed at surface_z_max + c_target_z

        For each individual:
          1. Apply random torsions (if any)
          2. Centre molecule at origin
          3. Apply random Euler rotation (around origin)
          4. Find the lowest relevant atom (O > C > any) in the rotated molecule
          5. Set COM Z so that atom lands at surface_z_max + target_z
          6. Record (x, y, com_z) as the genome position
        """
        mode_label = {
            'O':   f"O-facing-surface (target {self.o_target_z:.2f} Å)",
            'C':   f"C-facing-surface (target {self.c_target_z:.2f} Å)",
            'any': f"lowest-atom fallback (target {self.c_target_z:.2f} Å)",
        }[self._placement_mode]

        logger.info(f"Initializing population — {mode_label}")
        logger.info(f"  Target Z absolute: "
                    f"{self.surface_z_max + self._placement_target_z:.2f} Å")

        for _ in range(self.population_size):
            x = self.surface_center_xy[0] + np.random.uniform(
                -self.search_radius, self.search_radius)
            y = self.surface_center_xy[1] + np.random.uniform(
                -self.search_radius, self.search_radius)

            euler_angles   = np.random.uniform(0, 360, 3)
            torsion_angles = np.random.uniform(0, 360, self.n_torsions)

            # Build a temporary molecule centred at origin, then rotate
            mol_tmp = self.molecule.copy()
            if self.n_torsions > 0:
                mol_tmp = self.torsion_handler.apply_torsions(mol_tmp, torsion_angles)

            mol_tmp.translate(-mol_tmp.get_center_of_mass())   # centre at origin
            self._apply_rotation(mol_tmp, euler_angles)         # rotate around origin

            # Pick the lowest atom of the relevant element type
            if self._placement_mode == 'O':
                lowest_z = mol_tmp.positions[self._o_idx, 2].min()
            elif self._placement_mode == 'C':
                lowest_z = mol_tmp.positions[self._c_idx, 2].min()
            else:
                lowest_z = mol_tmp.positions[:, 2].min()

            # COM Z so that the target atom lands at surface_z_max + target_z
            com_z = self.surface_z_max + self._placement_target_z - lowest_z

            individual = {
                'position':    np.array([x, y, com_z]),
                'orientation': euler_angles,
                'torsions':    torsion_angles,
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
        """Build a (surface + positioned molecule) Atoms object.

        Order is critical:
          1. Apply torsions (if any)
          2. Centre molecule at origin
          3. Rotate around origin   <- preserves O/C-facing-down orientation
          4. Translate COM to target position
          5. (optional) snap COM back into unit cell in XY
        """
        surface_copy = self.surface.copy()
        molecule_copy = self.molecule.copy()

        if self.n_torsions > 0:
            molecule_copy = self.torsion_handler.apply_torsions(
                molecule_copy,
                individual['torsions']
            )

        # ── correct order: centre → rotate → translate ──────────────────
        molecule_copy.translate(-molecule_copy.get_center_of_mass())
        self._apply_rotation(molecule_copy, individual['orientation'])
        molecule_copy.translate(individual['position'])
        # ────────────────────────────────────────────────────────────────

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
        """Rotate atoms in place using ZYX Euler angles (degrees).

        Rotates around the current COM, so call this only when the
        molecule is already centred at the origin.
        """
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
                         surface_z_max + max_height] after every
        position mutation so molecules cannot drift out of the MLFF
        interaction window over generations.
        """
        mutation_choice = np.random.random()

        if mutation_choice < 0.33:
            individual['position'] += np.random.normal(0, 0.5, 3)
            z_min = self.surface_z_max + self.surface_buffer
            z_max = self.surface_z_max + self.max_height
            individual['position'][2] = float(
                np.clip(individual['position'][2], z_min, z_max)
            )

        elif mutation_choice < 0.66:
            individual['orientation'] += np.random.normal(0, 10, 3)
            individual['orientation'] = individual['orientation'] % 360

        else:
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
