"""
Improved Genetic Algorithm for GOAD v1.0

Surface atoms are completely fixed during GA optimization.
Includes molecular torsions in the genome.
The molecule is kept inside the central unit cell so it doesn't
wrap to the edges via PBC.
"""

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from typing import Dict, List, Optional, Tuple
import logging

from ..utils.torsion_handler import TorsionHandler

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """
    Genetic Algorithm for molecular adsorption on surfaces.

    Key features for v1.0:
    - Surface atoms are COMPLETELY FIXED during GA
    - Only molecule position and orientation vary
    - Support for molecular torsions
    - Molecule kept inside the central unit cell (no edge-wrapping)
    """

    def __init__(self, surface: Atoms, molecule: Atoms, calculator,
                 surface_energy: float, molecule_energy: float,
                 n_fixed_layers: int = 1,
                 generations: int = 50, population_size: int = 30,
                 mutation_rate: float = 0.3, crossover_rate: float = 0.7,
                 elite_size: int = 5, verbose: bool = True,
                 search_radius: Optional[float] = None,
                 center_in_cell: bool = True):
        """
        Initialize GA.

        Args:
            surface: Surface structure
            molecule: Molecule structure
            calculator: ASE calculator
            surface_energy: Reference energy of surface
            molecule_energy: Reference energy of molecule
            n_fixed_layers: Number of layers to keep fixed (info only, all surface fixed in GA)
            generations: Number of generations
            population_size: Population size
            mutation_rate: Mutation rate (0-1)
            crossover_rate: Crossover rate (0-1)
            elite_size: Number of elite individuals to preserve
            verbose: Print progress
            search_radius: Lateral half-width of the initial sampling box (Å).
                           If None, auto-set to 1/4 of the smaller in-plane cell vector,
                           which keeps the molecule near the cell center.
            center_in_cell: If True, snap the molecule's center-of-mass back into
                            the central unit cell after every placement, so it
                            never wraps to an edge via PBC.
        """
        self.surface = surface
        self.molecule = molecule
        self.calculator = calculator
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
        self._user_search_radius = search_radius   # may be None

        # Surface properties (also resolves self.search_radius)
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

        # Vertical search-space parameters
        self.max_height = 8.0       # Maximum height above surface (Å)
        self.surface_buffer = 1.5   # Minimum distance from surface (Å)

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

        # Default search radius = 1/4 of the smaller in-plane vector.
        # That keeps initial COMs inside ~half the cell, well away from edges.
        if self._user_search_radius is None:
            self.search_radius = 0.25 * min(ax, ay)
            radius_src = "auto (1/4 of min in-plane cell)"
        else:
            self.search_radius = float(self._user_search_radius)
            radius_src = "user-specified"

        logger.info(f"Surface Z range: {self.surface_z_min:.2f} - {self.surface_z_max:.2f} Å")
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
        logger.info(f"Generations: {self.generations}")
        logger.info("Surface: FIXED during GA search (will relax in post-GA BFGS)")
        logger.info("Molecule: FREE TO MOVE")
        logger.info(f"\nGenome composition:")
        logger.info(f"  Position (X, Y, Z):     3 genes")
        logger.info(f"  Orientation (α, β, γ): 3 genes")
        logger.info(f"  Torsions:               {self.n_torsions} genes")
        logger.info(f"  Total genes per individual: {6 + self.n_torsions}")
        logger.info("=" * 60 + "\n")

        # Initialize population
        self._initialize_population()

        # Main GA loop
        for gen in range(self.generations):
            self._evaluate_population()

            if self.verbose:
                best_gen = min(self.fitness_history[-self.population_size:])
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
        """Initialize random population near the cell center."""
        logger.info("Initializing population...")

        for i in range(self.population_size):
            # Random molecule position above surface, centered on slab COM
            x = self.surface_center_xy[0] + np.random.uniform(-self.search_radius, self.search_radius)
            y = self.surface_center_xy[1] + np.random.uniform(-self.search_radius, self.search_radius)
            z = self.surface_z_max + np.random.uniform(self.surface_buffer, self.max_height)

            # Random orientation (Euler angles in degrees)
            euler_angles = np.random.uniform(0, 360, 3)

            # Random torsion angles (0-360 degrees)
            torsion_angles = np.random.uniform(0, 360, self.n_torsions)

            individual = {
                'position': np.array([x, y, z]),
                'orientation': np.array(euler_angles),
                'torsions': np.array(torsion_angles),
                'energy': None,
                'structure': None
            }

            self.population.append(individual)

    # ------------------------------------------------------------------
    # Fitness evaluation
    # ------------------------------------------------------------------
    def _evaluate_population(self):
        """Evaluate fitness of all individuals."""
        for individual in self.population:
            if individual['energy'] is None:
                individual['energy'] = self._calculate_energy(individual)
                self.fitness_history.append(individual['energy'])

                if individual['energy'] < self.best_energy:
                    self.best_energy = individual['energy']
                    self.best_individual = individual.copy()

    def _calculate_energy(self, individual: Dict) -> float:
        """Calculate adsorption energy of a single placement."""
        try:
            system = self._create_system(individual)

            # Fix all surface atoms (GA-stage convention)
            surface_atoms_count = len(self.surface)
            fixed_indices = list(range(surface_atoms_count))
            system.set_constraint(FixAtoms(indices=fixed_indices))

            # Attach calculator (modern ASE API)
            system.calc = self.calculator

            energy = system.get_potential_energy()
            e_ads = energy - (self.surface_energy + self.molecule_energy)

            individual['structure'] = system
            return e_ads

        except Exception as e:
            logger.warning(f"Energy calculation failed: {e}")
            return 1000.0  # large penalty

    # ------------------------------------------------------------------
    # System builder (with cell-centering)
    # ------------------------------------------------------------------
    def _create_system(self, individual: Dict) -> Atoms:
        """Build a (surface + positioned molecule) Atoms object."""
        surface_copy = self.surface.copy()
        molecule_copy = self.molecule.copy()

        # Apply torsions FIRST (before positioning)
        if self.n_torsions > 0:
            molecule_copy = self.torsion_handler.apply_torsions(
                molecule_copy,
                individual['torsions']
            )

        # Position molecule so its COM lands at the requested point
        molecule_copy.translate(
            individual['position'] - molecule_copy.get_center_of_mass()
        )

        # Apply rotation
        self._apply_rotation(molecule_copy, individual['orientation'])

        # ------------------------------------------------------------------
        # Keep the molecule's COM inside the central unit cell.
        # This prevents PBC wrapping the molecule to the edges/corners.
        # Vertical (Z) position is preserved.
        # ------------------------------------------------------------------
        if self.center_in_cell:
            cell = np.array(surface_copy.get_cell())
            A2 = cell[:2, :2]                       # 2x2 in-plane cell matrix
            com = molecule_copy.get_center_of_mass()
            try:
                inv = np.linalg.inv(A2)
                frac_xy = inv @ com[:2]             # fractional in-plane coords
                frac_xy = frac_xy % 1.0             # wrap to [0,1)
                new_xy = A2 @ frac_xy               # back to Cartesian
                shift = np.array([new_xy[0] - com[0],
                                  new_xy[1] - com[1],
                                  0.0])
                molecule_copy.translate(shift)
            except np.linalg.LinAlgError:
                # Non-periodic / singular cell: skip centering silently
                pass

        # Combine
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
        # Sort population by fitness (lowest energy first)
        self.population.sort(key=lambda x: x['energy'])

        # Keep elite
        new_population = self.population[:self.elite_size].copy()

        # Generate offspring
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
            'structure':   None
        }
        for i in range(self.n_torsions):
            child['torsions'][i] = parent1['torsions'][i] if np.random.random() < 0.5 \
                                   else parent2['torsions'][i]
        return child

    def _mutate(self, individual: Dict) -> Dict:
        """Mutate position, orientation, or torsions (one of the three)."""
        mutation_choice = np.random.random()

        if mutation_choice < 0.33:
            # Mutate position (small Gaussian step)
            individual['position'] += np.random.normal(0, 0.5, 3)

        elif mutation_choice < 0.66:
            # Mutate orientation
            individual['orientation'] += np.random.normal(0, 10, 3)
            individual['orientation'] = individual['orientation'] % 360

        else:
            # Mutate torsions
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
            logger.info(f"Best position (Å): X={p[0]:.2f}, Y={p[1]:.2f}, Z={p[2]:.2f}")
            logger.info(f"Best orientation (°): α={o[0]:.1f}, β={o[1]:.1f}, γ={o[2]:.1f}")
            if self.n_torsions > 0:
                logger.info(
                    "Best torsions (°): "
                    + " ".join(f"{t:.1f}" for t in self.best_individual['torsions'])
                )

        return results