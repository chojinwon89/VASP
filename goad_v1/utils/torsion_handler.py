"""
Torsion handling for molecular optimization in GOAD v1.0

Detects rotatable bonds and applies torsion angles to molecules
"""

import numpy as np
from ase import Atoms
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TorsionHandler:
    """Detect and manipulate molecular torsions"""

    def __init__(self, molecule: Atoms):
        """
        Initialize torsion handler.

        Args:
            molecule: ASE Atoms object (molecule)
        """
        self.molecule = molecule
        self.rotatable_bonds = []
        self.torsion_angles = []
        self.n_torsions = 0

        # Detect torsions using RDKit if available
        self._detect_torsions_rdkit()

    def _detect_torsions_rdkit(self):
        """
        Detect rotatable bonds using RDKit.
        Build RDKit mol from an XYZ block (works reliably for any small molecule).
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, rdDetermineBonds
        except ImportError:
            logger.warning("RDKit not available, torsion detection disabled")
            self.rotatable_bonds = []
            self.n_torsions = 0
            return

        try:
            # 1) Make an XYZ string from the ASE Atoms (no temp file needed)
            symbols = self.molecule.get_chemical_symbols()
            positions = self.molecule.get_positions()
            xyz_lines = [str(len(symbols)), "from_ase"]
            for s, (x, y, z) in zip(symbols, positions):
                xyz_lines.append(f"{s} {x:.6f} {y:.6f} {z:.6f}")
            xyz_block = "\n".join(xyz_lines)

            # 2) Parse with RDKit and perceive bonds from geometry
            mol = Chem.MolFromXYZBlock(xyz_block)
            if mol is None:
                logger.warning("RDKit could not parse XYZ; skipping torsion detection")
                self.rotatable_bonds = []
                self.n_torsions = 0
                return

            mol = Chem.RWMol(mol)
            try:
                rdDetermineBonds.DetermineConnectivity(mol)
                rdDetermineBonds.DetermineBondOrders(mol, charge=0)
            except Exception as e:
                # Fall back: connectivity only (no bond orders) -> still gives single bonds
                logger.warning(f"DetermineBondOrders failed ({e}); using connectivity only")
                try:
                    rdDetermineBonds.DetermineConnectivity(mol)
                except Exception:
                    pass

            # 3) Identify rotatable bonds (single, acyclic, not to H, both ends > 1 neighbor)
            rotatable_bonds = []
            for bond in mol.GetBonds():
                if bond.IsInRing():
                    continue
                if bond.GetBondType() != Chem.BondType.SINGLE:
                    continue
                a = bond.GetBeginAtom()
                b = bond.GetEndAtom()
                if a.GetAtomicNum() == 1 or b.GetAtomicNum() == 1:
                    continue
                if a.GetDegree() < 2 or b.GetDegree() < 2:
                    continue
                rotatable_bonds.append((a.GetIdx(), b.GetIdx()))

            self.rotatable_bonds = rotatable_bonds
            self.n_torsions = len(rotatable_bonds)
            self.torsion_angles = [0.0] * self.n_torsions

            logger.info(f"Detected {self.n_torsions} rotatable bonds:")
            for i, (b, e) in enumerate(rotatable_bonds):
                logger.info(f"  Torsion {i}: atoms {b}-{e}")

        except Exception as e:
            logger.warning(f"Error detecting torsions: {e}")
            self.rotatable_bonds = []
            self.n_torsions = 0

    def apply_torsions(self, molecule_copy: Atoms, torsion_angles: List[float]) -> Atoms:
        """
        Apply torsion angles to a molecule copy.

        Args:
            molecule_copy: Copy of molecule to rotate
            torsion_angles: List of torsion angles in degrees

        Returns:
            Modified molecule with torsions applied
        """
        if len(torsion_angles) != self.n_torsions:
            logger.warning(f"Torsion angle count mismatch: {len(torsion_angles)} vs {self.n_torsions}")
            return molecule_copy

        if self.n_torsions == 0:
            return molecule_copy

        # Apply each torsion
        for i, (torsion_angle, (bond_begin, bond_end)) in enumerate(
            zip(torsion_angles, self.rotatable_bonds)
        ):
            molecule_copy = self._apply_single_torsion(
                molecule_copy, bond_begin, bond_end, torsion_angle
            )

        return molecule_copy

    def _apply_single_torsion(self, atoms: Atoms, bond_begin: int, bond_end: int,
                             angle_degrees: float) -> Atoms:
        """
        Apply a single torsion rotation.

        Rotates atoms connected to bond_end around the bond_begin-bond_end axis.

        Args:
            atoms: Molecule to modify
            bond_begin: Index of first atom in bond
            bond_end: Index of second atom in bond
            angle_degrees: Rotation angle in degrees

        Returns:
            Modified molecule
        """
        # Convert angle to radians
        angle_rad = np.deg2rad(angle_degrees)

        positions = atoms.get_positions()

        # Get bond axis
        axis = positions[bond_end] - positions[bond_begin]
        axis = axis / np.linalg.norm(axis)

        # Find atoms connected to bond_end (excluding bond_begin)
        try:
            from ase.neighborlist import neighbor_list
            i, j = neighbor_list('ij', atoms, cutoff=1.6)

            # Find neighbors of bond_end
            neighbors_to_rotate = []
            for idx_pair in range(len(i)):
                if i[idx_pair] == bond_end and j[idx_pair] != bond_begin:
                    neighbors_to_rotate.append(j[idx_pair])
                elif j[idx_pair] == bond_end and i[idx_pair] != bond_begin:
                    neighbors_to_rotate.append(i[idx_pair])

        except:
            logger.warning("Could not determine connected atoms for torsion")
            return atoms

        # Build set of atoms to rotate (connected to bond_end)
        atoms_to_rotate = set(neighbors_to_rotate)
        to_check = list(neighbors_to_rotate)

        while to_check:
            current = to_check.pop(0)
            try:
                from ase.neighborlist import neighbor_list
                i, j = neighbor_list('ij', atoms, cutoff=1.6)

                for idx_pair in range(len(i)):
                    if i[idx_pair] == current and j[idx_pair] != bond_begin and j[idx_pair] != bond_end:
                        if j[idx_pair] not in atoms_to_rotate:
                            atoms_to_rotate.add(j[idx_pair])
                            to_check.append(j[idx_pair])
                    elif j[idx_pair] == current and i[idx_pair] != bond_begin and i[idx_pair] != bond_end:
                        if i[idx_pair] not in atoms_to_rotate:
                            atoms_to_rotate.add(i[idx_pair])
                            to_check.append(i[idx_pair])
            except:
                break

        # Apply rotation to selected atoms
        rotation_matrix = self._rotation_matrix_axis_angle(axis, angle_rad)

        for atom_idx in atoms_to_rotate:
            # Translate to bond_end as origin
            relative_pos = positions[atom_idx] - positions[bond_end]

            # Rotate
            rotated_pos = relative_pos @ rotation_matrix.T

            # Translate back
            positions[atom_idx] = rotated_pos + positions[bond_end]

        atoms.set_positions(positions)
        return atoms

    @staticmethod
    def _rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Create rotation matrix from axis and angle (Rodrigues' formula).

        Args:
            axis: Unit vector (rotation axis)
            angle: Rotation angle in radians

        Returns:
            3x3 rotation matrix
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Skew-symmetric cross-product matrix
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # Rodrigues' rotation formula: R = I + sin(θ)K + (1-cos(θ))K²
        R = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)   # '*' not '@'
        return R

    def get_info(self) -> Dict:
        """
        Get torsion information.

        Returns:
            Dictionary with torsion info
        """
        return {
            'n_torsions': self.n_torsions,
            'rotatable_bonds': self.rotatable_bonds,
            'current_angles': self.torsion_angles,
        }

    def get_torsion_range(self) -> Tuple[List[float], List[float]]:
        """
        Get torsion angle ranges.

        Returns:
            Tuple of (min_angles, max_angles) in degrees
        """
        # Torsions can rotate 0-360 degrees
        min_angles = [0.0] * self.n_torsions
        max_angles = [360.0] * self.n_torsions

        return min_angles, max_angles
