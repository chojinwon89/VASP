"""
Molecule analysis for GOAD v1.0

Provides detailed information about molecular properties
"""

import numpy as np
from ase import Atoms
from ase.data import atomic_masses
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class MoleculeAnalyzer:
    """Analyze molecule structure and properties"""

    def __init__(self, molecule: Atoms):
        """
        Initialize molecule analyzer.

        Args:
            molecule: ASE Atoms object with molecular structure
        """
        self.molecule = molecule
        self._info = None

    def analyze(self) -> Dict:
        """
        Perform complete molecule analysis.

        Returns:
            Dictionary with molecule information
        """
        logger.info("Analyzing molecule structure...")

        info = {
            "n_atoms": len(self.molecule),
            "formula": self._get_formula(),
            "elements": self._get_composition(),
            "mass": self._get_mass(),
            "size_category": self._categorize_size(),
            "dimensions": self._get_dimensions(),
            "center_of_mass": self._get_center_of_mass(),
        }

        self._info = info
        return info

    def _get_formula(self) -> str:
        """Get molecular formula"""
        symbols = self.molecule.get_chemical_symbols()
        composition = {}

        for symbol in symbols:
            composition[symbol] = composition.get(symbol, 0) + 1

        # Format formula
        formula = ""
        for element in sorted(composition.keys()):
            count = composition[element]
            if count == 1:
                formula += element
            else:
                formula += f"{element}{count}"

        return formula

    def _get_composition(self) -> Dict[str, int]:
        """Get element composition"""
        composition = {}
        for symbol in self.molecule.get_chemical_symbols():
            composition[symbol] = composition.get(symbol, 0) + 1
        return composition

    def _get_mass(self) -> float:
        """Get molecular mass in amu"""
        from ase.data import atomic_numbers

        symbols = self.molecule.get_chemical_symbols()
        mass = 0
        for symbol in symbols:
            # Get atomic number from symbol, then get mass
            atomic_num = atomic_numbers[symbol]
            mass += atomic_masses[atomic_num]
        return mass

    def _categorize_size(self) -> str:
        """Categorize molecule size"""
        n_atoms = len(self.molecule)

        if n_atoms <= 3:
            return "Very small (diatomic/triatomic)"
        elif n_atoms <= 10:
            return "Small"
        elif n_atoms <= 30:
            return "Medium"
        elif n_atoms <= 100:
            return "Large"
        else:
            return "Very large"

    def _get_dimensions(self) -> Dict[str, float]:
        """Get molecular dimensions"""
        positions = self.molecule.get_positions()

        return {
            "x": positions[:, 0].max() - positions[:, 0].min(),
            "y": positions[:, 1].max() - positions[:, 1].min(),
            "z": positions[:, 2].max() - positions[:, 2].min(),
        }

    def _get_center_of_mass(self) -> np.ndarray:
        """Get center of mass"""
        return self.molecule.get_center_of_mass()

    def get_info_text(self) -> str:
        """
        Get formatted text with molecule information.

        Returns:
            Formatted string
        """
        if not self._info:
            return "No analysis performed"

        info = self._info
        text = "=" * 60 + "\n"
        text += "MOLECULE INFORMATION\n"
        text += "=" * 60 + "\n\n"

        text += f"Molecular formula: {info['formula']}\n"
        text += f"Number of atoms: {info['n_atoms']}\n"
        text += f"Size category: {info['size_category']}\n"
        text += f"Molecular mass: {info['mass']:.2f} amu\n\n"

        text += "ELEMENTAL COMPOSITION:\n"
        for element, count in sorted(info['elements'].items()):
            percentage = (count / info['n_atoms']) * 100
            text += f"  {element}: {count} atoms ({percentage:.1f}%)\n"

        text += f"\nMOLECULAR DIMENSIONS:\n"
        text += f"  X: {info['dimensions']['x']:.2f} Å\n"
        text += f"  Y: {info['dimensions']['y']:.2f} Å\n"
        text += f"  Z: {info['dimensions']['z']:.2f} Å\n"

        com = info['center_of_mass']
        text += f"\nCenter of mass:\n"
        text += f"  X: {com[0]:.2f} Å\n"
        text += f"  Y: {com[1]:.2f} Å\n"
        text += f"  Z: {com[2]:.2f} Å\n"

        return text
