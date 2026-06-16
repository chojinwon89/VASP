"""
Surface type detection and analysis for GOAD v1.0

Identifies slab vs porous structures and analyzes layer properties
"""

import numpy as np
from ase import Atoms
from ase.geometry import get_distances
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SurfaceAnalyzer:
    """Analyze surface structure and detect slab vs porous"""

    def __init__(self, surface: Atoms, vacuum_threshold: float = 5.0):
        """
        Initialize surface analyzer.

        Args:
            surface: ASE Atoms object with surface structure
            vacuum_threshold: Minimum distance to detect vacuum (Angstrom)
        """
        self.surface = surface
        self.vacuum_threshold = vacuum_threshold
        self._info = None
        self.surface_type = None
        self.n_layers = None
        self.layer_heights = None

    def analyze(self) -> Dict:
        """
        Perform complete surface analysis.

        Returns:
            Dictionary with surface information
        """
        logger.info("Analyzing surface structure...")

        # Detect surface type
        self._detect_surface_type()

        # Get basic properties
        info = {
            "n_atoms": len(self.surface),
            "surface_type": self.surface_type,
            "elements": self._get_composition(),
            "dimensions": self._get_dimensions(),
            "surface_area": self._get_surface_area(),
        }

        # If slab, analyze layers
        if self.surface_type == "slab":
            info["layers"] = self._analyze_layers()
        else:
            info["message"] = "Porous structures (MOF/Zeolite) are under development"

        self._info = info
        return info

    def _detect_surface_type(self) -> None:
        """Detect if surface is slab or porous"""
        z_coords = self.surface.get_positions()[:, 2]
        z_min = z_coords.min()
        z_max = z_coords.max()
        z_range = z_max - z_min

        # Get cell height
        cell_height = self.surface.get_cell()[2, 2]

        # Calculate vacuum
        vacuum = cell_height - z_range

        logger.info(f"  Z-range: {z_range:.2f} Å")
        logger.info(f"  Cell height: {cell_height:.2f} Å")
        logger.info(f"  Vacuum: {vacuum:.2f} Å")

        if vacuum >= self.vacuum_threshold:
            self.surface_type = "slab"
            logger.info("  Type: SLAB (detected)")
        else:
            self.surface_type = "porous"
            logger.info("  Type: POROUS/MOF/Zeolite (detected)")

    def _analyze_layers(self) -> Dict:
        """
        Analyze atomic layers for slab surfaces.

        Returns:
            Dictionary with layer information
        """
        positions = self.surface.get_positions()
        z_coords = positions[:, 2]

        n_atoms = len(z_coords)

        # Cluster z-coordinates into layers
        z_unique = []
        z_tolerance = 0.3  # Atoms within 0.3 Å are in same layer

        # Get sorted indices
        sorted_indices = np.argsort(z_coords)
        sorted_z = z_coords[sorted_indices]

        # Cluster atoms by z-coordinate
        layer_z_values = []
        for z in sorted_z:
            # Check if z belongs to existing cluster
            belongs_to_layer = False
            for z_ref in layer_z_values:
                if abs(z - z_ref) < z_tolerance:
                    belongs_to_layer = True
                    break

            if not belongs_to_layer:
                layer_z_values.append(z)

        # Sort z-values from top to bottom (highest Z first)
        layer_z_values = sorted(layer_z_values, reverse=True)
        n_layers = len(layer_z_values)

        logger.info(f"  Detected {n_layers} atomic layers")
        logger.info(f"  Layer heights (Å): {[f'{z:.2f}' for z in layer_z_values]}")

        # Build layer information
        layer_info = {
            "n_layers": n_layers,
            "layer_positions": layer_z_values,
            "total_height": layer_z_values[0] - layer_z_values[-1] if n_layers > 1 else 0,
            "layers_list": []
        }

        # Assign atoms to layers
        for layer_idx, z_ref in enumerate(layer_z_values):
            layer_atoms = []

            for atom_idx in range(n_atoms):
                if abs(z_coords[atom_idx] - z_ref) < z_tolerance:
                    layer_atoms.append(atom_idx)

            layer_info["layers_list"].append({
                "layer_number": layer_idx,
                "z_position": z_ref,
                "n_atoms": len(layer_atoms),
                "atom_indices": layer_atoms
            })

        return layer_info

    def _get_composition(self) -> Dict[str, int]:
        """Get element composition"""
        composition = {}
        for symbol in self.surface.get_chemical_symbols():
            composition[symbol] = composition.get(symbol, 0) + 1
        return composition

    def _get_dimensions(self) -> Dict[str, float]:
        """Get structure dimensions"""
        positions = self.surface.get_positions()
        return {
            "x": positions[:, 0].max() - positions[:, 0].min(),
            "y": positions[:, 1].max() - positions[:, 1].min(),
            "z": positions[:, 2].max() - positions[:, 2].min(),
        }

    def _get_surface_area(self) -> float:
        """Get XY surface area"""
        cell = self.surface.get_cell()
        a = np.linalg.norm(cell[0])
        b = np.linalg.norm(cell[1])
        return a * b

    def get_info_text(self) -> str:
        """
        Get formatted text with surface information.

        Returns:
            Formatted string
        """
        if not self._info:
            return "No analysis performed"

        info = self._info
        text = "=" * 60 + "\n"
        text += "SURFACE INFORMATION\n"
        text += "=" * 60 + "\n\n"

        text += f"Surface Type: {info['surface_type'].upper()}\n"
        text += f"Number of atoms: {info['n_atoms']}\n"
        text += f"Composition: {', '.join([f'{el}: {count}' for el, count in info['elements'].items()])}\n\n"

        text += "SYSTEM DIMENSIONS:\n"
        text += f"  X: {info['dimensions']['x']:.2f} Å\n"
        text += f"  Y: {info['dimensions']['y']:.2f} Å\n"
        text += f"  Z: {info['dimensions']['z']:.2f} Å\n"
        text += f"  Surface area (XY): {info['surface_area']:.2f} Ų\n\n"

        if info['surface_type'] == 'slab' and 'layers' in info:
            layers = info['layers']
            text += f"ATOMIC LAYERS: {layers['n_layers']} layers detected\n"
            text += f"  Total height: {layers['total_height']:.2f} Å\n\n"
            text += "Layer positions (from top to bottom):\n"
            for layer in layers['layers_list']:
                text += f"  Layer {layer['layer_number']}: "
                text += f"Z = {layer['z_position']:.2f} Å, "
                text += f"Atoms = {layer['n_atoms']}\n"
        else:
            text += info.get('message', '') + "\n"

        return text

    def is_slab(self) -> bool:
        """Check if surface is slab type"""
        if self.surface_type is None:
            self._detect_surface_type()
        return self.surface_type == "slab"

    def get_layer_count(self) -> Optional[int]:
        """Get number of atomic layers (slab only)"""
        if not self.is_slab():
            return None

        if self._info is None:
            self.analyze()

        return self._info['layers']['n_layers'] if 'layers' in self._info else None

    def get_layer_atom_indices(self, layer_num: int) -> List[int]:
        """Get atom indices for a specific layer"""
        if not self.is_slab() or self._info is None:
            return []

        layers = self._info.get('layers', {}).get('layers_list', [])
        for layer in layers:
            if layer['layer_number'] == layer_num:
                return layer['atom_indices']

        return []
