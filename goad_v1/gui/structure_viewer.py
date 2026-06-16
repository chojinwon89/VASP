"""
3D Structure viewer for GOAD v1.0

Displays surface and molecule structures using matplotlib
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from ase import Atoms
from ase.data import colors


class StructureViewer:
    """3D structure viewer widget"""

    def __init__(self, parent_frame, width=6, height=5, dpi=100):
        """
        Initialize structure viewer.

        Args:
            parent_frame: Tkinter frame to embed viewer in
            width: Figure width in inches
            height: Figure height in inches
            dpi: Figure DPI
        """
        self.parent = parent_frame
        self.current_structure = None

        # Create matplotlib figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Colors for atoms
        self.atom_colors = {
            'H': '#FFFFFF',
            'C': '#909090',
            'N': '#3050F8',
            'O': '#FF0D0D',
            'S': '#FFFF30',
            'P': '#FF8000',
            'Cu': '#B87333',
            'Pt': '#E195B5',
            'Au': '#FFD700',
            'Ag': '#C0C0C0',
            'Al': '#C8C8C8',
        }

    def display_structure(self, structure: Atoms, title: str = "Structure"):
        """
        Display a structure.

        Args:
            structure: ASE Atoms object
            title: Title for the plot
        """
        self.current_structure = structure

        self.ax.clear()

        positions = structure.get_positions()
        symbols = structure.get_chemical_symbols()

        # Plot atoms
        for symbol, pos in zip(symbols, positions):
            color = self.atom_colors.get(symbol, '#FF00FF')

            # Atom size based on type
            size = 100 if symbol != 'H' else 50

            self.ax.scatter(
                [pos[0]], [pos[1]], [pos[2]],
                c=color,
                s=size,
                edgecolors='black',
                linewidth=0.5,
                alpha=0.8
            )

        # Set labels and title
        self.ax.set_xlabel('X (Å)')
        self.ax.set_ylabel('Y (Å)')
        self.ax.set_zlabel('Z (Å)')
        self.ax.set_title(title, fontsize=12, fontweight='bold')

        # Set aspect ratio
        positions = structure.get_positions()
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        z_range = positions[:, 2].max() - positions[:, 2].min()

        max_range = max(x_range, y_range, z_range)

        center_x = positions[:, 0].mean()
        center_y = positions[:, 1].mean()
        center_z = positions[:, 2].mean()

        margin = max_range * 0.1
        self.ax.set_xlim(center_x - max_range/2 - margin, center_x + max_range/2 + margin)
        self.ax.set_ylim(center_y - max_range/2 - margin, center_y + max_range/2 + margin)
        self.ax.set_zlim(center_z - max_range/2 - margin, center_z + max_range/2 + margin)

        # Rotate view slightly for better visualization
        self.ax.view_init(elev=20, azim=45)

        self.fig.tight_layout()
        self.canvas.draw()

    def display_combined(self, surface: Atoms, molecule: Atoms, title: str = "Combined Structure"):
        """
        Display surface and molecule together.

        Args:
            surface: ASE Atoms object (surface)
            molecule: ASE Atoms object (molecule)
            title: Title for the plot
        """
        # Combine for visualization
        combined = surface + molecule

        # Mark surface atoms with larger size
        self.ax.clear()

        positions = combined.get_positions()
        symbols = combined.get_chemical_symbols()

        surface_count = len(surface)

        # Plot surface atoms
        for i in range(surface_count):
            symbol = symbols[i]
            pos = positions[i]
            color = self.atom_colors.get(symbol, '#FF00FF')

            self.ax.scatter(
                [pos[0]], [pos[1]], [pos[2]],
                c=color,
                s=200,  # Larger for surface
                edgecolors='black',
                linewidth=1,
                alpha=0.9,
                marker='o'
            )

        # Plot molecule atoms
        for i in range(surface_count, len(combined)):
            symbol = symbols[i]
            pos = positions[i]
            color = self.atom_colors.get(symbol, '#FF00FF')

            self.ax.scatter(
                [pos[0]], [pos[1]], [pos[2]],
                c=color,
                s=100,  # Smaller for molecule
                edgecolors='red',
                linewidth=0.5,
                alpha=0.8,
                marker='o'
            )

        # Draw bonds in molecule
        try:
            from ase.neighborlist import neighbor_list
            i_arr, j_arr = neighbor_list('ij', combined, cutoff=1.6)

            for idx_pair in range(len(i_arr)):
                atom_i = int(i_arr[idx_pair])
                atom_j = int(j_arr[idx_pair])

                # Only draw bonds within molecule or between molecule atoms
                if atom_i >= surface_count or atom_j >= surface_count:
                    if atom_i >= surface_count and atom_j >= surface_count:
                        pos_i = positions[atom_i]
                        pos_j = positions[atom_j]

                        self.ax.plot(
                            [pos_i[0], pos_j[0]],
                            [pos_i[1], pos_j[1]],
                            [pos_i[2], pos_j[2]],
                            'k-',
                            linewidth=0.5,
                            alpha=0.5
                        )
        except Exception as e:
            pass  # Bonds not critical for visualization

        # Labels and title
        self.ax.set_xlabel('X (Å)')
        self.ax.set_ylabel('Y (Å)')
        self.ax.set_zlabel('Z (Å)')
        self.ax.set_title(title, fontsize=12, fontweight='bold')

        # Set aspect ratio
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        z_range = positions[:, 2].max() - positions[:, 2].min()

        max_range = max(x_range, y_range, z_range)

        center_x = positions[:, 0].mean()
        center_y = positions[:, 1].mean()
        center_z = positions[:, 2].mean()

        margin = max_range * 0.1
        self.ax.set_xlim(center_x - max_range/2 - margin, center_x + max_range/2 + margin)
        self.ax.set_ylim(center_y - max_range/2 - margin, center_y + max_range/2 + margin)
        self.ax.set_zlim(center_z - max_range/2 - margin, center_z + max_range/2 + margin)

        self.ax.view_init(elev=20, azim=45)

        self.fig.tight_layout()
        self.canvas.draw()

    def rotate_view(self, azim: int = 0, elev: int = 0):
        """Rotate view"""
        if self.current_structure is not None:
            self.ax.view_init(elev=elev, azim=azim)
            self.canvas.draw()
