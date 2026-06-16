"""
Window 2: Reference Energy Calculation for GOAD v1.0

Calculates baseline energies for surface and molecule
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from ase.io import write
from ase.optimize import BFGS
from ase.constraints import FixAtoms
import os
from datetime import datetime
import logging
from pathlib import Path

from ..utils.calculator_manager import CalculatorManager

logger = logging.getLogger(__name__)


class ReferenceEnergiesWindow:
    """Window for reference energy calculation"""

    def __init__(self, root, surface, molecule, surface_analyzer, molecule_analyzer,
                 n_fixed_layers, n_total_layers, on_complete_callback=None):
        """
        Initialize reference energies window.

        Args:
            root: Tkinter root window
            surface: ASE Atoms object (surface)
            molecule: ASE Atoms object (molecule)
            surface_analyzer: SurfaceAnalyzer instance
            molecule_analyzer: MoleculeAnalyzer instance
            n_fixed_layers: Number of layers to keep fixed
            n_total_layers: Total number of layers
            on_complete_callback: Function to call when complete
        """
        self.root = root
        self.root.title("GOAD v1.0 - Reference Energies")
        self.root.geometry("900x600")

        self.surface = surface
        self.molecule = molecule
        self.surface_analyzer = surface_analyzer
        self.molecule_analyzer = molecule_analyzer
        self.n_fixed_layers = n_fixed_layers
        self.n_total_layers = n_total_layers
        self.on_complete_callback = on_complete_callback

        # Calculator (will be set later or passed in)
        self.calculator = None

        # Results storage
        self.surface_energy = None
        self.molecule_energy = None
        self.surface_relaxed = None
        self.molecule_relaxed = None
        self.optimize_reference = True

        # Build GUI
        self._build_gui()

    def _build_gui(self):
        """Build the GUI layout"""

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(
            main_frame,
            text="GOAD v1.0 - Reference Energy Calculation",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(0, 20))

        # Left panel: Options
        left_panel = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        # Summary info
        info_frame = ttk.Frame(left_panel)
        info_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(info_frame, text="Setup Summary:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Fixed layers: {self.n_fixed_layers}/{self.n_total_layers}").pack(anchor=tk.W)
        self.summary_calc_label = ttk.Label(
            info_frame,
            text=f"Calculator: MatterSim 1M (default)",
            foreground="blue"
        )
        self.summary_calc_label.pack(anchor=tk.W)

        # Calculator selection
        ttk.Label(left_panel, text="Calculator:", font=("Arial", 10, "bold")).pack(
            anchor=tk.W, pady=(10, 5))

        self.calculator_var = tk.StringVar(value="1m")

        calc_frame = ttk.Frame(left_panel)
        calc_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(
            calc_frame,
            text="MatterSim 1M (Fast)",
            variable=self.calculator_var,
            value="1m",
            command=self._update_calculator_info
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            calc_frame,
            text="MatterSim 5M (Balanced)",
            variable=self.calculator_var,
            value="5m",
            command=self._update_calculator_info
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            calc_frame,
            text="MatterSim 5M + D3 (Accurate)",
            variable=self.calculator_var,
            value="5m_d3",
            command=self._update_calculator_info
        ).pack(anchor=tk.W, pady=2)

        self.calc_info_label = ttk.Label(
            left_panel,
            text="",
            font=("Arial", 8),
            foreground="gray",
            justify=tk.LEFT
        )
        self.calc_info_label.pack(anchor=tk.W, pady=(5, 10))
        self._update_calculator_info()

        # Relaxation option
        ttk.Label(left_panel, text="Reference Structures:", font=("Arial", 10, "bold")).pack(
            anchor=tk.W, pady=(10, 5))

        self.relax_var = tk.BooleanVar(value=True)

        ttk.Radiobutton(
            left_panel,
            text="Relax both (0-1000 steps)",
            variable=self.relax_var,
            value=True
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            left_panel,
            text="Use as-is (single point)",
            variable=self.relax_var,
            value=False
        ).pack(anchor=tk.W, pady=2)

        ttk.Label(
            left_panel,
            text="• Surface: respects fixed layers\n"
                 "• Molecule: fully relaxed",
            font=("Arial", 9),
            foreground="gray",
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(10, 0))

        # Calculate button
        ttk.Button(
            left_panel,
            text="Calculate References",
            command=self._calculate_references
        ).pack(fill=tk.X, pady=(20, 0))

        # Next button
        self.next_button = ttk.Button(
            left_panel,
            text="Next: Genetic Algorithm",
            command=self._go_to_ga,
            state=tk.DISABLED
        )
        self.next_button.pack(fill=tk.X, pady=(10, 0))

        # Right panel: Results
        right_panel = ttk.LabelFrame(main_frame, text="Calculation Results", padding="10")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Notebook for results
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Log tab
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="📋 Log")

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=25,
            width=60,
            wrap=tk.WORD,
            font=('Courier', 9),
            state='disabled'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Results tab
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="⚡ Energies")

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=25,
            width=60,
            wrap=tk.WORD,
            font=('Courier', 10),
            state='disabled'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def _update_calculator_info(self):
        """Update calculator info label"""
        calc_type = self.calculator_var.get()
        info = CalculatorManager.get_calculator_info(calc_type)

        if info:
            info_text = f"{info['description']}\nSpeed: {info['speed']} | Accuracy: {info['accuracy']}"
            self.calc_info_label.config(text=info_text)

    def _log(self, message: str):
        """Add message to log"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        logger.info(message)

    def _calculate_references(self):
        """Calculate reference energies"""
        # Load selected calculator
        try:
            calc_type = self.calculator_var.get()
            self._log(f"Loading calculator: {calc_type.upper()}...")
            self.calculator = CalculatorManager.get_calculator(calc_type)
            self._log("✓ Calculator loaded successfully\n")
        except Exception as e:
            messagebox.showerror(
                "Calculator Error",
                f"Could not load calculator:\n{e}"
            )
            self._log(f"ERROR loading calculator: {e}")
            return

        self._log("=" * 60)
        self._log("REFERENCE ENERGY CALCULATION")
        self._log("=" * 60)

        try:
            self.optimize_reference = self.relax_var.get()

            if self.optimize_reference:
                self._log("\nMode: RELAXATION (0-1000 steps max)")
                self._log("• Surface: relaxed with fixed layers")
                self._log("• Molecule: fully relaxed\n")
                self._calculate_with_relaxation()
            else:
                self._log("\nMode: SINGLE POINT (as-is)")
                self._log("• Surface: no optimization")
                self._log("• Molecule: no optimization\n")
                self._calculate_single_point()

            self._display_results()
            self.next_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Calculation Error", f"Error during calculation:\n{e}")
            self._log(f"ERROR: {e}")
            logger.error(f"Calculation error: {e}")

    def _calculate_with_relaxation(self):
        """Calculate with relaxation"""
        self._log("\n[1/2] Calculating surface reference...")

        # Surface copy with fixed layers
        surface_copy = self.surface.copy()
        surface_copy.set_calculator(self.calculator)

        # Fix the specified number of layers from bottom
        self._log(f"  Fixing bottom {self.n_fixed_layers} layers...")
        fixed_indices = self._get_fixed_layer_indices()

        if fixed_indices:
            surface_copy.set_constraint(FixAtoms(indices=fixed_indices))

        # Relax
        self._log("  Optimizing surface...")
        opt = BFGS(surface_copy, logfile=None)
        opt.run(fmax=0.05, steps=1000)

        self.surface_energy = surface_copy.get_potential_energy()
        self.surface_relaxed = surface_copy
        self._log(f"  ✓ E_surface = {self.surface_energy:.4f} eV")

        self._log("\n[2/2] Calculating molecule reference...")
        molecule_copy = self.molecule.copy()
        molecule_copy.set_calculator(self.calculator)

        self._log("  Optimizing molecule (fully free)...")
        opt = BFGS(molecule_copy, logfile=None)
        opt.run(fmax=0.05, steps=1000)

        self.molecule_energy = molecule_copy.get_potential_energy()
        self.molecule_relaxed = molecule_copy
        self._log(f"  ✓ E_molecule = {self.molecule_energy:.4f} eV")

    def _calculate_single_point(self):
        """Calculate single point energies - NO relaxation, pure energy calculation"""
        from ase.constraints import FixAtoms
        import numpy as np

        self._log("\n[1/2] Calculating surface single-point (FIXED)...")

        surface_copy = self.surface.copy()
        surface_original_pos = surface_copy.get_positions().copy()

        # Fix ALL atoms to prevent any movement/relaxation
        surface_copy.set_constraint(FixAtoms(indices=range(len(surface_copy))))
        surface_copy.set_calculator(self.calculator)
        self.surface_energy = surface_copy.get_potential_energy()
        surface_final_pos = surface_copy.get_positions()

        # Verify no movement
        pos_change_surface = np.max(np.abs(surface_final_pos - surface_original_pos))
        self._log(f"  Position change: {pos_change_surface:.4e} Å")

        self.surface_relaxed = surface_copy
        self._log(f"  E_surface = {self.surface_energy:.4f} eV")

        self._log("\n[2/2] Calculating molecule single-point (FIXED)...")

        molecule_copy = self.molecule.copy()
        molecule_original_pos = molecule_copy.get_positions().copy()

        # Fix ALL atoms to prevent any movement/relaxation
        molecule_copy.set_constraint(FixAtoms(indices=range(len(molecule_copy))))
        molecule_copy.set_calculator(self.calculator)
        self.molecule_energy = molecule_copy.get_potential_energy()
        molecule_final_pos = molecule_copy.get_positions()

        # Verify no movement
        pos_change_molecule = np.max(np.abs(molecule_final_pos - molecule_original_pos))
        self._log(f"  Position change: {pos_change_molecule:.4e} Å")

        self.molecule_relaxed = molecule_copy
        self._log(f"  E_molecule = {self.molecule_energy:.4f} eV")

    def _get_fixed_layer_indices(self) -> list:
        """Get indices of atoms to fix (bottom N layers)"""
        layers_info = self.surface_analyzer._info.get('layers', {}).get('layers_list', [])

        # Get the bottom N layers
        fixed_indices = []
        for i in range(self.n_total_layers - self.n_fixed_layers, self.n_total_layers):
            if i < len(layers_info):
                fixed_indices.extend(layers_info[i]['atom_indices'])

        return fixed_indices

    def _display_results(self):
        """Display results in the results tab"""
        text = "=" * 60 + "\n"
        text += "REFERENCE ENERGIES\n"
        text += "=" * 60 + "\n\n"

        text += f"Calculation Mode: {'Relaxation' if self.optimize_reference else 'Single Point'}\n"
        text += f"Fixed Layers: {self.n_fixed_layers}/{self.n_total_layers}\n\n"

        text += "RESULTS:\n"
        text += "-" * 60 + "\n"
        text += f"E_surface:        {self.surface_energy:>12.4f} eV\n"
        text += f"E_molecule:       {self.molecule_energy:>12.4f} eV\n"
        text += f"\nTotal reference:  {self.surface_energy + self.molecule_energy:>12.4f} eV\n"
        text += "=" * 60 + "\n\n"

        text += "These energies will be used to calculate:\n"
        text += "E_adsorption = E_system - (E_surface + E_molecule)\n"

        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        self.results_text.config(state='disabled')

    def _go_to_ga(self):
        """Proceed to genetic algorithm"""
        if self.surface_energy is None or self.molecule_energy is None:
            messagebox.showwarning("Incomplete", "Calculate reference energies first")
            return

        if self.on_complete_callback:
            self.on_complete_callback(
                surface_relaxed=self.surface_relaxed,
                molecule_relaxed=self.molecule_relaxed,
                surface_energy=self.surface_energy,
                molecule_energy=self.molecule_energy,
                n_fixed_layers=self.n_fixed_layers,
                calculator=self.calculator
            )

    def set_calculator(self, calculator):
        """Set the calculator to use"""
        self.calculator = calculator
