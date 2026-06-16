"""
Window 4: Final Optimization for GOAD v1.0

Final full relaxation respecting fixed layers
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.io import write
import os
from datetime import datetime
import threading
import logging

logger = logging.getLogger(__name__)


class FinalOptimizationWindow:
    """Window for final structure optimization"""

    def __init__(self, root, best_structure, best_energy, n_fixed_layers,
                 surface_analyzer, calculator, output_dir="results"):
        """
        Initialize final optimization window.

        Args:
            root: Tkinter root window
            best_structure: Best structure from GA
            best_energy: Best energy from GA
            n_fixed_layers: Number of fixed layers
            surface_analyzer: SurfaceAnalyzer instance
            calculator: ASE calculator
            output_dir: Output directory for results
        """
        self.root = root
        self.root.title("GOAD v1.0 - Final Optimization")
        self.root.geometry("900x700")

        self.best_structure = best_structure
        self.best_energy = best_energy
        self.n_fixed_layers = n_fixed_layers
        self.surface_analyzer = surface_analyzer
        self.calculator = calculator
        self.output_dir = output_dir

        # Optimization state
        self.optimize_final = True
        self.running = False
        self.opt_thread = None
        self.optimized_structure = None
        self.final_energy = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Build GUI
        self._build_gui()

    def _build_gui(self):
        """Build the GUI layout"""

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(
            main_frame,
            text="GOAD v1.0 - Final Optimization",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(0, 20))

        # Left panel: Options and controls
        left_panel = ttk.LabelFrame(main_frame, text="Final Step", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10), width=300)

        # Summary
        summary_frame = ttk.Frame(left_panel)
        summary_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(summary_frame, text="GA Results:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(summary_frame, text=f"E_adsorption: {self.best_energy:.4f} eV").pack(anchor=tk.W)
        ttk.Label(summary_frame, text=f"Fixed layers: {self.n_fixed_layers}").pack(anchor=tk.W)

        # Options
        ttk.Label(left_panel, text="What to do:", font=("Arial", 10, "bold")).pack(
            anchor=tk.W, pady=(10, 5))

        self.optimize_var = tk.BooleanVar(value=True)

        ttk.Radiobutton(
            left_panel,
            text="Optimize final structure",
            variable=self.optimize_var,
            value=True
        ).pack(anchor=tk.W, pady=2)

        ttk.Radiobutton(
            left_panel,
            text="Use as final result",
            variable=self.optimize_var,
            value=False
        ).pack(anchor=tk.W, pady=2)

        ttk.Label(
            left_panel,
            text="• Respects fixed layers\n"
                 "• Max 500 steps\n"
                 "• Convergence: fmax=0.02 eV/Å",
            font=("Arial", 9),
            foreground="gray",
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(10, 0))

        # Execute button
        ttk.Button(
            left_panel,
            text="Execute",
            command=self._execute
        ).pack(fill=tk.X, pady=(30, 10))

        # Save and exit button
        self.save_button = ttk.Button(
            left_panel,
            text="Save Results & Exit",
            command=self._save_and_exit,
            state=tk.DISABLED
        )
        self.save_button.pack(fill=tk.X, pady=(10, 0))

        # Right panel: Results
        right_panel = ttk.LabelFrame(main_frame, text="Optimization Log", padding="10")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Notebook
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Log tab
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="📋 Log")

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=25,
            width=80,
            wrap=tk.WORD,
            font=('Courier', 9),
            state='disabled'
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Results tab
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="⚡ Results")

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=25,
            width=80,
            wrap=tk.WORD,
            font=('Courier', 10),
            state='disabled'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def _log(self, message: str):
        """Add message to log"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update()
        logger.info(message)

    def _execute(self):
        """Execute final optimization"""
        self.optimize_final = self.optimize_var.get()

        self._log("=" * 80)
        self._log("FINAL STEP")
        self._log("=" * 80)

        if self.optimize_final:
            self._log("\nPerforming full relaxation...")
            self._log(f"Fixed layers: {self.n_fixed_layers}")
            self._log("Constraints: Bottom N layers completely fixed")
            self._log("Free atoms: Molecule + upper surface layers")
            self._log("Convergence: fmax = 0.02 eV/Å")
            self._log("Max steps: 500\n")

            self.running = True
            self.opt_thread = threading.Thread(target=self._optimize_thread)
            self.opt_thread.daemon = True
            self.opt_thread.start()
        else:
            self._log("\nUsing GA result as final structure")
            self.optimized_structure = self.best_structure
            self.final_energy = self.best_energy
            self._display_results()
            self.save_button.config(state=tk.NORMAL)

    def _optimize_thread(self):
        """Run optimization in thread"""
        try:
            structure = self.best_structure.copy()
            structure.set_calculator(self.calculator)

            # Get fixed layer indices
            fixed_indices = self._get_fixed_layer_indices()

            if fixed_indices:
                self._log(f"Fixing {len(fixed_indices)} atoms from bottom layers...")
                structure.set_constraint(FixAtoms(indices=fixed_indices))

            # Optimize
            self._log("Starting BFGS optimizer...")
            opt = BFGS(structure, logfile=None)
            opt.run(fmax=0.02, steps=500)

            self.final_energy = structure.get_potential_energy()
            self.optimized_structure = structure

            self._log("\n✓ Optimization completed successfully")

            # Calculate improvement
            surface_count = len(self.best_structure) - len(self.surface_analyzer.surface)
            initial_energy = self.best_energy
            improvement = initial_energy - self.final_energy
            improvement_pct = (improvement / abs(initial_energy)) * 100 if initial_energy != 0 else 0

            self._log(f"\nEnergy improvement:")
            self._log(f"  Before: {initial_energy:.4f} eV")
            self._log(f"  After:  {self.final_energy:.4f} eV")
            self._log(f"  Δ:      {improvement:.4f} eV ({improvement_pct:.1f}%)")

            self._display_results()
            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            self._log(f"\nERROR during optimization: {e}")
            messagebox.showerror("Optimization Error", f"Error:\n{e}")
            logger.error(f"Optimization error: {e}")

        finally:
            self.running = False

    def _get_fixed_layer_indices(self) -> list:
        """Get indices of atoms to fix"""
        # Get surface atom count
        surface_count = len(self.best_structure) - len(self.surface_analyzer.surface)

        # From surface_analyzer, get layer info
        layers_info = self.surface_analyzer._info.get('layers', {}).get('layers_list', [])
        n_total_layers = self.surface_analyzer._info.get('layers', {}).get('n_layers', 0)

        fixed_indices = []

        # Get bottom N layers
        for i in range(n_total_layers - self.n_fixed_layers, n_total_layers):
            if i < len(layers_info):
                # These indices are relative to surface only, add surface_count offset
                layer_indices = layers_info[i]['atom_indices']
                fixed_indices.extend(layer_indices)

        return fixed_indices

    def _display_results(self):
        """Display final results"""
        text = "=" * 80 + "\n"
        text += "FINAL RESULTS\n"
        text += "=" * 80 + "\n\n"

        text += "OPTIMIZATION SUMMARY:\n"
        text += "-" * 80 + "\n"
        text += f"Mode: {'Full relaxation' if self.optimize_final else 'GA result'}\n"
        text += f"Fixed layers: {self.n_fixed_layers}\n\n"

        if self.optimized_structure and self.final_energy is not None:
            improvement = self.best_energy - self.final_energy
            improvement_pct = (improvement / abs(self.best_energy)) * 100 if self.best_energy != 0 else 0

            text += "ENERGIES (eV):\n"
            text += f"  GA best:      {self.best_energy:>12.4f}\n"
            text += f"  After relax:  {self.final_energy:>12.4f}\n"
            text += f"  Improvement:  {improvement:>12.4f} ({improvement_pct:>6.1f}%)\n\n"

            text += "STRUCTURE INFO:\n"
            text += f"  Total atoms: {len(self.optimized_structure)}\n"
            text += f"  Surface atoms: {len(self.surface_analyzer.surface)}\n"
            text += f"  Molecule atoms: {len(self.optimized_structure) - len(self.surface_analyzer.surface)}\n\n"

        text += "=" * 80 + "\n"
        text += "Ready to save results\n"

        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        self.results_text.config(state='disabled')

    def _save_and_exit(self):
        """Save results and exit"""
        if self.optimized_structure is None:
            messagebox.showwarning("No Results", "No optimized structure to save")
            return

        try:
            # Create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join(self.output_dir, f"GOAD_v1_results_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)

            self._log(f"\nSaving results to: {result_dir}")

            # Save structure
            structure_file = os.path.join(result_dir, "final_optimized_structure.cif")
            write(structure_file, self.optimized_structure)
            self._log(f"  ✓ Structure: {structure_file}")

            # Save summary
            summary_file = os.path.join(result_dir, "summary.txt")
            with open(summary_file, 'w') as f:
                f.write("GOAD v1.0 - FINAL RESULTS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Optimization Date: {timestamp}\n")
                f.write(f"Final Energy (E_ads): {self.final_energy:.4f} eV\n")
                f.write(f"GA Best Energy: {self.best_energy:.4f} eV\n")
                f.write(f"Improvement: {self.best_energy - self.final_energy:.4f} eV\n")
                f.write(f"Fixed Layers: {self.n_fixed_layers}\n")

            self._log(f"  ✓ Summary: {summary_file}")

            messagebox.showinfo(
                "Success",
                f"Results saved to:\n{result_dir}\n\n"
                f"Final E_ads: {self.final_energy:.4f} eV"
            )

            self._log("\n✓ All results saved successfully")
            self._log("GOAD v1.0 workflow complete!")

        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save results:\n{e}")
            logger.error(f"Save error: {e}")
