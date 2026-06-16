"""
Window 3: Genetic Algorithm Optimization for GOAD v1.0

Runs GA with fixed surface and free molecule
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import logging

from ..ga.genetic_algorithm import GeneticAlgorithm

logger = logging.getLogger(__name__)


class GAWindow:
    """Window for genetic algorithm optimization"""

    def __init__(self, root, surface_relaxed, molecule_relaxed, surface_energy,
                 molecule_energy, n_fixed_layers, calculator, on_complete_callback=None):
        """
        Initialize GA window.

        Args:
            root: Tkinter root window
            surface_relaxed: Relaxed surface structure
            molecule_relaxed: Relaxed molecule structure
            surface_energy: Reference surface energy
            molecule_energy: Reference molecule energy
            n_fixed_layers: Number of fixed layers
            calculator: ASE calculator
            on_complete_callback: Function to call when complete
        """
        self.root = root
        self.root.title("GOAD v1.0 - Genetic Algorithm")
        self.root.geometry("1100x700")

        self.surface = surface_relaxed
        self.molecule = molecule_relaxed
        self.surface_energy = surface_energy
        self.molecule_energy = molecule_energy
        self.n_fixed_layers = n_fixed_layers
        self.calculator = calculator
        self.on_complete_callback = on_complete_callback

        # GA parameters
        self.ga_params = {
            'generations': 50,
            'population_size': 30,
            'mutation_rate': 0.3,
            'crossover_rate': 0.7,
            'elite_size': 5,
        }

        # GA instance and state
        self.ga = None
        self.running = False
        self.ga_thread = None
        self.ga_results = None

        # Build GUI
        self._build_gui()

    def _build_gui(self):
        """Build the GUI layout"""

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(
            main_frame,
            text="GOAD v1.0 - Genetic Algorithm",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(0, 20))

        # Left panel: Parameters and controls
        left_panel = ttk.LabelFrame(main_frame, text="GA Configuration", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10), width=300)

        # Summary
        summary_frame = ttk.Frame(left_panel)
        summary_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(summary_frame, text="Setup Summary:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(summary_frame, text=f"E_surface: {self.surface_energy:.4f} eV").pack(anchor=tk.W)
        ttk.Label(summary_frame, text=f"E_molecule: {self.molecule_energy:.4f} eV").pack(anchor=tk.W)
        ttk.Label(summary_frame, text=f"Fixed layers: {self.n_fixed_layers}").pack(anchor=tk.W)

        # GA Parameters
        params_frame = ttk.LabelFrame(left_panel, text="Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 20))

        # Generations
        ttk.Label(params_frame, text="Generations:").pack(anchor=tk.W)
        self.gen_spin = ttk.Spinbox(params_frame, from_=10, to=500, width=10)
        self.gen_spin.set(self.ga_params['generations'])
        self.gen_spin.pack(anchor=tk.W, pady=(0, 10))

        # Population size
        ttk.Label(params_frame, text="Population Size:").pack(anchor=tk.W)
        self.pop_spin = ttk.Spinbox(params_frame, from_=10, to=200, width=10)
        self.pop_spin.set(self.ga_params['population_size'])
        self.pop_spin.pack(anchor=tk.W, pady=(0, 10))

        # Mutation rate
        ttk.Label(params_frame, text="Mutation Rate:").pack(anchor=tk.W)
        self.mut_spin = ttk.Spinbox(params_frame, from_=0.1, to=0.9, increment=0.1, width=10)
        self.mut_spin.set(self.ga_params['mutation_rate'])
        self.mut_spin.pack(anchor=tk.W, pady=(0, 10))

        # Crossover rate
        ttk.Label(params_frame, text="Crossover Rate:").pack(anchor=tk.W)
        self.cross_spin = ttk.Spinbox(params_frame, from_=0.1, to=0.9, increment=0.1, width=10)
        self.cross_spin.set(self.ga_params['crossover_rate'])
        self.cross_spin.pack(anchor=tk.W, pady=(0, 10))

        # Elite size
        ttk.Label(params_frame, text="Elite Size:").pack(anchor=tk.W)
        self.elite_spin = ttk.Spinbox(params_frame, from_=1, to=50, width=10)
        self.elite_spin.set(self.ga_params['elite_size'])
        self.elite_spin.pack(anchor=tk.W)

        # Control buttons
        ttk.Button(
            left_panel,
            text="Run GA",
            command=self._run_ga
        ).pack(fill=tk.X, pady=(20, 10))

        self.stop_button = ttk.Button(
            left_panel,
            text="Stop",
            command=self._stop_ga,
            state=tk.DISABLED
        )
        self.stop_button.pack(fill=tk.X, pady=(0, 10))

        # Next button
        self.next_button = ttk.Button(
            left_panel,
            text="Next: Final Optimization",
            command=self._go_to_final_optimization,
            state=tk.DISABLED
        )
        self.next_button.pack(fill=tk.X, pady=(10, 0))

        # Right panel: Results
        right_panel = ttk.LabelFrame(main_frame, text="Optimization Results", padding="10")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Notebook
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Plot tab
        plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(plot_frame, text="📈 Energy Evolution")

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

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

    def _run_ga(self):
        """Run the genetic algorithm"""
        try:
            # Get parameters
            self.ga_params['generations'] = int(self.gen_spin.get())
            self.ga_params['population_size'] = int(self.pop_spin.get())
            self.ga_params['mutation_rate'] = float(self.mut_spin.get())
            self.ga_params['crossover_rate'] = float(self.cross_spin.get())
            self.ga_params['elite_size'] = int(self.elite_spin.get())

            # Create GA instance
            self.ga = GeneticAlgorithm(
                surface=self.surface,
                molecule=self.molecule,
                calculator=self.calculator,
                surface_energy=self.surface_energy,
                molecule_energy=self.molecule_energy,
                n_fixed_layers=self.n_fixed_layers,
                **self.ga_params,
                verbose=True
            )

            # Run in separate thread
            self.running = True
            self.stop_button.config(state=tk.NORMAL)

            self._log("=" * 80)
            self._log("Starting Genetic Algorithm")
            self._log("=" * 80)

            self.ga_thread = threading.Thread(target=self._run_ga_thread)
            self.ga_thread.daemon = True
            self.ga_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Could not start GA:\n{e}")
            logger.error(f"GA error: {e}")

    def _run_ga_thread(self):
        """Run GA in thread"""
        try:
            self.ga_results = self.ga.run()

            # Update UI
            self._log("\n" + "=" * 80)
            self._log("GA COMPLETED SUCCESSFULLY")
            self._log("=" * 80)

            self._display_results()
            self.next_button.config(state=tk.NORMAL)

        except Exception as e:
            self._log(f"\nERROR: {e}")
            messagebox.showerror("GA Error", f"Error during GA execution:\n{e}")
            logger.error(f"GA execution error: {e}")

        finally:
            self.running = False
            self.stop_button.config(state=tk.DISABLED)

    def _stop_ga(self):
        """Stop the GA"""
        self.running = False
        self._log("\nGA stopped by user")
        self.stop_button.config(state=tk.DISABLED)

    def _display_results(self):
        """Display GA results"""
        if not self.ga_results:
            return

        # Plot energy evolution
        fitness_history = self.ga_results['fitness_history']
        generations = self.ga_results['generations']

        self.ax.clear()
        self.ax.plot(fitness_history, 'b-', linewidth=1, alpha=0.5, label='Evaluation')
        self.ax.plot(range(0, len(fitness_history), self.ga_params['population_size']),
                    [min(fitness_history[i:i+self.ga_params['population_size']])
                     for i in range(0, len(fitness_history), self.ga_params['population_size'])],
                    'r-', linewidth=2, label='Best per generation')
        self.ax.set_xlabel('Evaluation Number')
        self.ax.set_ylabel('Adsorption Energy (eV)')
        self.ax.set_title('GA Energy Evolution')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

        # Display results text
        text = "=" * 80 + "\n"
        text += "GENETIC ALGORITHM RESULTS\n"
        text += "=" * 80 + "\n\n"

        text += f"Generations completed: {self.ga_results['generations']}\n"
        text += f"Population size: {self.ga_results['population_size']}\n"
        text += f"Total evaluations: {len(fitness_history)}\n\n"

        text += "BEST SOLUTION FOUND:\n"
        text += "-" * 80 + "\n"
        text += f"E_adsorption:     {self.ga_results['best_energy']:>12.4f} eV\n\n"

        if self.ga_results['best_individual']:
            best = self.ga_results['best_individual']
            text += "POSITIONING GENES (6):\n"
            text += f"Position (Å):     X={best['position'][0]:>8.2f}  Y={best['position'][1]:>8.2f}  Z={best['position'][2]:>8.2f}\n"
            text += f"Orientation (°):  α={best['orientation'][0]:>8.1f}  β={best['orientation'][1]:>8.1f}  γ={best['orientation'][2]:>8.1f}\n"

            if 'torsions' in best and len(best['torsions']) > 0:
                text += f"\nTORSION GENES ({len(best['torsions'])}):\n"
                for i, torsion in enumerate(best['torsions']):
                    text += f"  Torsion {i}: {torsion:>8.1f}°\n"

        text += "\n" + "=" * 80 + "\n"
        text += "Ready for final optimization step (optional)\n"

        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        self.results_text.config(state='disabled')

    def _go_to_final_optimization(self):
        """Proceed to final optimization"""
        if not self.ga_results:
            messagebox.showwarning("Incomplete", "Run GA first")
            return

        if self.on_complete_callback:
            self.on_complete_callback(
                best_structure=self.ga_results['best_structure'],
                best_energy=self.ga_results['best_energy'],
                n_fixed_layers=self.n_fixed_layers,
                calculator=self.calculator
            )
