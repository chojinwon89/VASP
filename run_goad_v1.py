#!/usr/bin/env python3
"""
GOAD v1.0 - Global Optimization with ASE Design
Main launcher integrating all workflow steps

Usage:
    python3 run_goad_v1.py
"""

import tkinter as tk
from tkinter import messagebox
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import GUI windows
from goad_v1.gui.analysis_window import AnalysisWindow
from goad_v1.gui.reference_energies_window import ReferenceEnergiesWindow
from goad_v1.gui.ga_window import GAWindow
from goad_v1.gui.final_optimization_window import FinalOptimizationWindow


class GOADv1Workflow:
    """Main GOAD v1.0 workflow manager"""

    def __init__(self):
        """Initialize GOAD v1.0"""
        self.root = tk.Tk()
        self.root.title("GOAD v1.0 - Global Optimization with ASE Design")
        self.root.geometry("400x200")

        # Setup calculator (to be assigned)
        self.calculator = None
        self._setup_calculator()

        # Start analysis window
        self._show_analysis_window()

    def _setup_calculator(self):
        """Verify that MatterSim is available"""
        try:
            from mattersim.forcefield import MatterSimCalculator
            logger.info("MatterSim is available - calculator will be selected in next step")
            self.calculator = None  # Will be set in reference energies window

        except ImportError:
            logger.error("MatterSim not available")
            messagebox.showerror(
                "Missing Dependency",
                "MatterSim is required but not installed.\n\n"
                "Please install it with:\n"
                "pip install mattersim"
            )
            sys.exit(1)

    def _show_analysis_window(self):
        """Show analysis window"""
        self.root.withdraw()

        analysis_root = tk.Tk()
        analysis_window = AnalysisWindow(
            analysis_root,
            on_complete_callback=self._on_analysis_complete
        )

        analysis_root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        analysis_root.mainloop()

    def _on_analysis_complete(self, surface, molecule, surface_analyzer, molecule_analyzer,
                              n_fixed_layers, n_total_layers):
        """Called when analysis is complete"""
        logger.info(f"Analysis complete: {n_fixed_layers}/{n_total_layers} layers to fix")

        # Show reference energies window
        ref_root = tk.Tk()
        ref_window = ReferenceEnergiesWindow(
            ref_root,
            surface=surface,
            molecule=molecule,
            surface_analyzer=surface_analyzer,
            molecule_analyzer=molecule_analyzer,
            n_fixed_layers=n_fixed_layers,
            n_total_layers=n_total_layers,
            on_complete_callback=self._on_reference_complete
        )

        # Set calculator
        ref_window.set_calculator(self.calculator)

        ref_root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        ref_root.mainloop()

    def _on_reference_complete(self, surface_relaxed, molecule_relaxed, surface_energy,
                              molecule_energy, n_fixed_layers, calculator):
        """Called when reference energies are calculated"""
        logger.info(f"Reference energies calculated:")
        logger.info(f"  E_surface: {surface_energy:.4f} eV")
        logger.info(f"  E_molecule: {molecule_energy:.4f} eV")

        # Show GA window
        ga_root = tk.Tk()
        ga_window = GAWindow(
            ga_root,
            surface_relaxed=surface_relaxed,
            molecule_relaxed=molecule_relaxed,
            surface_energy=surface_energy,
            molecule_energy=molecule_energy,
            n_fixed_layers=n_fixed_layers,
            calculator=calculator,
            on_complete_callback=self._on_ga_complete
        )

        ga_root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        ga_root.mainloop()

    def _on_ga_complete(self, best_structure, best_energy, n_fixed_layers, calculator):
        """Called when GA is complete"""
        logger.info(f"GA complete: Best E_ads = {best_energy:.4f} eV")

        # Show final optimization window
        final_root = tk.Tk()

        # Import here to avoid circular imports
        from goad_v1.analysis.surface_analyzer import SurfaceAnalyzer
        # Create a dummy surface analyzer with the best structure
        # This is a simplified version - in production you'd pass the original

        # For now, we'll create a minimal surface analyzer
        # In real use, you'd pass the original surface_analyzer
        surface_analyzer_dummy = None
        try:
            # Extract surface from best_structure
            # The surface is the first N atoms (this is a simplification)
            # In production, you'd track this through the workflow
            surface_analyzer_dummy = type('obj', (object,), {
                '_info': {'layers': {'layers_list': [], 'n_layers': 0}},
                'surface': best_structure[:len(best_structure)//2]  # Rough estimate
            })()
        except:
            pass

        final_window = FinalOptimizationWindow(
            final_root,
            best_structure=best_structure,
            best_energy=best_energy,
            n_fixed_layers=n_fixed_layers,
            surface_analyzer=surface_analyzer_dummy,
            calculator=calculator
        )

        final_root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        final_root.mainloop()

    def _on_window_close(self):
        """Handle window close"""
        logger.info("Window closed")
        pass

    def run(self):
        """Start the workflow"""
        self.root.mainloop()


def main():
    """Main entry point"""
    logger.info("Starting GOAD v1.0")

    try:
        workflow = GOADv1Workflow()
        workflow.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"Fatal error:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
