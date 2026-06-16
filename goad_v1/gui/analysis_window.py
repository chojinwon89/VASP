"""
Window 1: Structure and Molecule Analysis for GOAD v1.0

Loads CIF files and analyzes surface/molecule properties
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from ase.io import read
from ase import Atoms
import logging
from pathlib import Path

from ..analysis.surface_analyzer import SurfaceAnalyzer
from ..analysis.molecule_analyzer import MoleculeAnalyzer
from .structure_viewer import StructureViewer

logger = logging.getLogger(__name__)


class AnalysisWindow:
    """Main window for structure analysis"""

    def __init__(self, root, on_complete_callback=None):
        """
        Initialize analysis window.

        Args:
            root: Tkinter root window
            on_complete_callback: Function to call when analysis is complete
        """
        self.root = root
        self.root.title("GOAD v1.0 - Structure Analysis")
        self.root.geometry("900x700")

        self.on_complete_callback = on_complete_callback

        # Data storage
        self.surface = None
        self.molecule = None
        self.surface_analyzer = None
        self.molecule_analyzer = None
        self.surface_type = None
        self.n_layers = None
        self.selected_fixed_layers = None

        # Configure logging
        logging.basicConfig(level=logging.INFO)

        # Build GUI
        self._build_gui()

    def _build_gui(self):
        """Build the GUI layout"""

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(
            main_frame,
            text="GOAD v1.0 - Structure Analysis",
            font=("Arial", 16, "bold")
        )
        title.pack(pady=(0, 20))

        # Left panel: File loading and analysis
        left_panel = ttk.LabelFrame(main_frame, text="Input Files & Analysis", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        # Surface section
        surface_frame = ttk.LabelFrame(left_panel, text="Surface", padding="10")
        surface_frame.pack(fill=tk.X, pady=(0, 10))

        self.surface_label = ttk.Label(surface_frame, text="No file loaded", foreground="red")
        self.surface_label.pack(anchor=tk.W, pady=(0, 5))

        ttk.Button(
            surface_frame,
            text="Load Surface CIF",
            command=self._load_surface
        ).pack(fill=tk.X, pady=(0, 5))

        # Molecule section
        molecule_frame = ttk.LabelFrame(left_panel, text="Molecule", padding="10")
        molecule_frame.pack(fill=tk.X, pady=(0, 10))

        self.molecule_label = ttk.Label(molecule_frame, text="No file loaded", foreground="red")
        self.molecule_label.pack(anchor=tk.W, pady=(0, 5))

        ttk.Button(
            molecule_frame,
            text="Load Molecule CIF",
            command=self._load_molecule
        ).pack(fill=tk.X, pady=(0, 5))

        # Analysis button
        ttk.Button(
            left_panel,
            text="Analyze Structures",
            command=self._analyze
        ).pack(fill=tk.X, pady=(0, 20))

        # Layer selection (appears after analysis)
        self.layer_frame = ttk.LabelFrame(left_panel, text="Fixed Layers", padding="10")
        self.layer_label = ttk.Label(
            self.layer_frame,
            text="Run analysis first",
            foreground="gray"
        )
        self.layer_label.pack(anchor=tk.W)

        self.layer_spinbox = None

        # Next button
        self.next_button = ttk.Button(
            left_panel,
            text="Next: Reference Energies",
            command=self._go_to_reference_energies,
            state=tk.DISABLED
        )
        self.next_button.pack(fill=tk.X, pady=(10, 0))

        # Right panel: Analysis results and visualization
        right_panel = ttk.LabelFrame(main_frame, text="Analysis & Visualization", padding="10")
        right_panel.pack(fill=tk.BOTH, expand=True)

        # Notebook for results
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Visualization tab
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="🔬 Structures")

        self.surface_viewer = StructureViewer(viz_frame, width=7, height=4)

        # Info frame below visualization
        info_viz_frame = ttk.Frame(viz_frame)
        info_viz_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(
            info_viz_frame,
            text="Surface (larger atoms) | Molecule (smaller atoms, red border)",
            font=("Arial", 8),
            foreground="gray"
        ).pack()

        # Surface info tab
        surface_info_frame = ttk.Frame(self.notebook)
        self.notebook.add(surface_info_frame, text="🔷 Surface")

        self.surface_text = scrolledtext.ScrolledText(
            surface_info_frame,
            height=25,
            width=60,
            wrap=tk.WORD,
            font=('Courier', 9),
            state='disabled'
        )
        self.surface_text.pack(fill=tk.BOTH, expand=True)

        # Molecule info tab
        molecule_info_frame = ttk.Frame(self.notebook)
        self.notebook.add(molecule_info_frame, text="🔶 Molecule")

        self.molecule_text = scrolledtext.ScrolledText(
            molecule_info_frame,
            height=25,
            width=60,
            wrap=tk.WORD,
            font=('Courier', 9),
            state='disabled'
        )
        self.molecule_text.pack(fill=tk.BOTH, expand=True)

    def _load_surface(self):
        """Load surface CIF file"""
        file_path = filedialog.askopenfilename(
            title="Select Surface CIF File",
            filetypes=[("CIF files", "*.cif"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.surface = read(file_path)
                filename = Path(file_path).name
                self.surface_label.config(text=f"✓ {filename}", foreground="green")
                logger.info(f"Loaded surface: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load surface file:\n{e}")
                logger.error(f"Error loading surface: {e}")

    def _load_molecule(self):
        """Load molecule CIF file"""
        file_path = filedialog.askopenfilename(
            title="Select Molecule CIF File",
            filetypes=[("CIF files", "*.cif"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.molecule = read(file_path)
                filename = Path(file_path).name
                self.molecule_label.config(text=f"✓ {filename}", foreground="green")
                logger.info(f"Loaded molecule: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load molecule file:\n{e}")
                logger.error(f"Error loading molecule: {e}")

    def _analyze(self):
        """Analyze loaded structures"""
        if self.surface is None or self.molecule is None:
            messagebox.showwarning("Missing Files", "Please load both surface and molecule files first")
            return

        try:
            # Analyze surface
            logger.info("Starting surface analysis...")
            self.surface_analyzer = SurfaceAnalyzer(self.surface)
            surface_info = self.surface_analyzer.analyze()
            self.surface_type = surface_info['surface_type']
            logger.info(f"Surface analysis complete: type={self.surface_type}")

            # Update surface text
            surface_text = self.surface_analyzer.get_info_text()
            self.surface_text.config(state='normal')
            self.surface_text.delete(1.0, tk.END)
            self.surface_text.insert(1.0, surface_text)
            self.surface_text.config(state='disabled')
            logger.info("Surface text updated")

            # Analyze molecule
            logger.info("Starting molecule analysis...")
            self.molecule_analyzer = MoleculeAnalyzer(self.molecule)
            molecule_info = self.molecule_analyzer.analyze()
            logger.info(f"Molecule analysis complete")

            # Update molecule text
            molecule_text = self.molecule_analyzer.get_info_text()
            self.molecule_text.config(state='normal')
            self.molecule_text.delete(1.0, tk.END)
            self.molecule_text.insert(1.0, molecule_text)
            self.molecule_text.config(state='disabled')
            logger.info("Molecule text updated")

            # Display structures (with robust error handling)
            logger.info("Attempting to display structures...")
            try:
                self.surface_viewer.display_combined(self.surface, self.molecule,
                                                    title="Surface + Molecule")
                logger.info("Structures displayed successfully")
            except Exception as e:
                logger.warning(f"Could not display structures: {e}")
                # Continue anyway, visualization is not critical

            # Handle surface type
            if self.surface_type == "slab":
                logger.info("Setting up layer selection...")
                self._setup_layer_selection(surface_info)
                self.next_button.config(state=tk.NORMAL)
                logger.info("Analysis complete - ready for next step")
                messagebox.showinfo(
                    "Analysis Complete",
                    f"Surface is type: SLAB\n"
                    f"Detected {self.n_layers} atomic layers\n\n"
                    f"Select how many layers to keep fixed and click 'Next'"
                )
            else:
                messagebox.showinfo(
                    "Analysis Complete",
                    "Surface type: POROUS (MOF/Zeolite)\n\n"
                    "This feature is under development.\n"
                    "Currently only SLAB surfaces are supported."
                )
                self.next_button.config(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            messagebox.showerror("Analysis Error", f"Error during analysis:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _setup_layer_selection(self, surface_info: dict):
        """Setup layer selection UI"""
        layers = surface_info.get('layers', {})
        self.n_layers = layers.get('n_layers', 0)

        if self.n_layers == 0:
            return

        # Clear previous widget
        if self.layer_spinbox:
            self.layer_spinbox.destroy()

        # Update label
        self.layer_label.config(
            text=f"Select fixed layers (1-{self.n_layers}):",
            foreground="black"
        )

        # Create spinbox for layer selection
        self.layer_spinbox = ttk.Spinbox(
            self.layer_frame,
            from_=1,
            to=self.n_layers,
            width=10
        )
        self.layer_spinbox.set(1)
        self.layer_spinbox.pack(pady=(5, 0))

        self.layer_frame.pack(fill=tk.X, pady=(0, 20))

    def _go_to_reference_energies(self):
        """Proceed to reference energy calculation"""
        if self.surface_type != "slab":
            messagebox.showwarning("Unsupported", "Only SLAB surfaces are supported")
            return

        try:
            self.selected_fixed_layers = int(self.layer_spinbox.get())
        except:
            messagebox.showerror("Error", "Invalid layer selection")
            return

        if self.on_complete_callback:
            self.on_complete_callback(
                surface=self.surface,
                molecule=self.molecule,
                surface_analyzer=self.surface_analyzer,
                molecule_analyzer=self.molecule_analyzer,
                n_fixed_layers=self.selected_fixed_layers,
                n_total_layers=self.n_layers
            )
