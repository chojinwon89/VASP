# GOAD v1.0 - Quick Start Guide

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed, then install required packages:

```bash
pip install ase rdkit matplotlib numpy
```

### MatterSim Installation
GOAD v1.0 uses MatterSim force field calculator. Install it with:

```bash
pip install mattersim
```

## Running GOAD v1.0

From the GOAD v1.0 directory:

```bash
python3 run_goad_v1.py
```

This will launch the main workflow interface.

## Workflow Steps

### Step 1: Structure Analysis (Window 1)
1. Click "Load Surface CIF" and select your surface structure file
2. Click "Load Molecule CIF" and select your molecule structure file
3. Click "Analyze Structures" to perform automatic analysis
4. For slab surfaces, use the spinbox to select how many layers to keep fixed
5. View results in the tabs:
   - **🔬 Structures**: 3D visualization of surface + molecule
   - **🔷 Surface**: Detailed surface information (layers, elements, etc.)
   - **🔶 Molecule**: Detailed molecule information (formula, mass, etc.)
6. Click "Next: Reference Energies" to proceed

### Step 2: Reference Energy Calculation (Window 2)
1. Review the setup summary (fixed layers, default calculator)
2. Select a calculator:
   - **MatterSim 1M** (Fast, lower accuracy) - DEFAULT
   - **MatterSim 5M** (Balanced speed/accuracy)
   - **MatterSim 5M + D3** (Slow, highest accuracy)
3. Choose reference structure calculation mode:
   - **Relax both** (0-1000 BFGS steps) - Recommended
   - **Use as-is** (single point energy)
4. Click "Calculate References"
5. View results in the tabs:
   - **📋 Log**: Real-time calculation progress
   - **⚡ Energies**: Final reference energy values
6. Click "Next: Genetic Algorithm" to proceed

### Step 3: Genetic Algorithm (Window 3)
1. Adjust GA parameters if desired:
   - **Generations**: Number of iterations (default: 50)
   - **Population Size**: Individuals per generation (default: 30)
   - **Mutation Rate**: 0-1, probability of random changes (default: 0.3)
   - **Crossover Rate**: 0-1, probability of parent mixing (default: 0.7)
   - **Elite Size**: Best individuals to preserve (default: 5)
2. Click "Run GA" to start the optimization
3. Monitor progress:
   - **📊 Energy Evolution**: Real-time plot of energy vs. evaluations
   - **📋 Results**: Best solution found with genes
4. Interpretation of results:
   - **POSITIONING GENES (6)**: X, Y, Z coordinates and α, β, γ Euler angles
   - **TORSION GENES (N)**: Dihedral angles for each rotatable bond
5. Click "Next: Final Optimization" when satisfied

### Step 4: Final Optimization (Window 4)
1. Review the best structure found by GA
2. Choose final optimization options:
   - **Relax final structure** (optional): Full BFGS with fixed layers
   - **Export results**: Save to timestamped results directory
3. Click "Complete Workflow" to finish

## Input File Formats

Both surface and molecule structures should be in **CIF format** (.cif files).

### Example Surface (Cu111 slab)
- Crystal structure with periodic boundary conditions
- Multiple atomic layers stacked along Z-axis
- Bottom N layers will be fixed during optimization

### Example Molecule
- Isolated molecular structure
- Can have rotatable bonds (will be detected automatically)
- Center of mass will be calculated and used for positioning

## Output Files

Results are saved in `results/` directory with timestamp:

```
results/YYYYMMDD_HHMMSS/
├── best_structure.traj        # ASE trajectory file
├── best_structure.cif         # CIF format
├── summary.txt               # Text summary of results
└── optimization_log.txt      # Detailed log
```

## Tips & Tricks

### For Quick Testing
- Use **MatterSim 1M** (fastest)
- Set generations to 20-30
- Population size 15-20
- Use "Relax both" for reference energies

### For Accurate Results
- Use **MatterSim 5M + D3** (highest accuracy)
- Set generations to 100+
- Population size 50+
- Use "Relax both" for reference energies

### Understanding the Genome
- **First 6 genes**: Control molecule position and orientation
  - Gene 0-2: X, Y, Z position (Å)
  - Gene 3-5: α, β, γ rotation angles (°)
- **Remaining genes**: Control molecular flexibility
  - Gene 6+: Dihedral angles for rotatable bonds (°)

### Energy Interpretation
- **Negative adsorption energy**: Favorable adsorption (molecule sticks)
- **Positive adsorption energy**: Unfavorable adsorption (molecule repelled)
- **Near-zero**: Weak or no interaction

## Troubleshooting

### "MatterSim not installed"
```bash
pip install mattersim
```

### "Could not load calculator"
- Ensure MatterSim is properly installed
- Check internet connection (models may need downloading)
- Try MatterSim 1M if 5M/D3 fail to load

### "Analysis Error"
- Ensure CIF files are valid
- Check that files contain proper crystal/molecular structures
- Try opening files in visualization tool first (ASE GUI, etc.)

### Performance Issues
- Reduce population size or generations
- Use MatterSim 1M instead of 5M/5M+D3
- Simplify molecule structure if possible

---
GOAD v1.0 - Genetic Optimization for Adsorbed structures using ASE Design
