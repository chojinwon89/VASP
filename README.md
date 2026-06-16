# GOAD-MLIP Adsorption Energy Workflow

Automated workflow for generating adsorption-energy databases of small molecules on metal surfaces using a **Genetic Algorithm global search** + **SevenNet-OMNI universal MLIP**.

## Overview

- **Energy engine**: SevenNet-OMNI (`7net-mf-ompa`, `modal='omat24'`) — equivariant GNN potential trained on PBE+D3 (OMat24 dataset). No DFT at any step.
- **Global search**: Genetic Algorithm (GA) encodes each pose as a genome of position (x,y,z) + orientation (α,β,γ) + torsions (τ₁…τₙ), evolves 40 individuals over 80 generations, repeated with 3 independent random seeds.
- **Local refinement**: BFGS relaxation with molecule + top surface layers free, bottom 2 layers fixed, fmax = 0.05 eV/Å.
- **Adsorption energy**: `E_ads = E_total − (E_surf + E_mol)`

## Repository Structure

```
VASP/
├── prep_inputs.py              # Build surface slabs and gas-phase references
├── batch_isopropanol.py        # Single-task GA search + relaxation (env-var driven)
├── generate_molecules.py       # SMILES → RDKit 3D coords → molecules/*.xyz
├── relax_gas_molecules.py      # MLIP relaxation of gas-phase molecules
├── molecules.csv               # Molecule name + SMILES table
├── molecules/                  # Raw RDKit geometries (.xyz)
├── molecules_relaxed/          # MLIP-relaxed gas-phase geometries (.xyz)
├── runs/                       # Per-task output directories
├── slurm-logs/                 # Slurm stdout/stderr logs
└── workflow/
    ├── make_tasks.py           # Generate workflow/tasks.csv
    ├── tasks.csv               # One row = one surface/adsorbate/seed task
    ├── run_one_task.py         # Execute one row from tasks.csv
    ├── collect_results.py      # Harvest results → workflow/summary.csv
    └── goad_array_kestrel.slurm  # Slurm job-array submission script
```

## Quick Start on Kestrel

```bash
# 1. Activate environment
conda activate goad

# 2. Generate molecule library from SMILES
python generate_molecules.py
python relax_gas_molecules.py

# 3. Generate task table
python workflow/make_tasks.py
head workflow/tasks.csv

# 4. Test one task interactively
python workflow/run_one_task.py --task-id 0 --tasks-csv workflow/tasks.csv

# 5. Submit full array to Slurm
N=$(($(wc -l < workflow/tasks.csv) - 1))
sbatch --array=0-$((N-1))%50 workflow/goad_array_kestrel.slurm

# 6. Collect results after completion
python workflow/collect_results.py
column -s, -t < workflow/summary.csv | less -S
```

## Workflow vs. Traditional VASP DFT

| Aspect | GOAD + SevenNet-OMNI | Traditional VASP DFT |
|---|---|---|
| Energy engine | Universal MLIP (PBE+D3) | Self-consistent DFT |
| Cost per single-point | ~1–5 s (CPU), ~0.05 s (GPU) | ~10–60 min (64-atom slab) |
| Configurational search | GA: ~9,600 poses/system | Manual: 3–10 hand-picked sites |
| Wall time per system | ~6–10 h (16 CPU cores) | Days to weeks |
| Accuracy (physisorption) | ~10–50 meV vs. PBE+D3 reference | The reference |

## Production Strategy

1. Survey all systems with GOAD + SevenNet-OMNI (~5 systems/day on CPU)
2. Validate top-K pose per system with VASP single-point or short BFGS
3. Publish E_ads from DFT validation with GA-assured global minimum confidence

## Kestrel HPC Details

- Allocation: `ccpc`
- Partition: `shared`
- Per-task resources: 1 node, 16 cores, 64 GB RAM, 24 h wall time
- Concurrency limit: 50 simultaneous array tasks (`%50`)
