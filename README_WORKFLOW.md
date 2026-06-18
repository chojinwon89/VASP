# GOAD-MLIP Workflow — Quick Start

## Prerequisites
- Create and activate the conda environment you use for GOAD on Kestrel.
- Install the Python requirements:
  - `pip install -r requirements.txt`
- Install RDKit with conda if it is not already available:
  - `conda install -c conda-forge rdkit`

## Supported surfaces (18 total)

| Metal | (111) | (110) | (100/001) |
|-------|-------|-------|-----------|
| Cu    | Cu111 | Cu110 | Cu001     |
| Pt    | Pt111 | Pt110 | Pt100     |
| Pd    | Pd111 | Pd110 | Pd100     |
| Ni    | Ni111 | Ni110 | Ni100     |
| Ag    | Ag111 | Ag110 | Ag100     |
| Au    | Au111 | Au110 | Au100     |

All slabs: 4×4×4, 15 Å vacuum, built with ASE default lattice constants.

## Supported molecules (9 total)

| Name        | SMILES      | Notes                              |
|-------------|-------------|------------------------------------|
| isopropanol | `CC(C)O`    | 3 heavy atoms, 1 OH                |
| CO2         | `O=C=O`     | rigid linear molecule              |
| ethanol     | `CCO`        |                                    |
| ethene      | `C=C`        | rigid (C=C double bond)            |
| ethane      | `CC`         | saturated C2                       |
| propane     | `CCC`        |                                    |
| propene     | `CC=C`       |                                    |
| propanol    | `CCCO`       | 3 rotatable bonds                  |
| glycerol    | `OCC(O)CO`  | 5 rotatable bonds; larger GA search |

> **Note on glycerol:** the genome has 11 variables. `batch_isopropanol.py`
> automatically uses `population_size=60` and `generations=100` for glycerol,
> and `population_size=50`, `generations=100` for propanol.

## Task count

18 surfaces × 9 molecules × 3 seeds = **486 Slurm array tasks** (task_id 0–485)

## Step-by-step

### Step 1 — Generate input structures
```bash
python prep_inputs.py
```
Builds all 18 surface CIFs and all 9 molecule CIFs under `inputs/`.

### Step 2 — Generate task table
```bash
python workflow/make_tasks.py
```
Writes `workflow/tasks.csv` with 486 rows.

### Step 3 — Test one task interactively
```bash
python workflow/run_one_task.py --task-id 0 --tasks-csv workflow/tasks.csv
```

### Step 4 — Submit full Slurm array
```bash
mkdir -p slurm-logs runs
sbatch --array=0-485%50 goad_array_kestrel.slurm
```

### Step 5 — Monitor
```bash
squeue -u $USER
tail -f slurm-logs/goad_array-*.out
```

### Step 6 — Collect results
```bash
python collect_results.py
column -s, -t < workflow/summary.csv
```

### Step 7 — Extract best geometries
```bash
python extract_poscar.py --verbose
```

### Step 8 — Set up DFT adsorbed jobs (all 3 seeds)
```bash
export VASP_PP_PATH=/home/jcho5/project/paw64/potpaw_PBE_64
python setup_vasp_jobs.py --all-seeds
```

### Step 9 — Set up DFT bare-slab reference jobs
```bash
python setup_slab_jobs.py
```
Creates `vasp_slab/<SurfaceName>/` with POSCAR (Selective Dynamics), INCAR, KPOINTS, POTCAR, Slurm script.

### Step 10 — Set up DFT gas-phase molecule jobs
```bash
python setup_molecule_jobs.py
```
Creates `vasp_mol/<MoleculeName>/` with POSCAR (20×20×20 Å box), INCAR (ISMEAR=0, Gamma-point), KPOINTS (1×1×1), POTCAR, Slurm script.

## Adsorption energy formula

```
E_ads = E_total(slab+mol) - E_surf(slab) - E_mol(gas)
```

## Submit all DFT jobs

```bash
# Adsorbed systems
for d in poscar/*/; do (cd "$d" && sbatch slm.vasp.kestrel); done

# Bare slabs
for d in vasp_slab/*/; do (cd "$d" && sbatch slm.vasp.kestrel); done

# Gas molecules
for d in vasp_mol/*/; do (cd "$d" && sbatch slm.vasp.kestrel); done
```
