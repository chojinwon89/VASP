# GOAD-MLIP Workflow — Quick Start

## Prerequisites
- Create and activate the conda environment you use for GOAD on Kestrel.
- Install the Python requirements:
  - `pip install -r requirements.txt`
- Install RDKit with conda if it is not already available:
  - `conda install -c conda-forge rdkit`

## Step-by-step

### Step 1 — Generate input structures
python prep_inputs.py

### Step 2 — Generate task table
python workflow/make_tasks.py

### Step 3 — Test one task interactively
python workflow/run_one_task.py --task-id 0 --tasks-csv workflow/tasks.csv

### Step 4 — Submit full Slurm array
mkdir -p slurm-logs runs
N=$(( $(wc -l < workflow/tasks.csv) - 1 ))
sbatch --array=0-$((N-1))%9 goad_array_kestrel.slurm

### Step 5 — Monitor
squeue -u $USER
tail -f slurm-logs/goad_array-*.out

### Step 6 — Collect results
python collect_results.py
column -s, -t < workflow/summary.csv
