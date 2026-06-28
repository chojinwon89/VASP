#!/bin/bash
#SBATCH --job-name=goad_chgnet_mace
#SBATCH --account=ccpc
#SBATCH --partition=standard
#SBATCH --nodes=1                       # 1 node per task (GA is single-node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=104             # full 104 cores per node for threading
#SBATCH --mem=200G                      # capped: prevents memory starvation when
                                        # multiple array tasks land on the same node
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/goad_chgnet_mace-%A_%a.out
#SBATCH --error=slurm-logs/goad_chgnet_mace-%A_%a.err

# NOTE: This script uses the goad-chgnet conda env which supports:
#   - CHGNet  (chgnet)
#   - MACE-MP-0 (mace_mp, mace_mp_d3)
#   - MACE-OFF23 (mace_off)
#
# For MatterSim / SevenNet jobs use goad_array_kestrel.slurm instead.
#
# Usage:
#   sbatch --array=0-107%20 goad_CHGNETMACE_kestrel.slurm workflow/tasks_custom.csv

set -euo pipefail

module purge
module load conda
conda activate /scratch/jcho5/envs/goad-chgnet

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Force serial energy evaluation on CPU.
# ProcessPoolExecutor was spawning 104 worker processes each loading the
# full CHGNet/MACE model, causing RAM exhaustion and deadlocks.
export GOAD_N_WORKERS=1

# Per-step watchdog timeout in seconds (default 300 = 5 min).
export GOAD_STEP_TIMEOUT=300

# Always cd to the directory where sbatch was submitted from
cd $SLURM_SUBMIT_DIR

mkdir -p slurm-logs runs

echo "=========================================="
echo "GOAD CHGNet/MACE Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Host:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "Workers:   ${GOAD_N_WORKERS}"
echo "Timeout:   ${GOAD_STEP_TIMEOUT}s per step"
echo "Work dir:  $(pwd)"
echo "Started:   $(date)"
echo "=========================================="

python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"

# Accept optional custom CSV as first positional argument
# Default: workflow/tasks.csv
# Custom:  sbatch --array=0-N%<throttle> goad_CHGNETMACE_kestrel.slurm workflow/tasks_custom.csv
TASKS_CSV="${1:-workflow/tasks.csv}"

echo "Tasks CSV: ${TASKS_CSV}"

python workflow/run_one_task.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --tasks-csv "${TASKS_CSV}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} finished: $(date)"
echo "=========================================="
