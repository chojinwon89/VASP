#!/bin/bash
#SBATCH --job-name=goad_array
#SBATCH --account=ccpc
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/goad_array-%A_%a.out
#SBATCH --error=slurm-logs/goad_array-%A_%a.err

set -euo pipefail

module purge
module load conda
conda activate goad

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/goad-global-optimization

mkdir -p slurm-logs runs

echo "=========================================="
echo "GOAD Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Host:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Started:   $(date)"
echo "=========================================="

python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"

python workflow/run_one_task.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --tasks-csv workflow/tasks.csv

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} finished: $(date)"
echo "=========================================="