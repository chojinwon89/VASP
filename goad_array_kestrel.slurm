#!/bin/bash
#SBATCH --job-name=goad_array
#SBATCH --account=ccpc
#SBATCH --partition=standard
#SBATCH --nodes=1                       # 1 node per task (GA is single-node)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=104             # full 104 cores per node for threading
#SBATCH --mem=0                         # use all available memory on node
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/goad_array-%A_%a.out
#SBATCH --error=slurm-logs/goad_array-%A_%a.err

# NOTE: The GA in this code is single-node (no MPI/multiprocessing).
# To use multiple nodes, increase the array throttle limit (e.g. %40)
# so more tasks run simultaneously on separate nodes:
#   sbatch --array=0-107%40 goad_array_kestrel.slurm workflow/tasks_custom.csv
#   ↑ This runs 40 independent tasks at once, each on its own node (= 40 nodes)

set -euo pipefail

module purge
module load conda
conda activate /scratch/jcho5/envs/goad

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Always cd to the directory where sbatch was submitted from
cd $SLURM_SUBMIT_DIR

mkdir -p slurm-logs runs

echo "=========================================="
echo "GOAD Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Host:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "Work dir:  $(pwd)"
echo "Started:   $(date)"
echo "=========================================="

python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"

# Accept optional custom CSV as first positional argument
# Default: workflow/tasks.csv
# Custom:  sbatch --array=0-N%<throttle> goad_array_kestrel.slurm workflow/tasks_custom.csv
TASKS_CSV="${1:-workflow/tasks.csv}"

echo "Tasks CSV: ${TASKS_CSV}"

python workflow/run_one_task.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --tasks-csv "${TASKS_CSV}"

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} finished: $(date)"
echo "=========================================="
