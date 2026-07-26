#!/bin/bash
#SBATCH --job-name=goad_gpu
#SBATCH --account=ccpc
#SBATCH --partition=gpu-h100        # GPU partition on Kestrel (check: sinfo | grep gpu)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # Enough threads to feed the GPU; no need for 104
#SBATCH --mem=120G                  # capped: GPU nodes have less RAM than CPU nodes
#SBATCH --gpus=1                    # 1 GPU per task
#SBATCH --gres=gpu:1                # required on Kestrel to actually allocate the GPU
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/goad_gpu-%A_%a.out
#SBATCH --error=slurm-logs/goad_gpu-%A_%a.err

# GPU notes
# ---------
# On GPU the calculator runs on CUDA (single GPU per task).
# ProcessPoolExecutor workers in genetic_algorithm.py each load the
# calculator, but CUDA context is per-process — each worker will try
# to use the GPU. With 1 GPU and many workers this causes contention.
#
# GOAD_N_WORKERS=1 forces the main process to own the GPU exclusively.
# Energy evals (~0.02 s on H100) are fast enough that serial-per-generation
# is still 5-20x faster than CPU.
#
# To run many tasks in parallel on separate GPU nodes use the throttle:
#   sbatch --array=0-647%8 goad_array_kestrel_gpu.slurm workflow/tasks_custom.csv
#   ↑ 8 tasks at once, each on its own GPU node

set -euo pipefail

module purge
module load conda
conda activate /scratch/jcho5/envs/goad

# On GPU: let torch use CUDA; restrict CPU threads to avoid oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Force single-process evaluation so one process owns the GPU exclusively.
# Each get_potential_energy() call on H100 takes ~0.02s vs ~0.6s on CPU,
# so serial-per-generation is still ~30x faster than CPU-parallel.
export GOAD_N_WORKERS=1

# Per-step watchdog timeout in seconds (default 300 = 5 min).
export GOAD_STEP_TIMEOUT=300

cd $SLURM_SUBMIT_DIR
mkdir -p slurm-logs runs

echo "=========================================="
echo "GOAD GPU Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Host:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Array ID:  ${SLURM_ARRAY_TASK_ID}"
echo "CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "GPUs:      ${SLURM_GPUS:-1}"
echo "Workers:   ${GOAD_N_WORKERS}"
echo "Timeout:   ${GOAD_STEP_TIMEOUT}s per step"
echo "Work dir:  $(pwd)"
echo "Started:   $(date)"
echo "=========================================="

python -c "
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
"

TASKS_CSV="${1:-workflow/tasks_custom.csv}"
echo "Tasks CSV: ${TASKS_CSV}"

# Resolve real task id from 0-based array index mapping (set by find_missing_tasks.py).
# Falls back to raw SLURM_ARRAY_TASK_ID for manual sbatch where array index == task id.
if [ -n "${GOAD_TASK_ID_LIST:-}" ]; then
    read -r -a _ids <<< "${GOAD_TASK_ID_LIST}"
    REAL_TASK_ID="${_ids[${SLURM_ARRAY_TASK_ID}]}"
elif [ -n "${GOAD_TASK_ID_FILE:-}" ]; then
    REAL_TASK_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${GOAD_TASK_ID_FILE}")
else
    REAL_TASK_ID="${SLURM_ARRAY_TASK_ID}"
fi

python workflow/run_one_task.py \
    --task-id "${REAL_TASK_ID}" \
    --tasks-csv "${TASKS_CSV}"

echo "=========================================="
echo "Task ${REAL_TASK_ID} (array index ${SLURM_ARRAY_TASK_ID}) finished: $(date)"
echo "=========================================="
