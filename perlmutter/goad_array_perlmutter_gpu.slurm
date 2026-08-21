#!/bin/bash
# ============================================================================
# GOAD array job — Perlmutter (NERSC) GPU nodes  [PRIMARY / RECOMMENDED]
# ----------------------------------------------------------------------------
# One array task = one (surface, adsorbate, seed, calculator) GOAD run on 1 A100.
#
# NERSC guidance: jobs that use 1 or 2 GPUs must use the `shared` QOS. With
# --gpus=1 in `shared` you get 1 A100 + 16 physical cores + 64 GB and are
# charged only 1/4 of a node-hour — ideal for this embarrassingly-parallel array.
#
# Submit from the repo root (so $SLURM_SUBMIT_DIR == repo root):
#   sbatch --array=0-199%40 perlmutter/goad_array_perlmutter_gpu.slurm workflow/tasks_custom.csv
#            ^ 200 tasks, 40 running at once (each = 1/4 GPU node)
#
# For bulk resubmission of only unfinished work, let find_missing_tasks.py drive it:
#   python find_missing_tasks.py --slurm-script perlmutter/goad_array_perlmutter_gpu.slurm \
#                                --max-in-flight 4900
# (Its generated submit_missing.sh passes GOAD_TASK_ID_LIST, handled below.)
# ============================================================================
#SBATCH -J goad_gpu
#SBATCH -A m5281          # your project (GPU hours). Delete to use Iris default.
#SBATCH -C gpu                      # GPU nodes (A100). Use "gpu&hbm80g" for 80 GB A100s.
#SBATCH -q shared                   # shared QOS: 1-2 GPUs, fractional-node charging
#SBATCH -t 12:00:00                 # shared allows up to 48h; tune down after a test batch
#SBATCH -n 1                        # one process (the GA is single-process)
#SBATCH -c 32                       # 32 logical CPUs = 16 physical cores (the shared 1-GPU slice)
#SBATCH --gpus=1                    # REQUIRED: explicitly request the GPU
#SBATCH --requeue                   # safe to requeue (tasks are idempotent / skip-if-finished)
#SBATCH -o slurm-logs/goad_gpu-%A_%a.out
#SBATCH -e slurm-logs/goad_gpu-%A_%a.err

set -euo pipefail

# --- Environment ------------------------------------------------------------
module load conda
# GOAD_ENV points at the conda env built by perlmutter/setup_perlmutter_env.sh.
# Override by exporting GOAD_ENV before sbatch (submit_missing.sh uses --export=ALL).
conda activate "${GOAD_ENV:-$PSCRATCH/goad-env}"

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Serial energy evaluation: one process owns the GPU exclusively. On A100 each
# eval is ~0.02 s, so serial-per-generation is far faster than CPU-parallel and
# avoids CUDA-context contention from ProcessPoolExecutor workers.
export GOAD_N_WORKERS=1

# Per-step watchdog (seconds): a hung BFGS step / eval raises TimeoutError
# instead of freezing the whole 12 h allocation.
export GOAD_STEP_TIMEOUT=300

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs runs

echo "=========================================="
echo "GOAD GPU (Perlmutter) array task ${SLURM_ARRAY_TASK_ID}"
echo "Host:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "Env:       ${GOAD_ENV:-$PSCRATCH/goad-env}"
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

# Default production task list (52,440 tasks). Override as first positional arg.
TASKS_CSV="${1:-workflow/tasks_custom.csv}"
echo "Tasks CSV: ${TASKS_CSV}"

# Map the 0-based Slurm array index -> real task_id.
# find_missing_tasks.py exports GOAD_TASK_ID_LIST (space-separated) so array
# indices stay small (< MaxArraySize) while real task_ids can be large.
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
