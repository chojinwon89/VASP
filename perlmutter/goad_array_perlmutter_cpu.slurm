#!/bin/bash
# ============================================================================
# GOAD array job — Perlmutter (NERSC) CPU nodes  [FALLBACK]
# ----------------------------------------------------------------------------
# Use the GPU script instead unless you have no GPU allocation — the MLIP
# calculators (SevenNet / MatterSim) are 10-30x faster on an A100 than on CPU.
#
# CPU node = 2x AMD EPYC 7763 (128 physical cores / 256 logical), 512 GB.
# In `shared` QOS you can take up to half a node; here we take 1/4 node
# (-c 64 = 32 physical cores) and are charged only for that fraction.
#
# Submit from the repo root:
#   sbatch --array=0-199%20 perlmutter/goad_array_perlmutter_cpu.slurm workflow/tasks_custom.csv
# ============================================================================
#SBATCH -J goad_cpu
#SBATCH -A m5281          # your project (CPU hours). Delete to use Iris default.
#SBATCH -C cpu
#SBATCH -q shared                   # sub-node, fractional-node charging (max 0.5 node)
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 64                       # 64 logical = 32 physical cores (1/4 node). Max in shared: -c 128 (half node).
#SBATCH --requeue
#SBATCH -o slurm-logs/goad_cpu-%A_%a.out
#SBATCH -e slurm-logs/goad_cpu-%A_%a.err

set -euo pipefail

module load conda
conda activate "${GOAD_ENV:-$PSCRATCH/goad-env}"

export SLURM_CPU_BIND="cores"
# NOTE: $SLURM_CPUS_PER_TASK counts LOGICAL CPUs (2 per physical core). If you
# see thread oversubscription slowing things down, halve these to the physical
# core count.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export GOAD_N_WORKERS=1
export GOAD_STEP_TIMEOUT=300

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm-logs runs

echo "=========================================="
echo "GOAD CPU (Perlmutter) array task ${SLURM_ARRAY_TASK_ID}"
echo "Host:      $(hostname)  |  CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "Env:       ${GOAD_ENV:-$PSCRATCH/goad-env}"
echo "Started:   $(date)"
echo "=========================================="

python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"

TASKS_CSV="${1:-workflow/tasks_custom.csv}"
echo "Tasks CSV: ${TASKS_CSV}"

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

echo "Task ${REAL_TASK_ID} (array index ${SLURM_ARRAY_TASK_ID}) finished: $(date)"
