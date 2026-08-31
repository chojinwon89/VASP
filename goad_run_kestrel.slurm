#!/bin/bash
#SBATCH --job-name=goad
#SBATCH --account=ccpc
#SBATCH --partition=standard
#SBATCH --nodes=2
#SBATCH --cpus-per-task=104
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# ---------- activate conda (same as your interactive shell) ----------
module purge
module load conda
conda activate /scratch/jcho5/envs/goad

# ---------- thread control (CPU-bound) ----------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---------- workdir ----------
cd /scratch/jcho5/goad-global-optimization

# ---------- run info ----------
echo "Host=$(hostname)  Job=$SLURM_JOB_ID  Partition=$SLURM_JOB_PARTITION  CPUs=$SLURM_CPUS_PER_TASK"
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
echo "Started: $(date)"
echo "================================================================"

# ---------- your workflow ----------
echo "[1/3] generate_surface_cifs.py"
python -u generate_surface_cifs.py

echo "[2/3] generate_molecule_cifs.py"
python -u generate_molecule_cifs.py

echo "[3/3] batch_isopropanol.py"
python -u batch_isopropanol.py

echo "================================================================"
echo "Ended: $(date)"
