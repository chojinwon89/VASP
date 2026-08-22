#!/bin/bash
# ============================================================================
# setup_perlmutter_env.sh — build the GOAD conda env on Perlmutter (NERSC)
# ----------------------------------------------------------------------------
# Run this ONCE on a Perlmutter LOGIN node:
#     bash perlmutter/setup_perlmutter_env.sh
#
# Where to put the env (set GOAD_ENV before running):
#   * BEST: /global/common/software/m5281/goad-env
#       -> read-optimised, persistent, meant for software/conda envs.
#   * OK:   $PSCRATCH/goad-env   (the default below)
#       -> fast + big, but PURGED after ~8 weeks of no access. Re-run if purged.
#   * AVOID $HOME (40 GB quota, slow for the many small files in a conda env).
#
# A100 GPUs are compute capability sm_80 -> the CUDA 12.1 PyTorch wheels work.
# ============================================================================
set -euo pipefail

GOAD_ENV="${GOAD_ENV:-$PSCRATCH/goad-env}"
PYVER="${PYVER:-3.11}"

echo ">> Target env: $GOAD_ENV  (python $PYVER)"
echo ">> Set GOAD_ENV to change location (e.g. /global/common/software/m5281/goad-env)"

module load conda

if [ ! -d "$GOAD_ENV" ]; then
    echo ">> Creating conda env..."
    conda create -y -p "$GOAD_ENV" "python=$PYVER" pip
fi
conda activate "$GOAD_ENV"

echo ">> Installing PyTorch (CUDA 12.1 build for A100)..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo ">> Installing GOAD scientific stack..."
pip install ase numpy matplotlib
# RDKit is required by generate_molecule_cifs.py (SMILES molecules) and by the
# GA torsion detection. Needed for the full tasks_custom.csv set.
pip install rdkit

echo ">> Installing MLIP calculators used by tasks_custom.csv (sevennet_omni) + MatterSim..."
pip install sevenn
pip install mattersim || echo "!! mattersim failed to install — fine if you only use SevenNet."

# Optional extra calculators (uncomment if your tasks use them):
# pip install chgnet
# pip install mace-torch

echo
echo ">> Verifying imports (torch.cuda is expected to be False on a login node — check on a GPU node):"
python - <<'PY'
import importlib
for m in ["ase", "numpy", "torch", "sevenn", "rdkit"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  ERR {m}: {e!r}")
import torch
print("  torch.cuda.is_available():", torch.cuda.is_available(), "(login node has no GPU)")
PY

echo
echo ">> Done. Before submitting jobs, export the env path so the .slurm scripts find it:"
echo "     export GOAD_ENV=$GOAD_ENV"
echo ">> Verify the GPU is visible on a compute node:"
echo "     salloc -A m5281 -C gpu -q shared -t 0:15:00 -n1 -c32 --gpus=1"
echo "     conda activate $GOAD_ENV && python -c 'import torch; print(torch.cuda.get_device_name(0))'"
