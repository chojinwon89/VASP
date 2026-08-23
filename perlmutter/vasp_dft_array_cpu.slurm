#!/bin/bash
#SBATCH -A m5281
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -n 32
#SBATCH -c 2
#SBATCH -t 06:00:00
#SBATCH -J vasp_dft
#SBATCH -o slurm-logs/vasp_%A_%a.out
#SBATCH -e slurm-logs/vasp_%A_%a.err
#
# VASP DFT geometry optimization as a SLURM job ARRAY over a list of job dirs.
# Each array task runs ONE <system>/<FUNC>/ VASP job.
#
# Why an array + shared QOS:
#   * 377 systems x 4 functionals = ~1500 small slab relaxations. One full 128-core
#     node per job is wasteful and blows the allocation. The `shared` QOS charges
#     for only the cores you ask for (here 32/128 of a node), and the array keeps
#     it to a single, throttled submission.
#
# Prepare a job list (skips already-converged jobs) and submit:
#   mkdir -p slurm-logs
#   python workflow/make_dft_joblist.py --jobs-dir dft_jobs --functional pbe > joblist_pbe.txt
#   N=$(grep -vc '^#' joblist_pbe.txt)
#   sbatch --array=0-$((N-1))%20 perlmutter/vasp_dft_array_cpu.slurm joblist_pbe.txt
#
# Tuning:
#   * Bigger/slower systems (r2scan, beef-vdw): raise -n (e.g. 64) and/or -t.
#   * %20 caps concurrent tasks; lower it to be gentle on a small balance.
#   * Pure MPI (OMP=1). For better VASP parallel efficiency you may add NCORE=8
#     to the INCARs (setup_vasp_jobs.py leaves it unset to match the SP protocol).

set -uo pipefail

JOBLIST="${1:?usage: sbatch [--array=...] perlmutter/vasp_dft_array_cpu.slurm <joblist.txt>}"
if [ ! -f "$JOBLIST" ]; then
    echo "ERROR: job list not found: $JOBLIST" >&2
    exit 1
fi

# Map this array index -> the Nth non-comment, non-empty line of the job list.
JOBDIR=$(grep -v '^#' "$JOBLIST" | grep -v '^[[:space:]]*$' \
         | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$JOBDIR" ]; then
    echo "No job dir for array index $SLURM_ARRAY_TASK_ID in $JOBLIST -- nothing to do."
    exit 0
fi

echo "==================================================================="
echo "Array task $SLURM_ARRAY_TASK_ID  ->  $JOBDIR"
echo "Host: $(hostname)   Started: $(date)"
echo "==================================================================="

cd "$JOBDIR" || { echo "ERROR: cannot cd into $JOBDIR" >&2; exit 1; }

if [ ! -f INCAR ] || [ ! -f POSCAR ] || [ ! -f POTCAR ]; then
    echo "ERROR: missing INCAR/POSCAR/POTCAR in $JOBDIR (run setup_vasp_jobs.py + build POTCAR)" >&2
    exit 1
fi

# --- VASP module (NERSC CPU build; license-gated) --------------------------
module load vasp-tpc/6.4.2-cpu

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

srun --cpu-bind=cores vasp_std
rc=$?

echo "-------------------------------------------------------------------"
if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "RESULT: converged (reached required accuracy)"
elif grep -q "General timing and accounting" OUTCAR 2>/dev/null; then
    echo "RESULT: VASP finished but NOT converged (check EDIFFG / raise NSW / restart from CONTCAR)"
else
    echo "RESULT: VASP did not finish cleanly (rc=$rc) -- inspect OUTCAR/std*"
fi
echo "Finished: $(date)"
exit $rc
