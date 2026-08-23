#!/bin/bash
#SBATCH --job-name=vasp_dft
#SBATCH --account=ccpc
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/vasp_dft-%A_%a.out
#SBATCH --error=slurm-logs/vasp_dft-%A_%a.err
#
# VASP DFT geometry optimization as a SLURM job ARRAY on Kestrel (CPU).
# Each array task runs ONE <system>/<FUNC>/ VASP job.
#
# Why an array: 377 systems x 4 functionals = ~1500 small slab relaxations.
# One array + a throttle (%N) is far tidier than 1500 separate sbatch calls and
# respects per-user queue limits.
#
# Prepare a job list (skips already-converged jobs) and submit:
#   mkdir -p slurm-logs
#   python workflow/make_dft_joblist.py --jobs-dir dft_jobs --functional pbe > joblist_pbe.txt
#   N=$(grep -vc '^#' joblist_pbe.txt)
#   sbatch --array=0-$((N-1))%30 vasp_dft_array_kestrel.slurm joblist_pbe.txt
#
# Resources: these benchmark slabs are small (~38 atoms, p(3x3)x4), so 1 full
# node (104 cores) pure-MPI is plenty. For the heavier functionals (r2scan,
# beef-vdw) raise --time, or --nodes=2 with --ntasks-per-node=104 if a job needs
# more memory/speed. If you see poor scaling on 104 ranks, add NCORE=8 to the
# INCARs (setup_vasp_jobs.py leaves NCORE unset to match the single-point setup).

set -uo pipefail

JOBLIST="${1:?usage: sbatch [--array=...] vasp_dft_array_kestrel.slurm <joblist.txt>}"
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
    echo "ERROR: missing INCAR/POSCAR/POTCAR in $JOBDIR (run setup_vasp_jobs.py first)" >&2
    exit 1
fi

# --- VASP module + threading (mirrors the proven Kestrel single-point setup) --
export OMP_NUM_THREADS=1
export OMP_STACKSIZE=1G
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
ulimit -s unlimited

module load vasp/6.3.2_openMP+tpc

srun vasp_std
rc=$?

echo "-------------------------------------------------------------------"
if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "RESULT: converged (reached required accuracy)"
elif grep -q "General timing and accounting" OUTCAR 2>/dev/null; then
    echo "RESULT: VASP finished but NOT converged (raise NSW / restart from CONTCAR)"
else
    echo "RESULT: VASP did not finish cleanly (rc=$rc) -- inspect OUTCAR/std*"
fi
echo "Finished: $(date)"
exit $rc
