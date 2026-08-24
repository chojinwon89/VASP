#!/usr/bin/env python3
"""
resubmit_dft.py
===============
Scan a staged VASP DFT tree (the `dft_jobs/<system>/<FUNC>/` layout that
setup_vasp_jobs.py produces), classify every job by its OUTCAR, and rebuild the
array job list of the ones that still need to run -- then optionally submit it.

Job states
----------
    done         OUTCAR has "reached required accuracy"        -> skip
    running      OUTCAR modified within --running-window-min    -> skip (in flight)
    unconverged  OUTCAR finished ("General timing...") but not converged -> RESUBMIT
    crashed      OUTCAR exists, stale, no finish marker (timeout/crash)  -> RESUBMIT
    not_started  no OUTCAR at all                                        -> RESUBMIT

`running` jobs are NEVER resubmitted, so it is safe to run this while an array is
still draining -- but for a clean re-count, run it once `squeue` is empty.

Typical use (Perlmutter CPU)
----------------------------
    # scan + write the resubmit list + print the sbatch command (does NOT submit):
    python workflow/resubmit_dft.py --jobs-dir dft_jobs --functional pbe

    # same, but actually submit the array:
    python workflow/resubmit_dft.py --jobs-dir dft_jobs --functional pbe --submit

    # continue relaxations from the last geometry instead of restarting:
    python workflow/resubmit_dft.py --jobs-dir dft_jobs --functional pbe \
        --restart-from-contcar --submit

    # all four functionals at once (mixed list is fine -- each dir has its INCAR):
    python workflow/resubmit_dft.py --jobs-dir dft_jobs --functional all --submit

On Kestrel add `--cluster kestrel`.
"""
import argparse
import argparse
import mmap
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

FUNC_DIR = {
    "pbe":      "PBE",
    "pbe-d3":   "PBE_D3",
    "r2scan":   "r2scan",
    "beef-vdw": "beef_vdw",
}

ARRAY_SCRIPT = {
    "perlmutter-cpu": "perlmutter/vasp_dft_array_cpu.slurm",
    "kestrel":        "vasp_dft_array_kestrel.slurm",
}

CONVERGED = "reached required accuracy"
FINISHED = "General timing and accounting"
CONVERGED_B = CONVERGED.encode()
FINISHED_B = FINISHED.encode()
RESUBMIT_STATES = ("not_started", "crashed", "unconverged")


def outcar_markers(path: Path):
    """Scan the ENTIRE OUTCAR for the convergence / finish markers.

    Returns (has_converged, has_finished). We must scan the whole file, not a
    tail: after 'reached required accuracy' VASP writes a large final block
    (forces, stress, DOS, timing), so on big jobs the marker sits well outside
    any fixed tail window and a tail-only check misreports a converged run as
    crashed. mmap + find scans the full file at C speed via the page cache
    without loading it into Python memory.
    """
    try:
        if path.stat().st_size == 0:
            return (False, False)
        with path.open("rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                return (mm.find(CONVERGED_B) != -1, mm.find(FINISHED_B) != -1)
            finally:
                mm.close()
    except (ValueError, OSError):
        return (False, False)


def classify(jobdir: Path, running_window_s: float) -> str:
    outcar = jobdir / "OUTCAR"
    if not outcar.is_file():
        return "not_started"
    has_conv, has_fin = outcar_markers(outcar)
    if has_conv:
        return "done"
    if has_fin:
        return "unconverged"
    age = time.time() - outcar.stat().st_mtime
    return "running" if age < running_window_s else "crashed"


def contcar_usable(jobdir: Path) -> bool:
    """A CONTCAR worth restarting from: non-empty and structurally plausible."""
    c = jobdir / "CONTCAR"
    if not c.is_file() or c.stat().st_size == 0:
        return False
    lines = c.read_text(errors="ignore").splitlines()
    if len(lines) < 8:
        return False
    try:
        # line 7 (0-indexed 6) is the per-species counts on a VASP5 POSCAR/CONTCAR
        counts = [int(x) for x in lines[6].split()]
        return sum(counts) > 0
    except Exception:
        # maybe "Selective dynamics"/direct offset shifts things; be lenient
        return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", default="dft_jobs",
                    help="Staged jobs tree (default: dft_jobs).")
    ap.add_argument("--functional", required=True,
                    choices=list(FUNC_DIR) + ["all"])
    ap.add_argument("--cluster", default="perlmutter-cpu",
                    choices=list(ARRAY_SCRIPT),
                    help="Selects the array script for the sbatch command "
                         "(default: perlmutter-cpu).")
    ap.add_argument("--out", default=None,
                    help="Resubmit list path (default: joblist_<func>_resubmit.txt).")
    ap.add_argument("--running-window-min", type=float, default=30.0,
                    help="OUTCARs touched within this many minutes count as "
                         "'running' and are NOT resubmitted (default: 30).")
    ap.add_argument("--restart-from-contcar", action="store_true",
                    help="For crashed/unconverged jobs with a usable CONTCAR, "
                         "copy CONTCAR->POSCAR so the relax continues.")
    ap.add_argument("--throttle", type=int, default=20,
                    help="Max concurrent array tasks (the %%N in --array).")
    ap.add_argument("--submit", action="store_true",
                    help="Actually run sbatch (default: just print the command).")
    ap.add_argument("--list-systems", action="store_true",
                    help="Also print the system name under each non-done state.")
    args = ap.parse_args()

    root = Path(args.jobs_dir)
    if not root.is_dir():
        sys.exit(f"ERROR: jobs dir not found: {root}")
    window_s = args.running_window_min * 60.0
    funcs = list(FUNC_DIR) if args.functional == "all" else [args.functional]

    counts = Counter()
    per_state = defaultdict(list)
    resubmit = []
    restarted = 0

    for sysdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in funcs:
            jd = sysdir / FUNC_DIR[f]
            if not (jd / "INCAR").is_file():
                continue
            st = classify(jd, window_s)
            counts[st] += 1
            per_state[st].append(jd)
            if st in RESUBMIT_STATES:
                if (args.restart_from_contcar and st in ("crashed", "unconverged")
                        and contcar_usable(jd)):
                    shutil.copyfile(jd / "CONTCAR", jd / "POSCAR")
                    restarted += 1
                resubmit.append(jd)

    total = sum(counts.values())

    # ------------------------------------------------------------------ report
    print("=" * 66)
    print("DFT job scan")
    print("=" * 66)
    print(f"  jobs-dir   : {root}")
    print(f"  functional : {args.functional}   (jobs found: {total})")
    print(f"  running window: {args.running_window_min:g} min")
    print()
    order = ["done", "running", "unconverged", "crashed", "not_started"]
    label = {"done": "converged (done)     ",
             "running": "running (in flight)  ",
             "unconverged": "finished, unconverged",
             "crashed": "crashed / timed out  ",
             "not_started": "not started          "}
    for st in order:
        n = counts.get(st, 0)
        mark = "  ->skip " if st in ("done", "running") else "  ->RERUN"
        print(f"  {label[st]} : {n:4d}{mark if n else ''}")
        if args.list_systems and n and st != "done":
            for jd in per_state[st][:200]:
                print(f"        {jd.parent.name}/{jd.name}")
    print()
    if args.restart_from_contcar:
        print(f"  restarted from CONTCAR: {restarted}")
    print(f"  TO RESUBMIT: {len(resubmit)} job(s)")
    if counts.get("running"):
        print(f"  NOTE: {counts['running']} job(s) look active and were excluded; "
              f"re-run once `squeue` is empty for an exact count.")

    if not resubmit:
        print("\nNothing to resubmit. \u2713")
        return

    out_path = Path(args.out) if args.out else Path(
        f"joblist_{args.functional}_resubmit.txt")
    out_path.write_text("".join(jd.as_posix() + "\n" for jd in resubmit))
    print(f"\n  wrote resubmit list -> {out_path}  ({len(resubmit)} lines)")

    n = len(resubmit)
    array_script = ARRAY_SCRIPT[args.cluster]
    cmd = ["sbatch", f"--array=0-{n - 1}%{args.throttle}", array_script,
           str(out_path)]
    cmd_str = " ".join(cmd)

    if args.submit:
        if not Path(array_script).is_file():
            sys.exit(f"\nERROR: array script not found: {array_script} "
                     f"(run from the repo root, or fix --cluster).")
        print(f"\n  submitting: {cmd_str}\n")
        rc = subprocess.call(cmd)
        sys.exit(rc)
    else:
        print("\n  to submit, run:\n")
        print(f"      {cmd_str}\n")
        print("  (add --submit to have this script run it for you)")


if __name__ == "__main__":
    main()
