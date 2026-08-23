#!/usr/bin/env python3
"""
make_dft_joblist.py
===================
Emit the list of VASP job directories (one per line) to feed the CPU job-array
submitter `perlmutter/vasp_dft_array_cpu.slurm`.

It scans a staged jobs tree (default `dft_jobs`, the layout setup_vasp_jobs.py
produces: <jobs-dir>/<system>/<FUNC>/{INCAR,...}) and prints every job dir for
the requested functional(s) that HAS an INCAR.  By default it SKIPS jobs whose
OUTCAR already shows a converged relax ("reached required accuracy"), so a
re-submission only picks up the remaining work -- handy for running in waves
under a limited allocation.

Usage
-----
    # PBE first, only the not-yet-converged ones:
    python workflow/make_dft_joblist.py --jobs-dir dft_jobs --functional pbe > joblist_pbe.txt

    # all four functionals in one list:
    python workflow/make_dft_joblist.py --jobs-dir dft_jobs --functional all > joblist_all.txt

    # include already-finished jobs too (force full re-run):
    python workflow/make_dft_joblist.py --functional pbe --include-done > joblist_pbe.txt

Then submit (throttle to 20 concurrent):
    N=$(grep -vc '^#' joblist_pbe.txt)
    sbatch --array=0-$((N-1))%20 perlmutter/vasp_dft_array_cpu.slurm joblist_pbe.txt
"""
import argparse
import sys
from pathlib import Path

# functional -> the subfolder name setup_vasp_jobs.py writes
FUNC_DIR = {
    "pbe":      "PBE",
    "pbe-d3":   "PBE_D3",
    "r2scan":   "r2scan",
    "beef-vdw": "beef_vdw",
}

CONVERGED_MARKER = "reached required accuracy"


def is_converged(outcar: Path) -> bool:
    """True if the OUTCAR shows a converged ionic relaxation."""
    if not outcar.is_file():
        return False
    try:
        # only the tail matters; avoid reading huge OUTCARs fully
        with outcar.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            tail = fh.read().decode("utf-8", errors="ignore")
    except Exception:
        return False
    return CONVERGED_MARKER in tail


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", default="dft_jobs",
                    help="Staged jobs tree (default: dft_jobs).")
    ap.add_argument("--functional", required=True,
                    choices=list(FUNC_DIR) + ["all"],
                    help="Functional subfolder(s) to list.")
    ap.add_argument("--include-done", action="store_true",
                    help="Do NOT skip already-converged jobs.")
    args = ap.parse_args()

    root = Path(args.jobs_dir)
    if not root.is_dir():
        sys.exit(f"ERROR: jobs dir not found: {root}")

    funcs = list(FUNC_DIR) if args.functional == "all" else [args.functional]

    n_listed = n_skipped = n_no_incar = 0
    for sysdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in funcs:
            jd = sysdir / FUNC_DIR[f]
            if not (jd / "INCAR").is_file():
                n_no_incar += 1
                continue
            if not args.include_done and is_converged(jd / "OUTCAR"):
                n_skipped += 1
                continue
            print(jd.as_posix())
            n_listed += 1

    print(f"# {n_listed} job(s) listed, {n_skipped} already-converged skipped "
          f"(functional={args.functional}, jobs-dir={root})", file=sys.stderr)
    if n_listed == 0:
        print("# nothing to run -- did you run setup_vasp_jobs.py for this "
              "functional yet?", file=sys.stderr)


if __name__ == "__main__":
    main()
