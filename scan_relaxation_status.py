#!/usr/bin/env python3
"""Scan VASP job dirs and classify each as single-point vs relaxation, and
whether a relaxation actually finished (force-converged).

Usage:
    python scan_relaxation_status.py [ROOT ...]        # default ROOT: vasp_mol
    python scan_relaxation_status.py vasp_mol dft_jobs
    python scan_relaxation_status.py --csv status.csv vasp_mol

The scan recurses (rglob), so it finds INCARs at any depth. On Kestrel the
dft_jobs layout is dft_jobs/<sys>/poscar/best/<FUNC>/INCAR — just pass
`dft_jobs` and every nested job dir is picked up automatically.

Classification per job dir (a dir containing an INCAR):
  INTENT   (from INCAR):   single-point (NSW=0 or IBRION=-1) | relax (NSW>0)
  OUTCOME  (from OUTCAR):
     no-outcar        -> not run yet
     running/partial  -> OUTCAR exists but no completion line
     single-point     -> exactly 1 ionic step
     relaxed          -> multi-step AND "reached required accuracy" printed
     relax-unconverged-> multi-step but NSW hit / no accuracy line
     relax-1step      -> intent relax but only 1 ionic step recorded
"""
import argparse
import csv
import re
import sys
from pathlib import Path

INCAR_INT = re.compile(r"^\s*([A-Z_]+)\s*=\s*([-+]?\d+)", re.M)


def read_incar_intent(incar: Path):
    """Return (nsw, ibrion) as ints, or (None, None) if unreadable."""
    try:
        text = incar.read_text(errors="ignore")
    except OSError:
        return None, None
    vals = {}
    for m in INCAR_INT.finditer(text):
        key = m.group(1)
        if key in ("NSW", "IBRION") and key not in vals:
            vals[key] = int(m.group(2))
    return vals.get("NSW"), vals.get("IBRION")


OUTCAR_NSW = re.compile(r"NSW\s*=\s*(\d+)")
OUTCAR_IBRION = re.compile(r"IBRION\s*=\s*([-+]?\d+)")


def scan_outcar(outcar: Path):
    """Stream OUTCAR once, returning:
    (n_ionic_steps, reached_accuracy, finished, nsw, ibrion).
    NSW/IBRION are parsed from the OUTCAR header (fallback when INCAR absent).
    """
    n_ionic = 0
    reached = False
    finished = False
    nsw = ibrion = None
    try:
        with outcar.open(errors="ignore") as fh:
            for line in fh:
                # One of these appears once per ionic step in the OUTCAR
                if "aborting loop because EDIFF is reached" in line:
                    n_ionic += 1
                elif "reached required accuracy - stopping structural" in line:
                    reached = True
                elif "General timing and accounting" in line:
                    finished = True
                elif nsw is None and "NSW" in line:
                    m = OUTCAR_NSW.search(line)
                    if m:
                        nsw = int(m.group(1))
                elif ibrion is None and "IBRION" in line:
                    m = OUTCAR_IBRION.search(line)
                    if m:
                        ibrion = int(m.group(1))
    except OSError:
        return 0, False, False, None, None
    return n_ionic, reached, finished, nsw, ibrion


def classify(job_dir: Path):
    nsw, ibrion = read_incar_intent(job_dir / "INCAR")

    outcar = job_dir / "OUTCAR"
    if not outcar.exists():
        intent = "unknown"
        if nsw is not None:
            intent = "single-point" if (nsw == 0 or ibrion == -1) else "relax"
        return intent, "no-outcar", 0, False

    n_ionic, reached, finished, o_nsw, o_ibrion = scan_outcar(outcar)

    # Prefer INCAR intent; fall back to OUTCAR header when INCAR is absent.
    if nsw is None:
        nsw, ibrion = o_nsw, o_ibrion
    intent = "unknown"
    if nsw is not None:
        intent = "single-point" if (nsw == 0 or ibrion == -1) else "relax"

    if n_ionic == 0:
        outcome = "running/partial" if not finished else "empty-outcar"
    elif intent == "single-point" or n_ionic == 1:
        outcome = "single-point" if intent != "relax" else "relax-1step"
    elif reached:
        outcome = "relaxed"
    else:
        outcome = "relax-unconverged"
    return intent, outcome, n_ionic, reached


def find_job_dirs(root: Path):
    """Any dir containing an OUTCAR (ran) or an INCAR (set up but not run)."""
    seen = set()
    for marker in ("OUTCAR", "INCAR"):
        for f in root.rglob(marker):
            d = f.parent
            if d not in seen:
                seen.add(d)
                yield d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=["vasp_mol"],
                    help="Root dir(s) to scan (default: vasp_mol)")
    ap.add_argument("--csv", help="Write full per-dir results to this CSV")
    ap.add_argument("--only", help="Only print rows whose outcome matches this "
                    "(e.g. relaxed, single-point, relax-unconverged, no-outcar)")
    args = ap.parse_args()

    rows = []
    for root_name in args.roots:
        root = Path(root_name)
        if not root.exists():
            print(f"WARNING: {root} does not exist, skipping", file=sys.stderr)
            continue
        for job_dir in sorted(find_job_dirs(root)):
            intent, outcome, n_ionic, reached = classify(job_dir)
            rows.append((job_dir.as_posix(), intent, outcome, n_ionic, reached))

    # Print table
    if args.only:
        shown = [r for r in rows if r[2] == args.only]
    else:
        shown = rows
    w = max((len(r[0]) for r in shown), default=20)
    print(f"{'DIR':<{w}}  {'INTENT':<12}  {'OUTCOME':<18}  {'IONIC':>5}  ACC")
    print("-" * (w + 45))
    for path, intent, outcome, n_ionic, reached in shown:
        print(f"{path:<{w}}  {intent:<12}  {outcome:<18}  {n_ionic:>5}  "
              f"{'yes' if reached else '-'}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY (by outcome):")
    counts = {}
    for r in rows:
        counts[r[2]] = counts.get(r[2], 0) + 1
    for k in sorted(counts):
        print(f"  {k:<20} {counts[k]}")
    print(f"  {'TOTAL':<20} {len(rows)}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wtr = csv.writer(fh)
            wtr.writerow(["dir", "intent", "outcome", "ionic_steps", "reached_accuracy"])
            wtr.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
