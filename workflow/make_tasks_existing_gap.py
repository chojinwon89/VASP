#!/usr/bin/env python3
"""
make_tasks_existing_gap.py
==========================
Emit a SevenNet-OMNI GOAD task list for the DFT systems that are still MISSING
an MLIP structure but are NOT covered by make_tasks_missing_sevennet.py.

make_tasks_missing_sevennet.py handles the 6 radical gap-fill species
(C2H2, CH3O, H, HCN, O, OH) + 8 extra open-shell radicals. What it does NOT
cover are the "existing"-group literature systems whose molecule already has a
gallery structure on other surfaces but is missing on a few specific ones -- in
practice ~22 systems, mostly Pt111 (CH3_Pt111, CO_Pt111, CH4_Pt111, ...) plus a
handful on Cu111/Pd111/Rh111.

This script finds those gaps directly (reusing check_dft_coverage's logic),
maps each molecule formula back to the gallery adsorbate input name the GOAD
pipeline expects (methanol, ethane, formic_acid, ...) by inverting the token
map the MANIFEST already encodes, and writes a task CSV whose schema matches
make_tasks_custom.py / make_tasks_missing_sevennet.py.

Usage
-----
    python workflow/make_tasks_existing_gap.py
    # -> workflow/tasks_existing_gap.csv  (+ prints the sbatch line)

Then on Perlmutter (needs the surface + molecule input CIFs generated first):
    sbatch --array=0-<N-1>%50 -t 00:20:00 \
        perlmutter/goad_array_perlmutter_gpu.slurm workflow/tasks_existing_gap.csv
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflow"))
from molecule_utils import carbon_count  # noqa: E402
from check_dft_coverage import (  # noqa: E402
    load_manifest, staged_dir_names, build_coverage)

# Handled by make_tasks_missing_sevennet.py -> excluded here so we don't double-run.
RADICALS = {"C2H2", "CH3O", "H", "HCN", "O", "OH",
            "CH", "C", "C2H5", "C2H3", "C2H", "CH2OH", "OOH", "COOH"}

POP, GEN = 60, 200


def seeds_for(n_c):
    if n_c <= 2:
        return [1, 2]
    if n_c <= 4:
        return [1, 2, 3]
    return [1, 2, 3, 4, 5]


def build_formula_to_name(rows, calc):
    """molecule formula -> gallery adsorbate input name, from filled rows.
    Ag100_methanol.cif (molecule=CH3OH) -> {'CH3OH': 'methanol'}."""
    f2n = {}
    for r in rows:
        g = (r.get("gallery_cif") or "").strip()
        if not g:
            continue
        stem = g[:-4] if g.lower().endswith(".cif") else g
        surface = r["metal"] + r["facet"]
        if stem.startswith(surface + "_"):
            stem = stem[len(surface) + 1:]
        for suf in (f"_{calc}", "_sevennet_omni", "_sevennet", "_mattersim"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        f2n.setdefault(r["molecule"], stem)
    return f2n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="DFT_results/MANIFEST.csv")
    ap.add_argument("--jobs-dir", default="dft_jobs")
    ap.add_argument("--exclude-groups", default="deoxy")
    ap.add_argument("--calc", default="sevennet_omni")
    ap.add_argument("--out", default="workflow/tasks_existing_gap.csv")
    ap.add_argument("--include-radicals", action="store_true",
                    help="Also emit the radical species (normally handled by "
                         "make_tasks_missing_sevennet.py).")
    args = ap.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (REPO / p)

    manifest = resolve(args.manifest)
    if not manifest.is_file():
        sys.exit(f"ERROR: manifest not found: {manifest}")
    excl = {g.strip() for g in args.exclude_groups.split(",") if g.strip()}

    rows = load_manifest(manifest, excl, set())
    staged = staged_dir_names(resolve(args.jobs_dir))
    phys = build_coverage(rows, staged)
    f2n = build_formula_to_name(rows, args.calc)

    missing = [p for p in phys.values() if p["status"] == "missing_no_cif"]
    if not args.include_radicals:
        missing = [p for p in missing if p["molecule"] not in RADICALS]

    tasks, unmapped = [], []
    tid = 0
    for p in sorted(missing, key=lambda x: (x["surface"], x["molecule"])):
        name = f2n.get(p["molecule"])
        if not name:
            unmapped.append(p["base_id"])
            continue
        n_c = carbon_count(name)
        for seed in seeds_for(n_c):
            tasks.append({
                "task_id": tid, "surface": p["surface"], "adsorbate": name,
                "seed": seed, "calculator": args.calc,
                "population_size": POP, "generations": GEN, "n_carbon": n_c})
            tid += 1

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if tasks:
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tasks[0].keys()))
            w.writeheader()
            w.writerows(tasks)

    n_sys = len({(t["surface"], t["adsorbate"]) for t in tasks})
    print("=" * 66)
    print("make_tasks_existing_gap")
    print("=" * 66)
    print(f"  missing (non-radical) systems : {n_sys}")
    print(f"  tasks written                 : {len(tasks)}  -> {out}")
    if tasks:
        print("\n  per-surface:")
        for surf, c in sorted(Counter(t["surface"] for t in tasks).items()):
            n = len({t["adsorbate"] for t in tasks if t["surface"] == surf})
            print(f"    {surf:8s} {n:2d} systems, {c:2d} tasks")
        print(f"\n  submit:  sbatch --array=0-{len(tasks) - 1}%50 -t 00:20:00 \\")
        print("             perlmutter/goad_array_perlmutter_gpu.slurm "
              f"{args.out}")
    if unmapped:
        print(f"\n  !! {len(unmapped)} system(s) had no gallery input name "
              f"(no filled row to invert): {', '.join(unmapped)}")
    if not tasks:
        print("\n  Nothing to generate -- no non-radical gaps. \u2713")


if __name__ == "__main__":
    main()
