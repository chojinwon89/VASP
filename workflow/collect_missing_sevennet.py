#!/usr/bin/env python3
"""
collect_missing_sevennet.py
===========================
Post-processing for the gap-fill GOAD/SevenNet batch (tasks_missing_sevennet.csv).

After the Perlmutter array job finishes, this walks the `runs/` tree, and for
every (surface, adsorbate) system it:

  1. finds all completed seeds (status.json state == "finished" + a valid
     result.json + a final_adsorbed.cif),
  2. picks the BEST seed = lowest final adsorption energy E_ads_eV
     (GOAD's global-minimum candidate),
  3. copies that seed's relaxed structure into a flat gallery-style folder
     named `{surface}_{adsorbate}_sevennet_omni.cif` (matches the existing
     SevenNet gallery convention), and
  4. writes a tidy summary CSV (one row per system) with energetics and the
     molecule-surface bond distance -- the quantity this whole benchmark is
     about -- plus a completeness report to stdout.

The task CSV is the authoritative list of "molecules discussed for evaluation",
so anything in it that has NO finished seed is reported as MISSING (re-run it).

Usage
-----
    # from the repo root, after the array job drains:
    python workflow/collect_missing_sevennet.py

    # options:
    python workflow/collect_missing_sevennet.py \
        --tasks-csv workflow/tasks_missing_sevennet.csv \
        --runs-root runs \
        --out-dir   collected/sevennet_missing \
        --summary   collected/sevennet_missing_summary.csv
"""
import argparse
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from molecule_utils import carbon_count  # noqa: E402

SURFACE_RE = re.compile(r"^([A-Z][a-z]?)(\d{3})$")


def find_run_dir(runs_root: Path, surface: str, adsorbate: str, seed: int,
                 calculator: str):
    """Reconstruct the run dir; fall back to globbing across C-buckets."""
    name = f"{surface}_{adsorbate}_seed{seed}_{calculator}"
    direct = runs_root / f"C{carbon_count(adsorbate)}" / name
    if direct.is_dir():
        return direct
    hits = sorted(runs_root.glob(f"C*/{name}"))
    return hits[0] if hits else None


def read_seed_result(run_dir: Path, surface: str, adsorbate: str):
    """Return (ok, info) for one seed's run directory."""
    status = run_dir / "status.json"
    result = run_dir / "result.json"
    cif = run_dir / f"{adsorbate}_on_{surface}" / "final_adsorbed.cif"

    state = None
    if status.is_file():
        try:
            state = json.loads(status.read_text()).get("state")
        except Exception:
            state = "unreadable"
    if state == "failed":
        return False, {"reason": "status=failed"}
    if not result.is_file():
        return False, {"reason": "no result.json (incomplete/running)"}
    try:
        res = json.loads(result.read_text())
    except Exception:
        return False, {"reason": "result.json unreadable"}
    if "E_ads_eV" not in res:
        return False, {"reason": "result.json missing E_ads_eV"}
    if not cif.is_file():
        return False, {"reason": "no final_adsorbed.cif"}

    dfin = (res.get("distances") or {}).get("final") or {}
    return True, {
        "E_ads_eV":       float(res["E_ads_eV"]),
        "E_total_eV":     res.get("E_total_eV"),
        "E_surface_eV":   res.get("E_surface_eV"),
        "E_molecule_eV":  res.get("E_molecule_eV"),
        "E_ads_pre_relax_eV": res.get("E_ads_pre_relax_eV"),
        "min_dist_3d_A":  dfin.get("dist_3d"),
        "z_gap_A":        dfin.get("z_min"),
        "cif":            cif,
        "run_dir":        run_dir,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-csv", default="workflow/tasks_missing_sevennet.csv")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out-dir",   default="collected/sevennet_missing")
    ap.add_argument("--summary",   default="collected/sevennet_missing_summary.csv")
    ap.add_argument("--calculator", default=None,
                    help="override; default = whatever the CSV rows say")
    args = ap.parse_args()

    tasks_csv = (REPO / args.tasks_csv) if not Path(args.tasks_csv).is_absolute() else Path(args.tasks_csv)
    runs_root = (REPO / args.runs_root) if not Path(args.runs_root).is_absolute() else Path(args.runs_root)
    out_dir   = (REPO / args.out_dir)   if not Path(args.out_dir).is_absolute()   else Path(args.out_dir)
    summary   = (REPO / args.summary)   if not Path(args.summary).is_absolute()   else Path(args.summary)

    if not tasks_csv.is_file():
        sys.exit(f"ERROR: task CSV not found: {tasks_csv}")
    if not runs_root.is_dir():
        sys.exit(f"ERROR: runs root not found: {runs_root}")

    # Group the authoritative task list into systems -> seeds.
    systems = defaultdict(list)  # (surface, adsorbate, calculator) -> [seed,...]
    for r in csv.DictReader(tasks_csv.open()):
        calc = args.calculator or r["calculator"]
        systems[(r["surface"], r["adsorbate"], calc)].append(int(r["seed"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    complete, missing, partial = [], [], []

    for (surface, adsorbate, calc), seeds in sorted(systems.items()):
        m = SURFACE_RE.match(surface)
        metal, facet = (m.group(1), m.group(2)) if m else ("", "")
        seed_results, failures = {}, {}
        for seed in sorted(set(seeds)):
            run_dir = find_run_dir(runs_root, surface, adsorbate, seed, calc)
            if run_dir is None:
                failures[seed] = "run dir not found (not run yet)"
                continue
            ok, info = read_seed_result(run_dir, surface, adsorbate)
            if ok:
                seed_results[seed] = info
            else:
                failures[seed] = info["reason"]

        if not seed_results:
            missing.append((surface, adsorbate, failures))
            continue
        if len(seed_results) < len(set(seeds)):
            partial.append((surface, adsorbate, failures))

        best_seed = min(seed_results, key=lambda s: seed_results[s]["E_ads_eV"])
        best = seed_results[best_seed]

        out_cif = out_dir / f"{surface}_{adsorbate}_{calc}.cif"
        shutil.copyfile(best["cif"], out_cif)
        complete.append((surface, adsorbate))

        rows.append({
            "surface":            surface,
            "metal":              metal,
            "facet":              facet,
            "adsorbate":          adsorbate,
            "n_carbon":           carbon_count(adsorbate),
            "calculator":         calc,
            "best_seed":          best_seed,
            "seeds_done":         "/".join(str(s) for s in sorted(seed_results)),
            "n_seeds_done":       len(seed_results),
            "n_seeds_expected":   len(set(seeds)),
            "E_ads_eV":           round(best["E_ads_eV"], 6),
            "E_total_eV":         best["E_total_eV"],
            "E_surface_eV":       best["E_surface_eV"],
            "E_molecule_eV":      best["E_molecule_eV"],
            "E_ads_pre_relax_eV": best["E_ads_pre_relax_eV"],
            "min_surf_ads_dist_A": best["min_dist_3d_A"],
            "z_gap_A":            best["z_gap_A"],
            "cif":               str(out_cif.relative_to(REPO)) if out_cif.is_relative_to(REPO) else str(out_cif),
            "run_dir":           str(best["run_dir"].relative_to(REPO)) if best["run_dir"].is_relative_to(REPO) else str(best["run_dir"]),
        })

    if rows:
        with summary.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ------------------------------------------------------------------ report
    n_total = len(systems)
    print("=" * 68)
    print("Collected SevenNet gap-fill structures")
    print("=" * 68)
    print(f"  systems (surface x adsorbate) : {n_total}")
    print(f"  complete (>=1 seed finished)  : {len(complete)}")
    print(f"  partial  (some seeds missing) : {len(partial)}")
    print(f"  MISSING  (no seed finished)   : {len(missing)}")
    print(f"  structures written to         : {out_dir}")
    print(f"  summary CSV                   : {summary}")

    if partial:
        print("\n-- partial systems (used best available seed) --")
        for s, a, fails in partial:
            print(f"   {s:8s} {a:10s}  incomplete seeds: {fails}")

    if missing:
        print("\n-- MISSING systems (re-run these) --")
        for s, a, fails in missing:
            print(f"   {s:8s} {a:10s}  {fails}")
        # emit the task_ids so re-running is trivial
        miss_set = {(s, a) for s, a, _ in missing}
        ids = []
        for r in csv.DictReader(tasks_csv.open()):
            if (r["surface"], r["adsorbate"]) in miss_set:
                ids.append(r["task_id"])
        print(f"\n   re-run task_ids ({len(ids)}): {','.join(ids)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
