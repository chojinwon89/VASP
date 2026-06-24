#!/usr/bin/env python
"""
collect_results.py
==================
Scan one or more runs directories for result.json files and write a
consolidated summary CSV.

Usage
-----
# Single runs directory, custom output filename
python collect_results.py \\
    --runs-dir /kfs3/scratch/jcho5/goad-global-optimization/runs \\
    --out results/summary_gpu_2026-06-24.csv

# Multiple runs directories merged into one file
python collect_results.py \\
    --runs-dir /home/jcho5/goad-global-optimization/runs \\
    --runs-dir /kfs3/scratch/jcho5/goad-global-optimization/runs \\
    --out workflow/summary_all.csv

Arguments
---------
--runs-dir / -r   Path to a runs directory. Repeatable for multiple locations.
--out      / -o   Output CSV path (directory will be created if needed).
"""

import argparse
import json
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect GOAD run results into a summary CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir", "-r",
        dest="runs_dirs",
        action="append",
        required=True,
        metavar="DIR",
        help=(
            "Path to a runs directory containing <run_name>/result.json files. "
            "May be repeated to merge multiple locations."
        ),
    )
    parser.add_argument(
        "--out", "-o",
        required=True,
        metavar="FILE",
        help=(
            "Output CSV file path, e.g. workflow/summary.csv or "
            "results/summary_gpu_2026-06-24.csv. "
            "Parent directories are created automatically."
        ),
    )
    return parser.parse_args()


def parse_system(system_name: str) -> tuple[str, str]:
    adsorbate, _, surface = system_name.partition("_on_")
    return surface, adsorbate


def format_table(rows, fieldnames):
    table = [[str(row.get(field, "")) for field in fieldnames] for row in rows]
    widths = [len(field) for field in fieldnames]
    for line in table:
        for idx, value in enumerate(line):
            widths[idx] = max(widths[idx], len(value))
    header = "  ".join(field.ljust(widths[idx]) for idx, field in enumerate(fieldnames))
    separator = "  ".join("-" * widths[idx] for idx in range(len(fieldnames)))
    print(header)
    print(separator)
    for line in table:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(line)))


def collect(runs_dirs: list[Path]) -> list[dict]:
    rows = []
    seen_run_dirs: set[str] = set()

    for base in runs_dirs:
        base = Path(base)
        if not base.exists():
            print(f"WARNING: runs directory not found, skipping: {base}")
            continue

        for result_file in sorted(base.glob("*/result.json")):
            run_dir = result_file.parent
            # Deduplicate in case the same physical path appears under two bases
            resolved = str(run_dir.resolve())
            if resolved in seen_run_dirs:
                continue
            seen_run_dirs.add(resolved)

            data = json.loads(result_file.read_text())
            status_file = run_dir / "status.json"
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            resolved_run_dir = status.get("run_dir", str(run_dir))

            surface, adsorbate = parse_system(data.get("system", ""))

            row = {
                "task_id": status.get("task_id", ""),
                "surface": status.get("surface", surface),
                "adsorbate": status.get("adsorbate", adsorbate),
                "seed": status.get("seed", data.get("ga", {}).get("best_seed", "")),
                "calculator": status.get("calculator", data.get("calculator", "")),
                "state": status.get("state", "finished"),
                "E_ads_eV": data.get("E_ads_eV", ""),
                "E_ads_pre_relax_eV": data.get("E_ads_pre_relax_eV", data.get("E_ads_pre_relax", "")),
                "best_seed": data.get("ga", {}).get("best_seed", ""),
                "run_dir": resolved_run_dir,
                "started_at": status.get("started_at", ""),
                "finished_at": status.get("finished_at", ""),
            }
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["task_id"] == "",
            int(row["task_id"]) if row["task_id"] != "" else 0,
            row["run_dir"],
        )
    )
    return rows


FIELDNAMES = [
    "task_id",
    "surface",
    "adsorbate",
    "seed",
    "calculator",
    "state",
    "E_ads_eV",
    "E_ads_pre_relax_eV",
    "best_seed",
    "run_dir",
    "started_at",
    "finished_at",
]


def main():
    args = parse_args()
    out_csv = Path(args.out)

    rows = collect(args.runs_dirs)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote summary to {out_csv}")
    print(f"Total runs found: {len(rows)}")
    print(f"Finished: {sum(r['state'] == 'finished' for r in rows)}")
    print(f"Failed:   {sum(r['state'] == 'failed'   for r in rows)}")
    if rows:
        print()
        format_table(rows, FIELDNAMES)


if __name__ == "__main__":
    main()
