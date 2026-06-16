#!/usr/bin/env python

import json
import csv
from pathlib import Path

runs_dir = Path("runs")
out_csv = Path("workflow/summary.csv")


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


rows = []

for result_file in sorted(runs_dir.glob("*/result.json")):
    run_dir = result_file.parent
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

out_csv.parent.mkdir(exist_ok=True)

fieldnames = [
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

with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Wrote summary to {out_csv}")
print(f"Total runs found: {len(rows)}")
print(f"Finished: {sum(r['state'] == 'finished' for r in rows)}")
print(f"Failed:   {sum(r['state'] == 'failed' for r in rows)}")
if rows:
    print()
    format_table(rows, fieldnames)