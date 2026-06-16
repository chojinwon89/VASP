#!/usr/bin/env python

import json
import csv
from pathlib import Path

runs_dir = Path("runs")
out_csv = Path("workflow/summary.csv")

rows = []

for status_file in sorted(runs_dir.glob("*/status.json")):
    status = json.loads(status_file.read_text())

    run_dir = Path(status["run_dir"])
    result_jsons = list(run_dir.glob("*.json"))

    row = {
        "task_id": status.get("task_id"),
        "surface": status.get("surface"),
        "adsorbate": status.get("adsorbate"),
        "seed": status.get("seed"),
        "calculator": status.get("calculator"),
        "state": status.get("state"),
        "run_dir": str(run_dir),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "n_result_jsons": len(result_jsons),
    }

    # Optional: if your batch script writes a final result JSON,
    # parse it here and add E_ads.
    final_candidates = list(run_dir.glob("*final*.json")) + list(run_dir.glob("*result*.json"))
    if final_candidates:
        try:
            data = json.loads(final_candidates[0].read_text())
            row["E_ads"] = data.get("E_ads", data.get("adsorption_energy", ""))
        except Exception:
            row["E_ads"] = ""

    rows.append(row)

out_csv.parent.mkdir(exist_ok=True)

fieldnames = [
    "task_id",
    "surface",
    "adsorbate",
    "seed",
    "calculator",
    "state",
    "E_ads",
    "run_dir",
    "started_at",
    "finished_at",
    "n_result_jsons",
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