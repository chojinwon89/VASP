#!/usr/bin/env python
"""
make_tasks_custom.py
====================
Generate a custom tasks CSV with per-pair control over:
  - which seeds to run
  - GA population_size
  - GA generations

Use this when you want to:
  - Add more seeds to uncertain/flexible systems (e.g. glycerol)
  - Rerun specific surface+molecule pairs with larger GA settings
  - Append extra tasks without regenerating all 486

Output: workflow/tasks_custom.csv

Usage
-----
    # 1. Edit CUSTOM_TASKS below
    python workflow/make_tasks_custom.py

    # 2. Submit (N = total rows - 1 printed at end of script)
    sbatch --array=0-<N>%20 goad_array_kestrel.slurm workflow/tasks_custom.csv
"""

import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# DEFAULT GA settings (used when not overridden per pair)
# ---------------------------------------------------------------------------
DEFAULT = {
    "seeds":            [0, 1, 2],
    "population_size":  40,
    "generations":      80,
    "calculator":       "sevennet_omni",
}

# ---------------------------------------------------------------------------
# CUSTOM TASKS — edit this list
# Each entry: (surface, adsorbate, overrides_dict)
# Leave overrides={} to use all defaults
# ---------------------------------------------------------------------------
CUSTOM_TASKS = [
    # --- More seeds for uncertain systems ---
    ("Cu111", "isopropanol", {"seeds": [3, 4, 5, 6, 7]}),

    # --- Larger GA for flexible molecules ---
    ("Cu111", "glycerol",   {"seeds": [0,1,2,3,4], "population_size": 60, "generations": 120}),
    ("Cu110", "glycerol",   {"seeds": [0,1,2,3,4], "population_size": 60, "generations": 120}),
    ("Cu001", "glycerol",   {"seeds": [0,1,2,3,4], "population_size": 60, "generations": 120}),

    # --- Rerun propanol with bigger population ---
    ("Cu111", "propanol",   {"seeds": [0,1,2], "population_size": 50, "generations": 100}),
    ("Pt111", "propanol",   {"seeds": [0,1,2], "population_size": 50, "generations": 100}),

    # --- Standard rerun of specific pairs ---
    ("Pt111", "ethanol",    {}),
    ("Ni111", "CO2",        {}),
]

# ---------------------------------------------------------------------------
# Generate CSV
# ---------------------------------------------------------------------------
out = Path("workflow/tasks_custom.csv")
out.parent.mkdir(exist_ok=True)

rows = []
task_id = 0

for surface, adsorbate, overrides in CUSTOM_TASKS:
    seeds           = overrides.get("seeds",            DEFAULT["seeds"])
    population_size = overrides.get("population_size",  DEFAULT["population_size"])
    generations     = overrides.get("generations",      DEFAULT["generations"])
    calculator      = overrides.get("calculator",       DEFAULT["calculator"])

    for seed in seeds:
        rows.append({
            "task_id":         task_id,
            "surface":         surface,
            "adsorbate":       adsorbate,
            "seed":            seed,
            "calculator":      calculator,
            "population_size": population_size,
            "generations":     generations,
        })
        task_id += 1

with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {task_id} tasks to {out}")
print()
print("Submit with:")
print(f"  sbatch --array=0-{task_id-1}%20 goad_array_kestrel.slurm workflow/tasks_custom.csv")
