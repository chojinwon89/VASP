#!/usr/bin/env python
"""
make_tasks_custom.py
====================
Generate a custom tasks CSV with per-pair control over:
  - which seeds to run
  - GA population_size
  - GA generations
  - calculator (sevennet_omni, mattersim, etc.)

Use this when you want to:
  - Add more seeds to uncertain/flexible systems
  - Rerun specific surface+molecule pairs with larger GA settings
  - Run the same pairs with multiple calculators for benchmarking
  - Append extra tasks without regenerating all 486

Output: workflow/tasks_custom.csv

Usage
-----
    # 1. Edit CUSTOM_TASKS below
    python workflow/make_tasks_custom.py

    # 2. Submit (N = total rows - 1 printed at end of script)
    sbatch --array=0-<N>%20 goad_array_kestrel.slurm workflow/tasks_custom.csv

NOTE: To run two calculators on the same pair, add two separate entries
      with different "calculator" values — one value per entry only.
"""

import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# DEFAULT GA settings (used when not overridden per pair)
# ---------------------------------------------------------------------------
DEFAULT = {
    "seeds":           [0, 1, 2, 3, 4, 5],
    "population_size": 60,
    "generations":     100,
    "calculator":      "sevennet_omni",   # one value only
}

# ---------------------------------------------------------------------------
# CUSTOM TASKS — edit this list
# Each entry: (surface, adsorbate, overrides_dict)
# Leave overrides={} to use all defaults
# To run two calculators: add two entries with different "calculator" values
# ---------------------------------------------------------------------------
CUSTOM_TASKS = [
    # --- sevennet_omni runs ---
    ("Cu111", "isopropanol", {"seeds": [0,1,2,3,4,5], "calculator": "sevennet_omni"}),
    ("Cu111", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "sevennet_omni"}),
    ("Cu110", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "sevennet_omni"}),
    ("Cu001", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "sevennet_omni"}),
    ("Cu111", "propanol",    {"seeds": [0,1,2], "population_size": 60, "generations": 100, "calculator": "sevennet_omni"}),
    ("Pt111", "propanol",    {"seeds": [0,1,2], "population_size": 60, "generations": 100, "calculator": "sevennet_omni"}),

    # --- mattersim runs — same pairs for benchmarking ---
    ("Cu111", "isopropanol", {"seeds": [0,1,2,3,4,5], "calculator": "mattersim"}),
    ("Cu111", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "mattersim"}),
    ("Cu110", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "mattersim"}),
    ("Cu001", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "mattersim"}),
    ("Cu111", "propanol",    {"seeds": [0,1,2], "population_size": 60, "generations": 100, "calculator": "mattersim"}),
    ("Pt111", "propanol",    {"seeds": [0,1,2], "population_size": 60, "generations": 100, "calculator": "mattersim"}),
]

# ---------------------------------------------------------------------------
# Generate CSV
# ---------------------------------------------------------------------------
out = Path("workflow/tasks_custom.csv")
out.parent.mkdir(exist_ok=True)

rows = []
task_id = 0

for surface, adsorbate, overrides in CUSTOM_TASKS:
    seeds           = overrides.get("seeds",           DEFAULT["seeds"])
    population_size = overrides.get("population_size", DEFAULT["population_size"])
    generations     = overrides.get("generations",     DEFAULT["generations"])
    calculator      = overrides.get("calculator",      DEFAULT["calculator"])

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
