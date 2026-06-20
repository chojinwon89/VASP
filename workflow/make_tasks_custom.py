#!/usr/bin/env python
"""
make_tasks_custom.py
====================
Generate a custom tasks CSV with per-pair control over:
  - which seeds to run
  - GA population_size
  - GA generations
  - calculator

Available calculators:
    sevennet_omni   — SevenNet-OMNI (omat24, PBE+D3)
    5m              — MatterSim 5M
    5m_d3           — MatterSim 5M + D3 dispersion
    1m              — MatterSim 1M

Two ways to define CUSTOM_TASKS:
  1. Manual list (this file) — explicit entry per surface+molecule+calculator
  2. Loop builder             — see workflow/Custom_Tasks.py

Output: workflow/tasks_custom.csv

Usage
-----
    python workflow/make_tasks_custom.py
    sbatch --array=0-<N>%20 goad_array_kestrel.slurm workflow/tasks_custom.csv

NOTE: one calculator value per entry only.
      To run two calculators, add two separate entries.
"""

import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# DEFAULT GA settings (fallback when not overridden per entry)
# ---------------------------------------------------------------------------
DEFAULT = {
    "seeds":           [0, 1, 2, 3, 4, 5],
    "population_size": 60,
    "generations":     100,
    "calculator":      "sevennet_omni",
}

# ---------------------------------------------------------------------------
# CUSTOM TASKS — edit this list manually
# Each entry: (surface, adsorbate, overrides_dict)
# Leave overrides={} to use all defaults
# ---------------------------------------------------------------------------
CUSTOM_TASKS = [
    # --- sevennet_omni ---
    ("Cu111", "isopropanol", {"seeds": [0,1,2,3,4,5], "calculator": "sevennet_omni"}),
    ("Cu111", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "sevennet_omni"}),
    ("Cu110", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "sevennet_omni"}),
    ("Cu001", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "sevennet_omni"}),
    ("Cu111", "propanol",    {"seeds": [0,1,2], "population_size": 60, "generations": 100, "calculator": "sevennet_omni"}),
    ("Pt111", "propanol",    {"seeds": [0,1,2], "population_size": 60, "generations": 100, "calculator": "sevennet_omni"}),

    # --- MatterSim 5M (same pairs for benchmarking) ---
    ("Cu111", "isopropanol", {"seeds": [0,1,2,3,4,5], "calculator": "5m"}),
    ("Cu111", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "5m"}),
    ("Cu110", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "5m"}),
    ("Cu001", "glycerol",    {"seeds": [0,1,2,3,4,5], "population_size": 60, "generations": 120, "calculator": "5m"}),
    ("Cu111", "propanol",    {"seeds": [0,1,2], "population_size": 60, "generations": 100, "calculator": "5m"}),
    ("Pt111", "propanol",    {"seeds": [0,1,2], "population_size": 60, "generations": 100, "calculator": "5m"}),
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
