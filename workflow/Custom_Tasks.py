#!/usr/bin/env python
"""
Custom_Tasks.py
===============
Programmatically build CUSTOM_TASKS using loops instead of writing
each entry by hand.

Edit the short lists at the top (SURFACES, MOLECULES, CALCS, SEEDS
and MOL_SETTINGS), then run:

    python workflow/Custom_Tasks.py

This generates workflow/tasks_custom.csv, ready for:

    sbatch --array=0-<N>%20 goad_array_kestrel.slurm workflow/tasks_custom.csv

Available calculators:
    sevennet_omni   — SevenNet-OMNI (omat24, PBE+D3)
    5m              — MatterSim 5M
    5m_d3           — MatterSim 5M + D3 dispersion
    1m              — MatterSim 1M
"""

import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# EDIT THESE LISTS
# ---------------------------------------------------------------------------

SURFACES = [
    "Cu111", "Cu110", "Cu001",
    # "Pt111", "Pt110", "Pt100",   # uncomment to add more metals
    # "Ni111", "Pd111", "Ag111", "Au111",
]

CALCS = [
    "sevennet_omni",
    "5m",           # MatterSim 5M
    # "5m_d3",      # MatterSim 5M + D3 dispersion
    # "1m",         # MatterSim 1M
]

SEEDS = [0, 1, 2, 3, 4, 5]

# Per-molecule GA settings
# Add or remove molecules here; adjust population_size and generations as needed
MOL_SETTINGS = {
    "glycerol":    {"population_size": 60, "generations": 120},
    "propanol":    {"population_size": 60, "generations": 100},
    "isopropanol": {"population_size": 60, "generations": 100},
    # "CO2":       {"population_size": 40, "generations": 80},
    # "ethanol":   {"population_size": 40, "generations": 80},
}

# ---------------------------------------------------------------------------
# Build CUSTOM_TASKS automatically (no manual entry needed)
# ---------------------------------------------------------------------------
CUSTOM_TASKS = [
    (surf, mol, {**settings, "seeds": SEEDS, "calculator": calc})
    for surf in SURFACES
    for mol, settings in MOL_SETTINGS.items()
    for calc in CALCS
]

# ---------------------------------------------------------------------------
# Preview what will be generated
# ---------------------------------------------------------------------------
print(f"Surfaces:    {SURFACES}")
print(f"Molecules:   {list(MOL_SETTINGS.keys())}")
print(f"Calculators: {CALCS}")
print(f"Seeds:       {SEEDS}")
print(f"Total entries in CUSTOM_TASKS: {len(CUSTOM_TASKS)}")
print(f"Total tasks (rows in CSV):     {len(CUSTOM_TASKS) * len(SEEDS)}")
print()

# ---------------------------------------------------------------------------
# Generate CSV
# ---------------------------------------------------------------------------
out = Path("workflow/tasks_custom.csv")
out.parent.mkdir(exist_ok=True)

rows = []
task_id = 0

for surface, adsorbate, overrides in CUSTOM_TASKS:
    seeds           = overrides.get("seeds",           SEEDS)
    population_size = overrides.get("population_size", 40)
    generations     = overrides.get("generations",     80)
    calculator      = overrides.get("calculator",      "sevennet_omni")

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
