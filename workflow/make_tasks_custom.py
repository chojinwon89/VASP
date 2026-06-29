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

Seeds
-----
Seed 0 is excluded: np.random.seed(0) produces a poor gen-1 population
for glycerol (+0.17 eV). Seeds 1-3 are used for both calculators.

Generations / Population
------------------------
All molecules use generations=200 and population_size=60.
Early stopping (patience=30, tol=0.001 eV) means the GA will terminate
well before 200 gens in practice — the high limit is just a safety cap.
"""

import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-calculator seed lists
# Seed 0 excluded — consistently bad gen-1 for glycerol with this RNG seed.
# ---------------------------------------------------------------------------
SEEDS_SEVENNET = [1, 2, 3]
SEEDS_5M       = [1, 2, 3]

# ---------------------------------------------------------------------------
# DEFAULT GA settings (fallback when not overridden per entry)
# ---------------------------------------------------------------------------
DEFAULT = {
    "seeds":           SEEDS_SEVENNET,
    "population_size": 60,
    "generations":     200,
    "calculator":      "sevennet_omni",
}

# ---------------------------------------------------------------------------
# Helper: build entries for a list of surfaces × molecules × calculators
# ---------------------------------------------------------------------------
def make_entries(surfaces, molecules, calculators, seeds=None, pop=60):
    seeds = seeds or DEFAULT["seeds"]
    entries = []
    for surf in surfaces:
        for mol, gen in molecules.items():
            for calc in calculators:
                calc_seeds = SEEDS_5M if calc == "5m" else SEEDS_SEVENNET
                entries.append(
                    (surf, mol, {"seeds": calc_seeds, "population_size": pop,
                                 "generations": gen, "calculator": calc})
                )
    return entries


# ---------------------------------------------------------------------------
# MOLECULES
# All use generations=200 and population_size=60.
# Early stopping will fire well before 200 gens in practice.
# ---------------------------------------------------------------------------
MOLECULES = {
    # Heavy oxygenates (more rotatable bonds → more generations needed)
    "glycerol":    200,
    "propanol":    200,
    "isopropanol": 200,
    "ethanol":     200,
    # Alkanes
    "propane":     200,
    "ethane":      200,
    # Alkenes
    "propene":     200,
    "ethene":      200,
    # Other
    "CO2":         200,
}

CALCS = ["sevennet_omni", "5m"]

# ---------------------------------------------------------------------------
# CUSTOM TASKS
# ---------------------------------------------------------------------------
CUSTOM_TASKS = (
    # Cu surfaces
    make_entries(["Cu111", "Cu110", "Cu001"], MOLECULES, CALCS) +
    # Pt surfaces
    make_entries(["Pt111", "Pt110", "Pt100"], MOLECULES, CALCS) +
    # Pd surfaces
    make_entries(["Pd111", "Pd110", "Pd100"], MOLECULES, CALCS) +
    # Ni surfaces
    make_entries(["Ni111", "Ni110", "Ni100"], MOLECULES, CALCS) +
    # Ag surfaces
    make_entries(["Ag111", "Ag110", "Ag100"], MOLECULES, CALCS) +
    # Au surfaces
    make_entries(["Au111", "Au110", "Au100"], MOLECULES, CALCS)
)

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
print("Molecule breakdown:")
from collections import Counter
mols = Counter(r["adsorbate"] for r in rows)
for mol, count in sorted(mols.items()):
    print(f"  {mol:<15}: {count} tasks")
print()
print("Metal breakdown:")
metals = Counter(r["surface"][:2] for r in rows)
for metal, count in sorted(metals.items()):
    print(f"  {metal}: {count} tasks")
print()
print("Calculator breakdown:")
calcs = Counter(r["calculator"] for r in rows)
for calc, count in sorted(calcs.items()):
    print(f"  {calc:<15}: {count} tasks")
print()
print("Submit with:")
print(f"  sbatch --array=0-{task_id-1}%20 goad_array_kestrel.slurm workflow/tasks_custom.csv")
