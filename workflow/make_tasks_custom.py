#!/usr/bin/env python
"""
make_tasks_custom.py
====================
Generate a custom tasks CSV with per-pair control over:
  - which seeds to run
  - GA population_size
  - GA generations
  - calculator

Both surfaces AND molecules are AUTO-DISCOVERED from inputs/*.cif
  - Surfaces : files matching <Metal><facet>.cif  e.g. Cu111.cif, Fe110.cif
  - Molecules: all other .cif files               e.g. glycerol.cif, 1-butene.cif

Workflow for adding new metals or molecules:
  1. Add to generate_surface_cifs.py or generate_molecule_cifs.py
  2. python generate_surface_cifs.py   (or generate_molecule_cifs.py)
  3. python workflow/make_tasks_custom.py   <- picks everything up automatically

Available calculators:
    sevennet_omni   - SevenNet-OMNI (omat24, PBE+D3)
    5m              - MatterSim 5M
    5m_d3           - MatterSim 5M + D3 dispersion
    1m              - MatterSim 1M

Output: workflow/tasks_custom.csv

Usage
-----
    python workflow/make_tasks_custom.py
    sbatch --array=0-<N>%20 goad_array_kestrel.slurm workflow/tasks_custom.csv

Seeds
-----
Seed 0 is excluded: np.random.seed(0) produces a poor gen-1 population
for glycerol (+0.17 eV). Seeds 1-3 are used for both calculators.

Generations / Population
------------------------
All molecules use generations=200 and population_size=60.
Early stopping (patience=30, tol=0.001 eV) means the GA will terminate
well before 200 gens in practice - the high limit is just a safety cap.
"""

import csv
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-calculator seed lists
# ---------------------------------------------------------------------------
SEEDS_SEVENNET = [1, 2, 3]
SEEDS_5M       = [1, 2, 3]

# ---------------------------------------------------------------------------
# DEFAULT GA settings
# ---------------------------------------------------------------------------
DEFAULT = {
    "seeds":           SEEDS_SEVENNET,
    "population_size": 60,
    "generations":     200,
    "calculator":      "sevennet_omni",
}

# ---------------------------------------------------------------------------
# CALCULATORS
# ---------------------------------------------------------------------------
CALCS = ["sevennet_omni", "5m"]

# ---------------------------------------------------------------------------
# inputs/ directory
# ---------------------------------------------------------------------------
INPUTS_DIR = Path("inputs")

# Pattern: one or two letters (Metal) followed by digits (facet)
# e.g. Cu111, Fe110, Ru0001, Mo100
_SURFACE_RE = re.compile(r'^[A-Z][a-z]?\d+$')

# ---------------------------------------------------------------------------
# AUTO-DISCOVER surfaces and molecules from inputs/*.cif
# ---------------------------------------------------------------------------

def discover_surfaces_and_molecules(inputs_dir: Path):
    """
    Scan inputs/*.cif and split into surfaces and molecules.

    Surface  : stem matches <Metal><facet> regex  e.g. Cu111, Ru0001
    Molecule : everything else                     e.g. glycerol, 1-butene
    """
    if not inputs_dir.exists():
        print(f"WARNING: inputs/ not found at {inputs_dir.resolve()}")
        print("         Run generate_surface_cifs.py and generate_molecule_cifs.py first.")
        return [], {}

    surfaces  = []
    molecules = {}   # name -> generations (all use DEFAULT)

    for cif in sorted(inputs_dir.glob("*.cif")):
        name = cif.stem
        if _SURFACE_RE.match(name):
            surfaces.append(name)
        else:
            molecules[name] = DEFAULT["generations"]

    return surfaces, molecules


# ---------------------------------------------------------------------------
# Helper: build task entries for surfaces x molecules x calculators
# ---------------------------------------------------------------------------
def make_entries(surfaces, molecules, calculators, pop=60):
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
# Build task list
# ---------------------------------------------------------------------------
ALL_SURFACES, ALL_MOLECULES = discover_surfaces_and_molecules(INPUTS_DIR)

if not ALL_SURFACES:
    print("ERROR: No surface CIFs found. Run generate_surface_cifs.py first.")
    raise SystemExit(1)

if not ALL_MOLECULES:
    print("ERROR: No molecule CIFs found. Run generate_molecule_cifs.py first.")
    raise SystemExit(1)

print(f"Discovered {len(ALL_SURFACES)} surfaces:")
for s in ALL_SURFACES:
    print(f"  {s}")
print()
print(f"Discovered {len(ALL_MOLECULES)} molecules:")
for m in sorted(ALL_MOLECULES):
    print(f"  {m}")
print()

CUSTOM_TASKS = make_entries(ALL_SURFACES, ALL_MOLECULES, CALCS)

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
    print(f"  {mol:<30}: {count} tasks")
print()
print("Surface breakdown:")
for surf, count in sorted(Counter(r["surface"] for r in rows).items()):
    print(f"  {surf:<10}: {count} tasks")
print()
print("Calculator breakdown:")
calcs = Counter(r["calculator"] for r in rows)
for calc, count in sorted(calcs.items()):
    print(f"  {calc:<15}: {count} tasks")
print()
print("Submit with:")
print(f"  sbatch --array=0-{task_id-1}%20 goad_array_kestrel.slurm workflow/tasks_custom.csv")
