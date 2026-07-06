#!/usr/bin/env python
"""
make_tasks_custom.py
====================
Generate a custom tasks CSV with per-pair control over:
  - which seeds to run
  - GA population_size
  - GA generations
  - calculator

Surfaces are AUTO-DISCOVERED from inputs/*.cif — no hardcoded list.
Just run generate_surface_cifs.py to add new metals, then re-run this script.

Molecules are defined in MOLECULES below (name -> generations).

Available calculators:
    sevennet_omni   — SevenNet-OMNI (omat24, PBE+D3)
    5m              — MatterSim 5M
    5m_d3           — MatterSim 5M + D3 dispersion
    1m              — MatterSim 1M

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
well before 200 gens in practice — the high limit is just a safety cap.
"""

import csv
import re
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
# MOLECULES  (name -> GA generations)
# Add new molecules here; they will be paired with ALL auto-discovered surfaces.
# ---------------------------------------------------------------------------
MOLECULES = {
    # Heavy oxygenates
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

# ---------------------------------------------------------------------------
# CALCULATORS
# ---------------------------------------------------------------------------
CALCS = ["sevennet_omni", "5m"]

# ---------------------------------------------------------------------------
# AUTO-DISCOVER surfaces from inputs/*.cif
# Rules:
#   - Filename must match a known surface pattern: <Metal><facet>.cif
#     e.g. Cu111.cif, Fe110.cif, Ru0001.cif
#   - Molecule CIFs (no digits at start, or known molecule names) are excluded.
#   - Cu001 alias is included.
# ---------------------------------------------------------------------------
INPUTS_DIR = Path("inputs")

# Pattern: one or two uppercase letters followed by digits (surface name)
_SURFACE_RE = re.compile(r'^[A-Z][a-z]?\d+$')

# Known molecule CIF names to explicitly exclude
_MOLECULE_NAMES = set(MOLECULES.keys()) | {
    "H2", "H2O", "CO", "CO2", "methanol", "formic_acid",
    "ethylene", "acetaldehyde", "acetic_acid", "DME",
    "butyric_acid", "1-butene", "isobutene", "butadiene", "methylmethacrylate",
    "valeric_acid", "1-pentene", "2-pentanone", "cyclopentanone",
    "furfural", "isoprene", "itaconic_acid", "caproic_acid",
    "5-HMF", "benzene", "5-heptanone", "toluene",
    "lactic_acid", "pyruvic_acid", "3-hydroxypropionic_acid", "3-MTHF",
    "propionic_acid",
}

def discover_surfaces(inputs_dir: Path) -> list:
    """
    Scan inputs/ for surface CIFs and return a sorted list of surface names.
    A CIF is treated as a surface if its stem matches <Metal><facet> pattern
    and is not in the known molecule list.
    """
    if not inputs_dir.exists():
        print(f"WARNING: inputs/ directory not found at {inputs_dir.resolve()}")
        print("         Run generate_surface_cifs.py first.")
        return []

    surfaces = []
    for cif in sorted(inputs_dir.glob("*.cif")):
        name = cif.stem
        if name in _MOLECULE_NAMES:
            continue
        if _SURFACE_RE.match(name):
            surfaces.append(name)
    return surfaces


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
ALL_SURFACES = discover_surfaces(INPUTS_DIR)

if not ALL_SURFACES:
    print("ERROR: No surface CIFs found in inputs/. Run generate_surface_cifs.py first.")
    raise SystemExit(1)

print(f"Discovered {len(ALL_SURFACES)} surfaces in inputs/:")
for s in ALL_SURFACES:
    print(f"  {s}")
print()

CUSTOM_TASKS = make_entries(ALL_SURFACES, MOLECULES, CALCS)

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
    print(f"  {mol:<25}: {count} tasks")
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
