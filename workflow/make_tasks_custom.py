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

Seed counts by carbon number
-----------------------------
  C0-C2 : seeds 1-3  (3 seeds x 2 calcs =  6 runs per surface)
  C3+   : seeds 1-6  (6 seeds x 2 calcs = 12 runs per surface)

Rationale for C3+ having more seeds:
  - Larger conformational space -> more variation between seeds
  - Early stopping (patience=30, tol=0.001 eV) means each seed will
    finish well before the 200-generation cap if conditions are met.
  - 6 seeds gives better statistical coverage at low extra cost.

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
"""

import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import carbon_count from batch_isopropanol.py (single source of truth)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from batch_isopropanol import carbon_count

# ---------------------------------------------------------------------------
# Seed lists
# C0-C2: 3 seeds   C3+: 6 seeds
# Seed 0 excluded — consistently bad gen-1 for glycerol with this RNG seed.
# ---------------------------------------------------------------------------
SEEDS_SEVENNET      = [1, 2, 3]          # C0-C2
SEEDS_5M            = [1, 2, 3]          # C0-C2
SEEDS_SEVENNET_C3   = [1, 2, 3, 4, 5, 6]  # C3+
SEEDS_5M_C3         = [1, 2, 3, 4, 5, 6]  # C3+

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
# Explicit set of known molecule names that must NEVER be classified as
# surfaces, even if their CIF stem happens to match _SURFACE_RE.
# This prevents short inorganic names like H2, O2, N2, NO from being
# mis-bucketed as metal+facet entries.
#
# Covers every molecule defined in:
#   - generate_molecule_cifs.py  (ASE_NAMED + SMILES_MOLECULES)
#   - batch_isopropanol.py       (MOLECULE_SMILES)
#   - setup_molecule_jobs.py     (MOLECULE_REGISTRY)
# ---------------------------------------------------------------------------
KNOWN_MOLECULE_NAMES: set = {
    # Inorganics / simple gases
    "H2", "O2", "N2", "CO", "NO",
    "CO2", "NO2", "SO2", "H2S", "NH3", "H2O",
    # C1 references
    "CH4", "methane", "methanol", "formaldehyde",
    "formate", "carbonate", "HCO", "CH2", "CH3",
    # Alkanes
    "ethane", "propane", "butane", "isobutane",
    "pentane", "isopentane", "hexane", "heptane", "octane",
    # Alkenes
    "ethylene", "ethene", "propene", "1-butene", "2-butene",
    "isobutene", "1-pentene", "butadiene", "isoprene",
    # Aromatics
    "benzene", "toluene", "furan", "pyrrole", "thiophene",
    "styrene", "xylene", "phenol", "aniline", "naphthalene",
    # Alcohols
    "ethanol", "isopropanol", "propanol", "glycerol",
    "1-butanol", "2-butanol", "pentanol", "sorbitol", "xylitol",
    # Aldehydes
    "acetaldehyde", "furfural", "5-HMF",
    "propanal", "butanal", "valeraldehyde", "hexanal",
    "benzaldehyde", "5-methylfurfural",
    # Ketones
    "acetone", "methylethylketone", "cyclobutanone",
    "2-pentanone", "2-hexanone", "cyclopentanone", "cyclohexanone",
    "acetophenone", "5-heptanone", "2-heptanone",
    # Carboxylic acids
    "formic_acid", "acetic_acid", "propionic_acid", "butyric_acid",
    "valeric_acid", "caproic_acid", "oxalic_acid", "malonic_acid",
    "succinic_acid", "glutaric_acid",
    # Hydroxy/keto acids
    "lactic_acid", "pyruvic_acid", "3-hydroxypropionic_acid",
    "itaconic_acid", "glycolic_acid", "malic_acid", "tartaric_acid",
    "levulinic_acid", "citric_acid", "gluconic_acid", "muconic_acid",
    # Esters/ethers
    "DME", "DMSO", "3-MTHF", "methylmethacrylate",
    "diethyl_ether", "THF", "ethyl_acetate",
    "furfuryl_alcohol", "gamma_valerolactone", "dimethyl_succinate",
}

# ---------------------------------------------------------------------------
# AUTO-DISCOVER surfaces and molecules from inputs/*.cif
# ---------------------------------------------------------------------------

def discover_surfaces_and_molecules(inputs_dir: Path):
    """
    Scan inputs/*.cif and split into surfaces and molecules.

    Surface  : stem matches <Metal><facet> regex AND is NOT in KNOWN_MOLECULE_NAMES
    Molecule : everything in KNOWN_MOLECULE_NAMES, OR stems that don't match the regex

    The KNOWN_MOLECULE_NAMES check takes priority over the regex so that
    short inorganic names like H2, O2, N2, NO are never mis-bucketed as
    metal+facet surfaces.
    """
    if not inputs_dir.exists():
        print(f"WARNING: inputs/ not found at {inputs_dir.resolve()}")
        print("         Run generate_surface_cifs.py and generate_molecule_cifs.py first.")
        return [], {}

    surfaces  = []
    molecules = {}   # name -> generations

    for cif in sorted(inputs_dir.glob("*.cif")):
        name = cif.stem
        # Known molecule names are always treated as molecules, regardless of
        # whether their stem happens to match the surface regex (e.g. H2, NO).
        if name in KNOWN_MOLECULE_NAMES:
            molecules[name] = DEFAULT["generations"]
        elif _SURFACE_RE.match(name):
            surfaces.append(name)
        else:
            molecules[name] = DEFAULT["generations"]

    return surfaces, molecules


# ---------------------------------------------------------------------------
# Helper: build task entries for surfaces x molecules x calculators
# Seeds are assigned based on carbon count:
#   C0-C2 -> 3 seeds    C3+ -> 6 seeds
# ---------------------------------------------------------------------------
def make_entries(surfaces, molecules, calculators, pop=60):
    entries = []
    for surf in surfaces:
        for mol, gen in molecules.items():
            n_c = carbon_count(mol)
            for calc in calculators:
                if n_c >= 3:
                    calc_seeds = SEEDS_5M_C3 if calc == "5m" else SEEDS_SEVENNET_C3
                else:
                    calc_seeds = SEEDS_5M    if calc == "5m" else SEEDS_SEVENNET
                entries.append(
                    (surf, mol, {"seeds": calc_seeds, "population_size": pop,
                                 "generations": gen, "calculator": calc,
                                 "n_carbon": n_c})
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
print(f"Discovered {len(ALL_MOLECULES)} molecules (with C# and seed count):")
for m in sorted(ALL_MOLECULES):
    n_c = carbon_count(m)
    n_seeds = 6 if n_c >= 3 else 3
    print(f"  {m:<30}  C{n_c}  ->  {n_seeds} seeds")
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
    n_carbon        = overrides.get("n_carbon",        0)

    for seed in seeds:
        rows.append({
            "task_id":         task_id,
            "surface":         surface,
            "adsorbate":       adsorbate,
            "seed":            seed,
            "calculator":      calculator,
            "population_size": population_size,
            "generations":     generations,
            "n_carbon":        n_carbon,
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
    n_c = carbon_count(mol)
    print(f"  {mol:<30}  C{n_c}  {count:>5} tasks")
print()
print("Carbon tier breakdown:")
c_tiers = Counter(f"C{r['n_carbon']}" for r in rows)
for tier, count in sorted(c_tiers.items()):
    print(f"  {tier}: {count} tasks")
print()
print("Calculator breakdown:")
calcs = Counter(r["calculator"] for r in rows)
for calc, count in sorted(calcs.items()):
    print(f"  {calc:<15}: {count} tasks")
print()
print("Submit with:")
print(f"  sbatch --array=0-{task_id-1}%20 goad_array_kestrel.slurm workflow/tasks_custom.csv")
