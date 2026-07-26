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

Seed counts by carbon number (three-tier scheme)
-------------------------------------------------
  C0-C2 : 2 seeds  [1, 2]        (2 seeds x 2 calcs =  4 runs per surface)
  C3-C4 : 3 seeds  [1, 2, 3]     (3 seeds x 2 calcs =  6 runs per surface)
  C5+   : 5 seeds  [1, 2, 3, 4, 5] (5 seeds x 2 calcs = 10 runs per surface)

Seed 0 excluded — consistently bad gen-1 for glycerol with this RNG seed.

Rationale for tiered seeds:
  - Larger conformational space at higher carbon count -> more seeds needed.
  - Three tiers give fine-grained control: reduce low-C overhead while
    keeping good statistical coverage for C5+ molecules.
  - Early stopping (patience=30, tol=0.001 eV) means each seed will
    finish well before the 200-generation cap if conditions are met.

Calculator ordering and selectivity
-------------------------------------
  - All sevennet_omni rows are generated BEFORE any 5m rows.
    This means when submitted as a Slurm array (ascending task_id),
    all SevenNet work runs first.
  - Use --fivem-min-carbon N to restrict 5m to molecules with
    carbon_count >= N (default 0 = 5m runs on everything).
    SevenNet always runs on all (non-capped) molecules.

Reversible carbon-number upper cap
------------------------------------
  - Use --max-carbon N to exclude molecules with carbon_count > N.
    Default 0 = no cap (all molecules included).
    This is purely a CSV-generation filter — no CIF files are moved
    or deleted, so the exclusion is fully reversible.
  - Example: --max-carbon 7 keeps C0-C7, defers C8+ until next run.

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
    python workflow/make_tasks_custom.py --max-carbon 7
    python workflow/make_tasks_custom.py --fivem-min-carbon 5
    python workflow/make_tasks_custom.py --seeds-c0-c2 2 --seeds-c3-c4 3 --seeds-c5plus 5
    sbatch --array=0-<N>%20 goad_array_kestrel.slurm workflow/tasks_custom.csv

CLI flags
---------
    --seeds-c0-c2 N         Seeds for C0-C2 molecules (default: 2)
    --seeds-c3-c4 N         Seeds for C3-C4 molecules (default: 3)
    --seeds-c5plus N        Seeds for C5+ molecules   (default: 5)
    --calculators A [B ...]  Calculators to run (default: sevennet_omni 5m)
    --fivem-min-carbon N    Only run 5m on molecules with carbon_count >= N (default: 0)
    --sevennet-min-carbon N Only run sevennet_omni on molecules with carbon_count >= N (default: 0)
    --max-carbon N          Exclude molecules with carbon_count > N (default: 0 = no cap)
    --out PATH              Output CSV path (default: workflow/tasks_custom.csv)
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Import carbon_count from a side-effect-free shared module.
# batch_isopropanol.py re-exports it for backward compatibility.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from molecule_utils import carbon_count

# ---------------------------------------------------------------------------
# Default seed counts (three-tier scheme)
# Seed 0 excluded — consistently bad gen-1 for glycerol with this RNG seed.
# ---------------------------------------------------------------------------
DEFAULT_SEEDS_C0C2   = 2   # C0-C2
DEFAULT_SEEDS_C3C4   = 3   # C3-C4
DEFAULT_SEEDS_C5PLUS = 5   # C5+

# ---------------------------------------------------------------------------
# DEFAULT GA settings
# ---------------------------------------------------------------------------
DEFAULT = {
    "population_size": 60,
    "generations":     200,
    "calculator":      "sevennet_omni",
}

# ---------------------------------------------------------------------------
# CALCULATORS (SevenNet first so Slurm array runs SevenNet before 5m)
# ---------------------------------------------------------------------------
DEFAULT_CALCS = ["sevennet_omni", "5m"]

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
    "1-butanol", "2-butanol", "pentanol", "ethylene_glycol",
    "sorbitol", "xylitol",
    # Carbonyls
    "glyoxal",
    # Aldehydes
    "acetaldehyde", "furfural", "5-HMF",
    "propanal", "butanal", "valeraldehyde", "hexanal",
    "benzaldehyde", "5-methylfurfural",
    # Phenols
    "2-ethylphenol", "hydroquinone",
    # Guaiacols
    "guaiacol", "4-methylguaiacol", "eugenol", "isoeugenol",
    # Syringols
    "syringol", "propyl_syringol", "syringaldehyde",
    # Sugars
    "levoglucosan", "alpha-D-glucopyranose", "D-fructofuranose",
    "D-xylopyranose", "1,6-anhydroglucofuranose",
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
    "diethyl_ether", "THF", "ethyl_acetate", "methyl_formate",
    "angelica_lactone", "gamma_butyrolactone",
    "furfuryl_alcohol", "gamma_valerolactone", "dimethyl_succinate",
    # Furan
    "2-furanone",
    # Oxygenates
    "hydroxyacetaldehyde", "acetal", "methylcyclopentenolone", "vanillin",
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
#
# Seeds are assigned based on a three-tier carbon scheme:
#   C0-C2 -> n_seeds_c0c2   seeds
#   C3-C4 -> n_seeds_c3c4   seeds
#   C5+   -> n_seeds_c5plus seeds
#
# Calculators are iterated in the order given so that all tasks for the
# first calculator appear before the second (SevenNet-first ordering).
#
# Selectivity:
#   fivem_min_carbon    -- skip 5m  for molecules with carbon_count < N
#   sevennet_min_carbon -- skip sevennet_omni for molecules with carbon_count < N
# ---------------------------------------------------------------------------
def make_entries(
    surfaces,
    molecules,
    calculators,
    pop=60,
    n_seeds_c0c2=DEFAULT_SEEDS_C0C2,
    n_seeds_c3c4=DEFAULT_SEEDS_C3C4,
    n_seeds_c5plus=DEFAULT_SEEDS_C5PLUS,
    fivem_min_carbon=0,
    sevennet_min_carbon=0,
):
    """Return a list of (surface, mol, overrides) task entries.

    Calculators are iterated at the OUTERMOST loop so all tasks for
    calculators[0] are appended before any tasks for calculators[1].
    This guarantees that when task_ids are assigned in order, sevennet_omni
    rows always precede 5m rows (assuming the default calculator order).
    """

    def _seeds_for_tier(n_c, n_c0c2, n_c3c4, n_c5plus):
        if n_c <= 2:
            return list(range(1, n_c0c2 + 1))
        elif n_c <= 4:
            return list(range(1, n_c3c4 + 1))
        else:
            return list(range(1, n_c5plus + 1))

    entries = []
    for calc in calculators:
        for surf in surfaces:
            for mol, gen in molecules.items():
                n_c = carbon_count(mol)

                # Apply per-calculator selectivity
                if calc == "5m" and n_c < fivem_min_carbon:
                    continue
                if calc == "sevennet_omni" and n_c < sevennet_min_carbon:
                    continue

                calc_seeds = _seeds_for_tier(n_c, n_seeds_c0c2, n_seeds_c3c4, n_seeds_c5plus)
                entries.append(
                    (surf, mol, {"seeds": calc_seeds, "population_size": pop,
                                 "generations": gen, "calculator": calc,
                                 "n_carbon": n_c})
                )
    return entries



def _tier_label(n_c):
    """Return a human-readable carbon tier label."""
    if n_c <= 2:
        return "C0-C2"
    elif n_c <= 4:
        return "C3-C4"
    else:
        return "C5+"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="make_tasks_custom.py",
        description="Generate workflow/tasks_custom.csv from inputs/*.cif.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python workflow/make_tasks_custom.py\n"
            "  python workflow/make_tasks_custom.py --max-carbon 7\n"
            "  python workflow/make_tasks_custom.py --fivem-min-carbon 5\n"
            "  python workflow/make_tasks_custom.py --seeds-c0-c2 2 --seeds-c3-c4 3 --seeds-c5plus 5\n"
        ),
    )
    parser.add_argument(
        "--seeds-c0-c2", type=int, default=DEFAULT_SEEDS_C0C2, metavar="N",
        help=f"Number of seeds for C0-C2 molecules (default: {DEFAULT_SEEDS_C0C2})",
    )
    parser.add_argument(
        "--seeds-c3-c4", type=int, default=DEFAULT_SEEDS_C3C4, metavar="N",
        help=f"Number of seeds for C3-C4 molecules (default: {DEFAULT_SEEDS_C3C4})",
    )
    parser.add_argument(
        "--seeds-c5plus", type=int, default=DEFAULT_SEEDS_C5PLUS, metavar="N",
        help=f"Number of seeds for C5+ molecules (default: {DEFAULT_SEEDS_C5PLUS})",
    )
    parser.add_argument(
        "--calculators", nargs="+", default=list(DEFAULT_CALCS), metavar="CALC",
        help=(
            "Calculators to use, in order (default: sevennet_omni 5m). "
            "All rows for the first calculator appear before the second."
        ),
    )
    parser.add_argument(
        "--fivem-min-carbon", type=int, default=0, metavar="N",
        help=(
            "Only generate 5m tasks for molecules with carbon_count >= N "
            "(default: 0 = 5m runs on all molecules)."
        ),
    )
    parser.add_argument(
        "--sevennet-min-carbon", type=int, default=0, metavar="N",
        help=(
            "Only generate sevennet_omni tasks for molecules with carbon_count >= N "
            "(default: 0 = sevennet_omni runs on all molecules)."
        ),
    )
    parser.add_argument(
        "--max-carbon", type=int, default=0, metavar="N",
        help=(
            "Exclude molecules with carbon_count > N from task generation. "
            "Default 0 means no cap (all molecules included). "
            "No CIF files are moved or deleted — fully reversible."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("workflow/tasks_custom.csv"), metavar="PATH",
        help="Output CSV path (default: workflow/tasks_custom.csv)",
    )

    args = parser.parse_args(argv)

    # -----------------------------------------------------------------------
    # Discover surfaces and molecules
    # -----------------------------------------------------------------------
    all_surfaces, all_molecules = discover_surfaces_and_molecules(INPUTS_DIR)

    if not all_surfaces:
        print("ERROR: No surface CIFs found. Run generate_surface_cifs.py first.")
        raise SystemExit(1)

    if not all_molecules:
        print("ERROR: No molecule CIFs found. Run generate_molecule_cifs.py first.")
        raise SystemExit(1)

    # -----------------------------------------------------------------------
    # Apply --max-carbon cap (molecules only; purely at generation time)
    # -----------------------------------------------------------------------
    if args.max_carbon > 0:
        capped = {m: g for m, g in all_molecules.items()
                  if carbon_count(m) > args.max_carbon}
        if capped:
            print(
                f"NOTE: --max-carbon {args.max_carbon} excludes "
                f"{len(capped)} molecule(s) (C{args.max_carbon+1}+):"
            )
            for m in sorted(capped):
                print(f"  {m:<30}  C{carbon_count(m)}  [deferred]")
            print()
        all_molecules = {m: g for m, g in all_molecules.items()
                         if carbon_count(m) <= args.max_carbon}

    # -----------------------------------------------------------------------
    # Print discovery summary
    # -----------------------------------------------------------------------
    print(f"Discovered {len(all_surfaces)} surfaces:")
    for s in all_surfaces:
        print(f"  {s}")
    print()

    print(f"Discovered {len(all_molecules)} molecules (with C# and seed count):")
    for m in sorted(all_molecules):
        n_c = carbon_count(m)
        if n_c <= 2:
            n_seeds = args.seeds_c0_c2
        elif n_c <= 4:
            n_seeds = args.seeds_c3_c4
        else:
            n_seeds = args.seeds_c5plus
        print(f"  {m:<30}  C{n_c}  ({_tier_label(n_c)})  ->  {n_seeds} seeds")
    print()

    # -----------------------------------------------------------------------
    # Build task entries (SevenNet-first: calculators iterated outermost)
    # -----------------------------------------------------------------------
    custom_tasks = make_entries(
        all_surfaces,
        all_molecules,
        args.calculators,
        pop=DEFAULT["population_size"],
        n_seeds_c0c2=args.seeds_c0_c2,
        n_seeds_c3c4=args.seeds_c3_c4,
        n_seeds_c5plus=args.seeds_c5plus,
        fivem_min_carbon=args.fivem_min_carbon,
        sevennet_min_carbon=args.sevennet_min_carbon,
    )

    # -----------------------------------------------------------------------
    # Generate CSV
    # -----------------------------------------------------------------------
    out = args.out
    out.parent.mkdir(exist_ok=True)

    rows = []
    task_id = 0

    for surface, adsorbate, overrides in custom_tasks:
        seeds           = overrides.get("seeds",           [])
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

    if not rows:
        print("ERROR: No tasks generated. Check inputs/ directory and flags.")
        raise SystemExit(1)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {task_id} tasks to {out}")
    print()
    print("Molecule breakdown:")
    mols = Counter(r["adsorbate"] for r in rows)
    for mol, count in sorted(mols.items()):
        n_c = carbon_count(mol)
        print(f"  {mol:<30}  C{n_c}  {count:>5} tasks")
    print()
    print("Carbon tier breakdown:")
    tiers = Counter(_tier_label(r["n_carbon"]) for r in rows)
    for tier, count in sorted(tiers.items()):
        print(f"  {tier}: {count} tasks")
    print()
    print("Calculator breakdown:")
    calcs_counter = Counter(r["calculator"] for r in rows)
    for calc, count in sorted(calcs_counter.items()):
        print(f"  {calc:<15}: {count} tasks")
    print()
    print("Submit with:")
    print(f"  sbatch --array=0-{task_id-1}%20 goad_array_kestrel.slurm {out}")


if __name__ == "__main__":
    main()
