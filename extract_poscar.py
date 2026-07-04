#!/usr/bin/env python
"""
extract_poscar.py
=================
Convert the final relaxed geometries from GOAD runs into VASP POSCAR files
ready for DFT single-point or full relaxation verification.

Directory layout written by batch_isopropanol.py:

    runs/<surface>_<adsorbate>_seed<N>_<calc>/
        <adsorbate>_on_<surface>/
            final_adsorbed.cif      <- source geometry (lowest-energy adsorbed config)
            final_adsorbed.traj     <- ASE trajectory (alternative source)
            result.json             <- E_ads metadata

Output (default, without --best-only):

    poscar/
        Cu111_isopropanol_seed0/POSCAR
        Cu110_isopropanol_seed0/POSCAR
        ...
        best/
            Cu111_isopropanol/POSCAR    <- lowest E_ads across all seeds per surface
            Cu110_isopropanol/POSCAR
            ...

Output with --best-only (best POSCARs written directly into --out-dir):

    poscar/best2/
        Cu111_isopropanol/POSCAR
        Cu110_isopropanol/POSCAR
        ...

Usage
-----
    # Default: reads ./runs, writes per-seed + best/ into ./poscar
    python extract_poscar.py

    # Only use SevenNet-OMNI runs (skip 5m / other calculators)
    python extract_poscar.py --calculator sevennet_omni

    # Only use 5m runs
    python extract_poscar.py --calculator 5m

    # Custom runs dir + calculator filter
    python extract_poscar.py \\
        --runs-dir /scratch/jcho5/.../runs \\
        --calculator sevennet_omni \\
        --best-only

    # Best POSCARs only, directly into a named directory (no extra best/ level)
    python extract_poscar.py \\
        --runs-dir /scratch/jcho5/.../runs \\
        --out-dir  /scratch/jcho5/.../poscar/best2 \\
        --best-only

    # All seeds + best, everything under a custom root
    python extract_poscar.py \\
        --runs-dir /scratch/jcho5/.../runs \\
        --out-dir  /scratch/jcho5/.../poscar

    python extract_poscar.py --verbose
"""

import argparse
import json
import io
from pathlib import Path

from ase.io import read, write
from ase import Atoms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_final_geometry(run_dir: Path):
    """
    Given a task run directory, find and load the final_adsorbed geometry.
    Returns (atoms, metadata_dict).  atoms is None if not found.
    """
    cif_candidates  = sorted(run_dir.glob("*/final_adsorbed.cif"))
    traj_candidates = sorted(run_dir.glob("*/final_adsorbed.traj"))

    atoms = None

    # Prefer .traj (last BFGS frame); fall back to .cif
    if traj_candidates:
        try:
            atoms = read(str(traj_candidates[0]), index=-1)
        except Exception:
            atoms = None

    if atoms is None and cif_candidates:
        try:
            atoms = read(str(cif_candidates[0]))
        except Exception:
            atoms = None

    # Load result.json for energy metadata
    result_file = run_dir / "result.json"
    metadata = {}
    if result_file.exists():
        try:
            metadata = json.loads(result_file.read_text())
        except Exception:
            pass

    return atoms, metadata


def sort_atoms_by_species(atoms: Atoms) -> Atoms:
    """
    Re-order atoms so that all atoms of the same element are contiguous.
    Required VASP convention: POTCAR order must match POSCAR species order.
    Cu (surface) is encountered first so it stays first.
    """
    symbols = atoms.get_chemical_symbols()
    seen = []
    for s in symbols:
        if s not in seen:
            seen.append(s)

    sorted_indices = []
    for element in seen:
        sorted_indices.extend(i for i, s in enumerate(symbols) if s == element)

    return atoms[sorted_indices]


def write_poscar(atoms: Atoms, out_path: Path,
                 sort_species: bool = True, comment: str = "") -> None:
    """
    Write atoms to a VASP5 POSCAR file.

    The comment is written as the first line of the POSCAR manually,
    because ASE's write_vasp() does not accept a 'label' keyword in
    older ASE versions (the version on this cluster).
    """
    if sort_species:
        atoms = sort_atoms_by_species(atoms)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    write(buf, atoms, format="vasp", vasp5=True)
    poscar_text = buf.getvalue()

    # Replace the first line (ASE's default system name) with our comment
    lines = poscar_text.splitlines(keepends=True)
    if lines:
        lines[0] = (comment or out_path.parent.name) + "\n"
    poscar_text = "".join(lines)

    out_path.write_text(poscar_text)


# ---------------------------------------------------------------------------
# Run collection
# ---------------------------------------------------------------------------

def parse_calculator_from_dirname(name: str) -> str:
    """
    Extract the calculator name from a run directory name.
    Pattern: <surface>_<adsorbate>_seed<N>_<calculator>
    e.g. Cu111_isopropanol_seed0_sevennet_omni -> 'sevennet_omni'
         Pt111_ethanol_seed1_5m               -> '5m'
    """
    parts = name.split("_seed")
    if len(parts) != 2:
        return ""
    seed_calc = parts[1]               # e.g. '0_sevennet_omni'
    _, *calc_parts = seed_calc.split("_")
    return "_".join(calc_parts)        # e.g. 'sevennet_omni'


def collect_runs(runs_dir: Path, calculator_filter: str = None) -> list:
    """
    Walk runs_dir and collect all task entries that have a final_adsorbed
    geometry.

    Parameters
    ----------
    calculator_filter : str or None
        If given (e.g. 'sevennet_omni' or '5m'), only runs whose directory
        name ends with that calculator string are included.
    """
    entries  = []
    skipped  = 0

    for task_dir in sorted(runs_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        name = task_dir.name

        # ---- Calculator filter ----
        calc = parse_calculator_from_dirname(name)
        if calculator_filter and calc != calculator_filter:
            skipped += 1
            continue

        parts = name.split("_seed")
        if len(parts) != 2:
            continue

        surface_adsorbate = parts[0]       # e.g. Cu111_isopropanol
        seed_str, *_      = parts[1].split("_")

        atoms, meta = load_final_geometry(task_dir)
        if atoms is None:
            continue   # no final geometry — task may still be running

        # Parse surface/adsorbate from result.json first (most reliable)
        surface   = meta.get("surface_cif",  "").replace("inputs/", "").replace(".cif", "")
        adsorbate = meta.get("molecule_cif", "").replace("inputs/", "").replace(".cif", "")

        # Fall back to directory name parsing
        if not surface or not adsorbate:
            tokens    = surface_adsorbate.split("_")
            surface   = tokens[0]
            adsorbate = "_".join(tokens[1:])

        entries.append({
            "run_dir":       task_dir,
            "surface":       surface,
            "adsorbate":     adsorbate,
            "seed":          int(seed_str) if seed_str.isdigit() else seed_str,
            "calculator":    calc,
            "atoms":         atoms,
            "E_ads_eV":      meta.get("E_ads_eV",      None),
            "E_total_eV":    meta.get("E_total_eV",    None),
            "E_surface_eV":  meta.get("E_surface_eV",  None),
            "E_molecule_eV": meta.get("E_molecule_eV", None),
            "timestamp":     meta.get("timestamp",     ""),
        })

    if calculator_filter:
        print("Calculator filter : '{}' — skipped {} non-matching dirs, "
              "kept {} runs".format(calculator_filter, skipped, len(entries)))

    return entries


def select_best_per_system(entries: list) -> dict:
    """
    For each unique (surface, adsorbate) pair, return the entry with the
    lowest E_ads_eV across all seeds.
    """
    best = {}
    for e in entries:
        key = "{}_{}".format(e["surface"], e["adsorbate"])
        if key not in best:
            best[key] = e
        else:
            current_e = best[key]["E_ads_eV"]
            new_e     = e["E_ads_eV"]
            if new_e is not None and (current_e is None or new_e < current_e):
                best[key] = e
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract final GOAD geometries as VASP POSCAR files."
    )
    parser.add_argument(
        "--runs-dir", default="runs",
        help="Path to the GOAD runs/ directory (default: ./runs)",
    )
    parser.add_argument(
        "--out-dir", default="poscar",
        help=(
            "Output root directory for POSCAR files (default: ./poscar). "
            "Without --best-only, per-seed POSCARs go into OUT_DIR/<label>/POSCAR "
            "and best POSCARs go into OUT_DIR/best/<label>/POSCAR. "
            "With --best-only, best POSCARs are written directly into "
            "OUT_DIR/<label>/POSCAR with NO extra best/ subdirectory."
        ),
    )
    parser.add_argument(
        "--calculator", default=None,
        metavar="CALC",
        help=(
            "Only extract runs from this calculator. "
            "Examples: sevennet_omni  5m  chgnet  mace "
            "(default: use all calculators found in runs/)"
        ),
    )
    parser.add_argument(
        "--best-only", action="store_true",
        help=(
            "Only write the best-seed POSCAR per (surface, adsorbate). "
            "POSCARs are placed directly in --out-dir/<system>/POSCAR "
            "(no extra best/ subdirectory)."
        ),
    )
    parser.add_argument(
        "--no-sort", action="store_true",
        help="Do NOT sort atoms by species (default: sort, VASP convention)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print one line per POSCAR written",
    )
    args = parser.parse_args()

    runs_dir     = Path(args.runs_dir)
    out_dir      = Path(args.out_dir)
    sort_species = not args.no_sort

    if not runs_dir.exists():
        print("ERROR: runs directory not found: {}".format(runs_dir))
        raise SystemExit(1)

    entries = collect_runs(runs_dir, calculator_filter=args.calculator)

    if not entries:
        msg = "No completed runs"
        if args.calculator:
            msg += " for calculator '{}'".format(args.calculator)
        print(msg + " with final_adsorbed.cif/.traj found in {}".format(runs_dir))
        raise SystemExit(0)

    print("Found {} completed run(s) in {}{}".format(
        len(entries), runs_dir,
        " [calculator={}]".format(args.calculator) if args.calculator else ""
    ))

    # Show breakdown of what was found
    from collections import Counter
    by_calc = Counter(e["calculator"] for e in entries)
    for calc, n in sorted(by_calc.items()):
        print("  {:<20}: {} runs".format(calc, n))
    print()

    written = 0

    # ---- Per-seed POSCARs --------------------------------------------------
    if not args.best_only:
        for e in entries:
            label   = "{}_{}_{}_seed{}".format(
                e["surface"], e["adsorbate"], e["calculator"], e["seed"])
            poscar  = out_dir / label / "POSCAR"
            e_str   = "{:.4f} eV".format(e["E_ads_eV"]) \
                      if e["E_ads_eV"] is not None else "E_ads unknown"
            comment = ("{} + {} | seed={} | calc={} | E_ads={}".format(
                e["surface"], e["adsorbate"], e["seed"],
                e["calculator"], e_str))
            write_poscar(e["atoms"], poscar,
                         sort_species=sort_species, comment=comment)
            if args.verbose:
                print("  [per-seed]  {}   E_ads={}".format(poscar, e_str))
            written += 1

    # ---- Best-seed POSCARs -------------------------------------------------
    best = select_best_per_system(entries)

    best_dir = out_dir if args.best_only else out_dir / "best"

    for key, e in sorted(best.items()):
        label   = "{}_{}".format(e["surface"], e["adsorbate"])
        poscar  = best_dir / label / "POSCAR"
        e_str   = "{:.4f} eV".format(e["E_ads_eV"]) \
                  if e["E_ads_eV"] is not None else "E_ads unknown"
        comment = ("{} + {} | BEST seed={} | calc={} | E_ads={}".format(
            e["surface"], e["adsorbate"], e["seed"],
            e["calculator"], e_str))
        write_poscar(e["atoms"], poscar,
                     sort_species=sort_species, comment=comment)
        print("  [best]  {}   seed={}  E_ads={}".format(
              poscar, e["seed"], e_str))
        written += 1

    # ---- Summary -----------------------------------------------------------
    print("\nWrote {} POSCAR file(s) under {}/".format(written, out_dir))
    print()
    if args.best_only:
        print("Best-seed layout:  {}/<surface>_<adsorbate>/POSCAR".format(out_dir))
    else:
        print("Per-seed layout:   {}/<surface>_<adsorbate>_<calc>_seed<N>/POSCAR".format(
              out_dir))
        print("Best-seed layout:  {}/best/<surface>_<adsorbate>/POSCAR  <- use for DFT".format(
              out_dir))
    print()
    print("Next steps:")
    print("  1. python setup_vasp_jobs.py --poscar-dir {} --functional r2scan".format(
          best_dir))
    print("  2. for d in {}/*/*/; do (cd \"$d\" && sbatch slm.vasp.kestrel); done".format(
          best_dir))
    print("  3. python calc_binding_energy.py --best-dir {} --output dft_eads.csv".format(
          best_dir))


if __name__ == "__main__":
    main()
