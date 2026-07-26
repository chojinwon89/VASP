#!/usr/bin/env python3
"""
extract_structures.py
=====================
Extract the final adsorbed geometry (``final_adsorbed.cif``) from finished
GOAD runs and save both a ``.cif`` file and a ``.png`` render into
``results/structure/``.

Directory layout consumed (same as ``extract_poscar.py``):

    runs/C<n>/<surface>_<adsorbate>_seed<N>_<calc>/
        <adsorbate>_on_<surface>/
            final_adsorbed.cif      <- source geometry (preferred: .traj last frame)
            final_adsorbed.traj     <- ASE trajectory (preferred over .cif)
            result.json             <- E_ads metadata

Outputs written to ``results/structure/`` (or ``--out-dir``):

    results/structure/
        <surface>_<adsorbate>.cif          <- best-only mode, single calculator
        <surface>_<adsorbate>_<calc>.cif   <- best-only, multiple calculators detected
        <surface>_<adsorbate>_<calc>_seed<N>.cif  <- --all-seeds mode
        <surface>_<adsorbate>.png          <- matching render (Agg backend)

**Default = best-only**: for each (surface, adsorbate) pair, export only the
single run with the lowest ``E_ads_eV`` across all seeds and calculators.
Use ``--all-seeds`` to export every finished run instead.

Naming scheme
-------------
- best-only + ``--calculator <calc>`` given:
    ``<surface>_<adsorbate>.cif``
- best-only + no ``--calculator`` (multiple calculators detected):
    ``<surface>_<adsorbate>_<calc>.cif``
- ``--all-seeds``:
    ``<surface>_<adsorbate>_<calc>_seed<N>.cif``

Example usage
-------------
::

    python extract_structures.py                       # best-only, all calculators
    python extract_structures.py --calculator sevennet_omni
    python extract_structures.py --all-seeds           # every finished run
    python extract_structures.py --out-dir results/structure --runs-dir runs
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless HPC nodes

import argparse
import sys
from pathlib import Path

from ase import Atoms
from ase.io import write

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extract_poscar import collect_runs, select_best_per_system


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_stem(entry: dict, multi_calc: bool, all_seeds: bool) -> str:
    """
    Return the base filename stem (without extension) for a structure entry.

    Rules:
    - all-seeds:   <surface>_<adsorbate>_<calc>_seed<N>
    - best-only + single calc (or --calculator filter given):
                   <surface>_<adsorbate>
    - best-only + multiple calcs detected:
                   <surface>_<adsorbate>_<calc>
    """
    surface   = entry["surface"]
    adsorbate = entry["adsorbate"]
    calc      = entry["calculator"]
    seed      = entry["seed"]

    if all_seeds:
        return f"{surface}_{adsorbate}_{calc}_seed{seed}"
    if multi_calc:
        return f"{surface}_{adsorbate}_{calc}"
    return f"{surface}_{adsorbate}"


def _build_render_atoms(atoms):
    """Return a metadata-light copy for PNG rendering (avoids ASE tag/occ issues)."""
    return Atoms(
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
    )


def _write_png_with_fallback(png_path: Path, render_atoms):
    """
    Try styled PNG render first, then plain PNG render fallback.

    Returns (styled_exception, plain_exception):
      - (None, None) on styled success
      - (styled_exc, None) if styled fails but plain succeeds
      - (styled_exc, plain_exc) if both fail
    """
    try:
        write(str(png_path), render_atoms, rotation="-70x,20y,10z", show_unit_cell=2)
        return None, None
    except Exception as styled_exc:
        try:
            write(str(png_path), render_atoms)
            return styled_exc, None
        except Exception as plain_exc:
            return styled_exc, plain_exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract final GOAD adsorbed geometries and write CIF + PNG pairs "
            "into results/structure/ (default: best-only — lowest E_ads per system)."
        )
    )
    parser.add_argument(
        "--runs-dir", default="runs",
        help="Path to the GOAD runs/ directory (default: ./runs)",
    )
    parser.add_argument(
        "--out-dir", default="results/structure",
        help="Output directory for CIF/PNG files (default: results/structure)",
    )
    parser.add_argument(
        "--calculator", default=None, metavar="CALC",
        help=(
            "Only use runs from this calculator "
            "(e.g. sevennet_omni or 5m). Default: all calculators."
        ),
    )
    parser.add_argument(
        "--all-seeds", action="store_true",
        help=(
            "Export every finished run (one CIF+PNG per seed) instead of "
            "best-only. Default is best-only (lowest E_ads per system)."
        ),
    )
    parser.add_argument(
        "--include-unfinished", action="store_true",
        help=(
            "Include runs with final_adsorbed.cif/.traj even if status.json "
            "is missing or state != finished."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print one line per file written.",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)

    if not runs_dir.exists():
        print(f"ERROR: runs directory not found: {runs_dir}")
        raise SystemExit(1)

    # Collect finished runs
    entries, skipped_not_finished = collect_runs(
        runs_dir,
        calculator_filter=args.calculator,
        include_unfinished=args.include_unfinished,
    )

    if not entries:
        msg = "No completed runs"
        if args.calculator:
            msg += f" for calculator '{args.calculator}'"
        if not args.include_unfinished:
            msg += " (status.json state=finished required)"
        print(msg + f" with final_adsorbed.cif/.traj found in {runs_dir}")
        if skipped_not_finished and not args.include_unfinished:
            print(f"Skipped (not finished per status.json): {skipped_not_finished}")
        raise SystemExit(0)

    print(f"Found {len(entries)} completed run(s) in {runs_dir}"
          + (f" [calculator={args.calculator}]" if args.calculator else ""))
    if skipped_not_finished and not args.include_unfinished:
        print(f"Skipped (not finished per status.json): {skipped_not_finished}")

    # Select entries to export
    if args.all_seeds:
        to_export = entries
    else:
        best = select_best_per_system(entries)
        to_export = list(best.values())

    # Determine whether multiple calculators are present (affects filename)
    calcs_in_export = {e["calculator"] for e in to_export}
    multi_calc = (args.calculator is None) and (len(calcs_in_export) > 1)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_png_fail = 0
    n_fail = 0

    for entry in sorted(to_export, key=lambda e: (e["surface"], e["adsorbate"])):
        stem     = _build_stem(entry, multi_calc=multi_calc, all_seeds=args.all_seeds)
        cif_path = out_dir / f"{stem}.cif"
        png_path = out_dir / f"{stem}.png"
        atoms    = entry["atoms"]
        e_ads    = entry["E_ads_eV"]
        e_str    = f"{e_ads:.4f}" if e_ads is not None else "unknown"

        try:
            write(str(cif_path), atoms)
        except Exception as exc:
            print(f"[ERR-cif] {stem}: {type(exc).__name__}: {repr(exc)}")
            n_fail += 1
            continue

        render_atoms = _build_render_atoms(atoms)
        styled_exc, plain_exc = _write_png_with_fallback(png_path, render_atoms)

        if plain_exc is None:
            print(f"[OK] {stem} | E={e_str} eV")
            if args.verbose:
                print(f"     cif: {cif_path}")
                print(f"     png: {png_path}")
            n_ok += 1
        else:
            print(
                f"[OK-cif/ERR-png] {stem}: "
                f"{type(plain_exc).__name__}: {repr(plain_exc)} "
                f"(styled: {type(styled_exc).__name__}: {repr(styled_exc)})"
            )
            if args.verbose:
                print(f"     cif: {cif_path}")
            n_png_fail += 1

    print()
    print(f"Done.")
    print(f"  CIF written    : {n_ok + n_png_fail}")
    print(f"  PNG written    : {n_ok}")
    print(f"  PNG failed     : {n_png_fail}")
    print(f"  Failed         : {n_fail}")
    print(f"  Output    : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
