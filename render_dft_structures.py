#!/usr/bin/env python
"""
render_dft_structures.py
========================
Render the DFT final (relaxed) structures as PNGs that line up 1:1 with the
GOAD+SevenNet gallery images, so they can be dropped into the
bond-distance-review "structure & geometry" page (dft_comparison.html) as the
right-hand "DFT (relaxed)" panel.

Runs on the cluster (Perlmutter / Kestrel) where the CONTCARs live; needs ASE.

It matches the page's system list exactly:

  * Targets are read from a CSV of (surface, molecule) pairs -- by default
    analysis_out/mlip_geom.csv (the 44 comparison systems). The output PNG for
    each target is named ``<surface>_<molecule>_dft.png`` using the *target*
    molecule string, so it matches ``<surface>_<molecule>_mlip.png`` on the page
    regardless of how the DFT job directories are named.
  * DFT CONTCARs are discovered under --dft-jobs and matched to a target by
    (surface, canon_molecule) so 'C2H4_Cu100_top/PBE/CONTCAR',
    'Cu100_ethene/pbe/CONTCAR' and '.../pbe/fully_relaxed/CONTCAR' all resolve.
  * Rendering uses the same ASE view as export_best_goad_structures.py
    (rotation -70x,20y,10z, show_unit_cell=2) for visual comparability.

Then sync the output dir back and rebuild the page:

    python build_dft_pages.py --analysis-dir analysis_out \
        --gallery <gallery> --out-dir <pages-clone> \
        --struct-compare dft_mlip_structure_compare.csv \
        --dft-png-dir dft_png

Usage
-----
    python render_dft_structures.py --dft-jobs poscar/best --out-dir dft_png
    python render_dft_structures.py --dft-jobs poscar/best2 --out-dir dft_png  # add more
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ase.io import read, write

from compare_dft_mlip_structures import (
    FUNC_DIRS, normalise_func, parse_surface_molecule,
)
from mol_canon import canon_molecule, match_keys  # noqa: F401  (canon re-exported)

# Which functional to prefer when several relaxations exist for one system.
FUNC_PRIORITY = ["pbe", "pbe_d3", "r2scan", "beef_vdw"]


def find_func_and_system(contcar: Path):
    """Walk up from a CONTCAR to find (func_key, system_dir_name).

    Handles <system>/<FUNC>/CONTCAR and <system>/<FUNC>/<sub>/CONTCAR
    (e.g. a trailing 'fully_relaxed' or 'single-point' directory).
    """
    parts = contcar.parent.parts
    for i in range(len(parts) - 1, -1, -1):
        fk = normalise_func(parts[i])
        if fk in FUNC_DIRS:
            system = parts[i - 1] if i - 1 >= 0 else ""
            return fk, system
    return None, None


def discover(root: Path, molecule_first: bool):
    """Yield (surface, molkey, func_key, contcar) for every DFT CONTCAR.

    molkey is emitted once per candidate key (canon / raw-lower / alias) so an
    index built from it matches targets by any of those spellings.
    """
    for contcar in root.rglob("CONTCAR"):
        fk, system = find_func_and_system(contcar)
        if fk is None or not system:
            continue
        surface, molecule = parse_surface_molecule(system, molecule_first)
        if molecule == "unknown":
            continue
        for key in match_keys(molecule):
            yield surface, key, fk, contcar


def load_targets(path: Path):
    """Return list of (surface, molecule) using the exact page molecule string."""
    targets = []
    with path.open() as f:
        for r in csv.DictReader(f):
            surf = r.get("surface", "").strip()
            mol = r.get("molecule", "").strip()
            if surf and mol:
                targets.append((surf, mol))
    return targets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dft-jobs", required=True,
                    help="Root of finished DFT relaxations (poscar/best, poscar/best2, "
                         "dft_jobs, ...). May be given multiple times.", action="append")
    ap.add_argument("--targets", default="analysis_out/mlip_geom.csv",
                    help="CSV with surface,molecule columns naming the systems to render.")
    ap.add_argument("--out-dir", default="dft_png")
    ap.add_argument("--rotation", default="-70x,20y,10z",
                    help="ASE rotation string (match the gallery renders).")
    ap.add_argument("--molecule-first", action="store_true",
                    help="DFT dirs are '<MOL>_<SURF>...' (default: try both).")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Build index: (surface, canon) -> {func_key: contcar}
    index = {}
    for root in args.dft_jobs:
        root = Path(root)
        if not root.is_dir():
            print(f"  WARNING: --dft-jobs not found: {root}")
            continue
        # try both name orders; molecule_first often True for GOAD-style dirs
        orders = [True, False] if not args.molecule_first else [True]
        for mf in orders:
            for surface, molkey, fk, contcar in discover(root, mf):
                index.setdefault((surface, molkey), {}).setdefault(fk, contcar)

    print(f"Indexed {len(index)} (surface, molecule) DFT keys "
          f"from {len(args.dft_jobs)} root(s).")

    targets = load_targets(Path(args.targets))
    print(f"Rendering {len(targets)} target systems -> {out}/")

    done = missing = 0
    for surf, mol in targets:
        cands = None
        for key in match_keys(mol):
            cands = index.get((surf, key))
            if cands:
                break
        if not cands:
            missing += 1
            print(f"  MISS  {surf}_{mol} (no DFT CONTCAR)")
            continue
        func = next((f for f in FUNC_PRIORITY if f in cands), next(iter(cands)))
        contcar = cands[func]
        try:
            atoms = read(contcar)
            png = out / f"{surf}_{mol}_dft.png"
            write(png, atoms, rotation=args.rotation, show_unit_cell=2)
            done += 1
            print(f"  OK    {surf}_{mol}  <- {func}  {contcar}")
        except Exception as e:                       # noqa: BLE001
            missing += 1
            print(f"  ERROR {surf}_{mol}: {e}")

    print(f"\nRendered {done} PNGs, {missing} missing/failed -> {out}/")
    print("Next: sync this dir back and run build_dft_pages.py --dft-png-dir "
          f"{out} --struct-compare dft_mlip_structure_compare.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
