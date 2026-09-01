#!/usr/bin/env python
"""
dft_contact_geometry.py
=======================
Compute the DFT relaxed adsorption geometry -- nearest metal<->adsorbate bond
distance (minimum-image PBC), the element pair, the adsorbate anchor atom and a
coarse coordination-based binding site -- straight from the DFT CONTCARs.

This is the DFT counterpart of ``mlip_contact_geometry.py``.  Crucially it needs
**only the DFT CONTCARs** (no MLIP .cif gallery), so it runs on the cluster with
nothing extra to upload.  It reuses:

  * the exact CONTCAR discovery / functional resolution of
    ``render_dft_structures.py`` (so it sees the same systems the DFT PNGs were
    rendered from), and
  * the exact contact metric of ``compare_dft_mlip_structures.py`` /
    ``mlip_contact_geometry.py`` (so the DFT and MLIP bond distances on the page
    are computed identically and are directly comparable).

The output CSV is consumed by ``build_dft_pages.py --struct-compare`` to fill the
"DFT contact" bond distance (and the MLIP-minus-DFT delta) on
``dft_comparison.html``.  RMSD / max-displacement are *not* produced here because
those need both structures aligned; use ``compare_dft_mlip_structures.py`` (which
also needs the gallery) if you want them.

Usage
-----
    python dft_contact_geometry.py \
        --dft-jobs poscar/best --dft-jobs poscar/best2 --dft-jobs dft_jobs \
        --targets analysis_out/dft_vs_mlip_pairs.csv \
        --out analysis_out/dft_geom.csv

Then add analysis_out/dft_geom.csv to the bundle and, on the machine that builds
the pages:

    python build_dft_pages.py --analysis-dir <analysis> --gallery <gallery> \
        --dft-png-dir dft_png --struct-compare <analysis>/dft_geom.csv \
        --out-dir <pages>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from ase.io import read as ase_read

from compare_dft_mlip_structures import (
    min_metal_adsorbate_contact,
    surface_metal,
)
from mlip_contact_geometry import coarse_site
from render_dft_structures import FUNC_PRIORITY, discover
from mol_canon import match_keys


def load_targets(path: Path):
    """Unique (surface, molecule) from a CSV with surface,molecule columns.

    The pairs CSV has one row per functional, so de-duplicate to one entry per
    system. The molecule string is kept verbatim (the page's spelling) so the
    output keys line up with the gallery / pairs naming.
    """
    seen, targets = set(), []
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (r.get("surface", "").strip(), r.get("molecule", "").strip())
            if all(key) and key not in seen:
                seen.add(key)
                targets.append(key)
    return targets


def build_index(dft_roots, molecule_first: bool):
    """(surface, molkey) -> {func_key: contcar}, exactly as render_dft_structures."""
    index: dict = {}
    for root in dft_roots:
        root = Path(root)
        if not root.is_dir():
            print(f"  WARNING: --dft-jobs not found: {root}", file=sys.stderr)
            continue
        orders = [True, False] if not molecule_first else [True]
        for mf in orders:
            for surface, molkey, fk, contcar in discover(root, mf):
                index.setdefault((surface, molkey), {}).setdefault(fk, contcar)
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dft-jobs", required=True, action="append", metavar="ROOT",
                    help="Root(s) of finished DFT relaxations (repeatable): "
                         "--dft-jobs poscar/best --dft-jobs poscar/best2 "
                         "--dft-jobs dft_jobs.")
    ap.add_argument("--targets", default="analysis_out/dft_vs_mlip_pairs.csv",
                    help="CSV with surface,molecule columns naming the systems "
                         "to score (default: the matched pairs).")
    ap.add_argument("--out", default="analysis_out/dft_geom.csv")
    ap.add_argument("--molecule-first", action="store_true",
                    help="DFT dirs are '<MOL>_<SURF>...' (default: try both orders).")
    args = ap.parse_args()

    index = build_index(args.dft_jobs, args.molecule_first)
    print(f"Indexed {len(index)} (surface, molecule) DFT keys "
          f"from {len(args.dft_jobs)} root(s).")

    targets = load_targets(Path(args.targets))
    print(f"Scoring {len(targets)} target systems -> {args.out}")

    rows, done, missing = [], 0, 0
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
            atoms = ase_read(str(contcar))
        except Exception as e:                       # noqa: BLE001
            missing += 1
            print(f"  ERROR {surf}_{mol}: {e}")
            continue

        metal = surface_metal(surf)
        d, pair, _, j = min_metal_adsorbate_contact(atoms, metal)
        site, ncoord, _ = coarse_site(atoms, metal, j)
        anchor = f"{atoms.get_chemical_symbols()[j]}{j}" if j >= 0 else ""
        rows.append({
            "surface": surf,
            "molecule": mol,
            "functional": func,
            "min_dist_dft": None if d is None else round(d, 3),
            "pair_dft": pair,
            "site_dft": site,
            "ncoord_dft": ncoord,
            "anchor_dft": anchor,
            "contcar": str(contcar),
        })
        done += 1
        if d is None:
            print(f"  OK    {surf}_{mol}  <- {func}  (no metal-adsorbate contact)")
        else:
            print(f"  OK    {surf}_{mol}  <- {func}  d={d:.3f} {pair} [{site}]")

    rows.sort(key=lambda r: (r["surface"], r["molecule"]))
    fields = ["surface", "molecule", "functional", "min_dist_dft", "pair_dft",
              "site_dft", "ncoord_dft", "anchor_dft", "contcar"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nScored {done} DFT systems ({missing} missing/failed) -> {args.out}")
    print("Next: add this CSV to the bundle and rebuild with "
          "build_dft_pages.py --struct-compare <...>/dft_geom.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
