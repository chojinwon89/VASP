#!/usr/bin/env python
"""
mlip_contact_geometry.py
========================
Compute the MLIP (GOAD + SevenNet-OMNI) relaxed adsorption geometry for the
DFT-vs-MLIP benchmark systems, straight from the gallery .cif files.

For every (surface, molecule) it reports the nearest metal<->adsorbate contact
(minimum-image PBC), the element pair, the adsorbate anchor atom, and a coarse
coordination-based binding site (atop / bridge / hollow) derived from how many
surface metal atoms sit within a tolerance of that shortest contact.

This is the MLIP half of the structure comparison; the DFT half comes from
compare_dft_mlip_structures.py run on the cluster where the CONTCARs live.
Geometry helpers are imported from compare_dft_mlip_structures.py so both sides
use exactly the same contact metric.

Usage
-----
    python mlip_contact_geometry.py \
        --gallery "/path/to/GOAD+Sevennet_Structures" \
        --pairs analysis_out/dft_vs_mlip_pairs.csv \
        --out analysis_out/mlip_geom.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    from ase.io import read as ase_read
except ImportError:
    sys.exit("ASE is required (conda install -c conda-forge ase).")

from compare_dft_mlip_structures import (
    min_metal_adsorbate_contact,
    surface_metal,
)
from mol_canon import canon_molecule, match_keys  # noqa: F401


def coarse_site(atoms, metal, j_ads, tol=0.45):
    """Classify the binding site by counting surface metal neighbours of the
    adsorbate anchor atom within (d_min + tol). 1->atop, 2->bridge, >=3->hollow.
    Returns (site_label, n_contacts, d_min)."""
    syms = atoms.get_chemical_symbols()
    metal_idx = [i for i, s in enumerate(syms) if s == metal]
    if not metal_idx or j_ads < 0:
        return "n/a", 0, None
    d = atoms.get_distances(j_ads, metal_idx, mic=True)
    d_min = float(np.min(d))
    n = int(np.sum(d <= d_min + tol))
    label = {1: "atop", 2: "bridge"}.get(n, "hollow" if n >= 3 else "n/a")
    return label, n, d_min


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", required=True,
                    help="Directory with the plain-named gallery .cif files "
                         "(these are the SevenNet-OMNI relaxed structures).")
    ap.add_argument("--pairs", default="analysis_out/dft_vs_mlip_pairs.csv",
                    help="dft_vs_mlip_pairs.csv: defines which systems to score.")
    ap.add_argument("--out", default="analysis_out/mlip_geom.csv")
    args = ap.parse_args()

    gallery = Path(args.gallery)
    if not gallery.is_dir():
        sys.exit(f"--gallery not found: {gallery}")

    # unique (surface, molecule) from the comparison pairs
    systems = []
    seen = set()
    for r in csv.DictReader(open(args.pairs)):
        key = (r["surface"], r["molecule"])
        if key not in seen:
            seen.add(key)
            systems.append(key)

    rows = []
    n_missing = 0
    for surface, molecule in sorted(systems):
        # Look the gallery .cif up by any spelling of the molecule (formula or
        # common name) so a formula-named pairs row still finds Ag100_ethane.cif.
        cif = None
        for key in (molecule, *match_keys(molecule)):
            cand = gallery / f"{surface}_{key}.cif"
            if cand.exists():
                cif = cand
                break
        if cif is None:
            n_missing += 1
            print(f"  ! missing cif: {surface}_{molecule}.cif", file=sys.stderr)
            continue
        atoms = ase_read(str(cif))
        metal = surface_metal(surface)
        d, pair, _, j = min_metal_adsorbate_contact(atoms, metal)
        site, ncoord, _ = coarse_site(atoms, metal, j)
        anchor = f"{atoms.get_chemical_symbols()[j]}{j}" if j >= 0 else ""
        rows.append({
            "surface": surface,
            "molecule": molecule,
            "mlip_min_dist": None if d is None else round(d, 3),
            "mlip_pair": pair,
            "mlip_anchor": anchor,
            "mlip_site": site,
            "mlip_ncoord": ncoord,
            "cif": cif.name,
        })

    rows.sort(key=lambda r: (r["surface"], r["molecule"]))
    fields = ["surface", "molecule", "mlip_min_dist", "mlip_pair",
              "mlip_anchor", "mlip_site", "mlip_ncoord", "cif"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Scored {len(rows)} MLIP systems ({n_missing} missing cif) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
