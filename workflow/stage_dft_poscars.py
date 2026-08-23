#!/usr/bin/env python
"""
stage_dft_poscars.py
====================
Bridge the MLIP-relaxed gallery structures to a VASP job tree that
setup_vasp_jobs.py can populate with INCAR/KPOINTS/POTCAR per functional.

For every benchmark system in DFT_results/MANIFEST.csv that already has an
MLIP-relaxed structure (the `gallery_cif` column, found in the structure/
gallery), this writes a species-sorted VASP5 POSCAR to:

    <out-dir>/<system_id>/POSCAR      e.g. dft_jobs/CO_Ag111/POSCAR

which is exactly the layout setup_vasp_jobs.py discovers (a bucket of
<system>/POSCAR dirs).  Systems whose `gallery_cif` is empty are skipped and
listed (they need the SevenNet run + collect first).

By default it stages the "well-known" scope (everything except the `deoxy`
group): 25 molecules x 7 non-magnetic metals x 3 facets.

Full geometry optimization note
-------------------------------
Unlike a single-point, a full relax will move every atom unless the lower slab
layers are fixed.  Use --fix-bottom-layers N to write Selective-dynamics POSCARs
with the bottom N metal layers frozen (F F F), matching the usual slab-DFT and
the GOAD MLIP protocol.  Default 0 (no constraint) reproduces the prior
single-point POSCARs exactly.

Usage
-----
    python workflow/stage_dft_poscars.py \
        --manifest DFT_results/MANIFEST.csv \
        --structure-dir structure \
        --out-dir dft_jobs \
        --fix-bottom-layers 2

Then, per functional (add --cluster perlmutter once that lands):
    for f in pbe pbe-d3 r2scan beef-vdw; do
        python setup_vasp_jobs.py --poscar-dir dft_jobs --functional $f
    done
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Reuse the repo's tested POSCAR writer / species sorter.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_poscar import write_poscar  # noqa: E402

from ase.io import read  # noqa: E402
from ase.constraints import FixAtoms  # noqa: E402


def cluster_layers(z_values, tol):
    """Group sorted z-values into layers; return list of (zmin,zmax) per layer."""
    zs = sorted(z_values)
    layers = []
    start = prev = zs[0]
    for z in zs[1:]:
        if z - prev > tol:
            layers.append((start, prev))
            start = z
        prev = z
    layers.append((start, prev))
    return layers


def fix_bottom_layers(atoms, metal: str, n_layers: int, tol: float) -> int:
    """Freeze atoms of `metal` in the bottom n_layers. Returns #atoms fixed."""
    idx_metal = [i for i, s in enumerate(atoms.get_chemical_symbols())
                 if s == metal]
    if not idx_metal or n_layers <= 0:
        return 0
    z = atoms.get_positions()[:, 2]
    layers = cluster_layers([z[i] for i in idx_metal], tol)
    if n_layers >= len(layers):
        cutoff = max(zmax for _, zmax in layers)  # fix all metal
    else:
        cutoff = layers[n_layers - 1][1] + tol * 0.5
    fixed = [i for i in idx_metal if z[i] <= cutoff]
    if fixed:
        atoms.set_constraint(FixAtoms(indices=fixed))
    return len(fixed)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="Path to MANIFEST.csv")
    ap.add_argument("--structure-dir", required=True,
                    help="Gallery dir holding the gallery_cif files")
    ap.add_argument("--out-dir", default="dft_jobs",
                    help="Output POSCAR tree root (default: dft_jobs)")
    ap.add_argument("--exclude-groups", default="deoxy",
                    help="Comma-separated MANIFEST groups to exclude "
                         "(default: deoxy -> stages the 25 well-known molecules)")
    ap.add_argument("--metals", default="",
                    help="Optional comma-separated metal filter (e.g. Ag,Pt)")
    ap.add_argument("--facets", default="",
                    help="Optional comma-separated facet filter (e.g. 111)")
    ap.add_argument("--molecules", default="",
                    help="Optional comma-separated molecule (formula) filter")
    ap.add_argument("--fix-bottom-layers", type=int, default=0, metavar="N",
                    help="Freeze bottom N metal layers via Selective dynamics "
                         "(default 0 = no constraint).")
    ap.add_argument("--layer-tol", type=float, default=0.7,
                    help="z-gap (Angstrom) separating slab layers (default 0.7)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    struct_dir = Path(args.structure_dir)
    out_dir = Path(args.out_dir)
    if not manifest.exists():
        ap.error(f"manifest not found: {manifest}")
    if not struct_dir.is_dir():
        ap.error(f"structure dir not found: {struct_dir}")

    exclude = {g.strip() for g in args.exclude_groups.split(",") if g.strip()}
    fmetals = {m.strip() for m in args.metals.split(",") if m.strip()}
    ffacets = {f.strip() for f in args.facets.split(",") if f.strip()}
    fmols = {m.strip() for m in args.molecules.split(",") if m.strip()}

    rows = list(csv.DictReader(manifest.open()))
    staged, skipped_no_gallery, missing_file = [], [], []
    staged_rows = []

    for r in rows:
        if r["group"] in exclude:
            continue
        if fmetals and r["metal"] not in fmetals:
            continue
        if ffacets and r["facet"] not in ffacets:
            continue
        if fmols and r["molecule"] not in fmols:
            continue
        gal = r["gallery_cif"].strip()
        sysid = r["system_id"]
        if not gal:
            skipped_no_gallery.append(sysid)
            continue
        cif = struct_dir / gal
        if not cif.exists():
            missing_file.append((sysid, gal))
            continue
        try:
            atoms = read(str(cif))
        except Exception as e:  # noqa: BLE001
            missing_file.append((sysid, f"{gal} (read error: {e})"))
            continue
        n_fixed = 0
        if args.fix_bottom_layers > 0:
            n_fixed = fix_bottom_layers(atoms, r["metal"],
                                        args.fix_bottom_layers, args.layer_tol)
        poscar = out_dir / sysid / "POSCAR"
        if not args.dry_run:
            write_poscar(atoms, poscar, sort_species=True, comment=sysid)
        staged.append(sysid)
        staged_rows.append({
            "system_id": sysid, "molecule": r["molecule"], "name": r["name"],
            "metal": r["metal"], "facet": r["facet"], "group": r["group"],
            "natoms": len(atoms), "n_fixed": n_fixed,
            "gallery_cif": gal, "poscar": str(poscar),
        })

    if not args.dry_run and staged_rows:
        idx = out_dir / "staged_systems.csv"
        idx.parent.mkdir(parents=True, exist_ok=True)
        with idx.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(staged_rows[0].keys()))
            w.writeheader()
            w.writerows(staged_rows)

    # ---- summary ----
    from collections import Counter
    print("=" * 66)
    print("stage_dft_poscars")
    print("=" * 66)
    print(f"Manifest        : {manifest}")
    print(f"Structure dir   : {struct_dir}")
    print(f"Output tree     : {out_dir}/<system_id>/POSCAR")
    print(f"Fix bottom lyrs : {args.fix_bottom_layers}"
          + (f" (tol {args.layer_tol} A)" if args.fix_bottom_layers else ""))
    print()
    print(f"STAGED          : {len(staged)} systems")
    if staged_rows:
        permetal = Counter(s["metal"] for s in staged_rows)
        print("  per metal     : " + ", ".join(f"{m}={permetal[m]}"
              for m in sorted(permetal)))
        if args.fix_bottom_layers:
            avgfix = sum(s["n_fixed"] for s in staged_rows) / len(staged_rows)
            print(f"  avg atoms fixed/system: {avgfix:.1f}")
    print(f"SKIPPED (no MLIP structure yet): {len(skipped_no_gallery)}")
    if skipped_no_gallery:
        sample = ", ".join(skipped_no_gallery[:10])
        more = f" ... (+{len(skipped_no_gallery)-10} more)" \
            if len(skipped_no_gallery) > 10 else ""
        print(f"  {sample}{more}")
        print("  -> run the SevenNet gap-fill + collect, then re-stage.")
    if missing_file:
        print(f"gallery_cif NAMED but FILE MISSING: {len(missing_file)}")
        for sid, g in missing_file[:10]:
            print(f"  {sid}: {g}")
    print()
    if not args.dry_run and staged_rows:
        print(f"Wrote {len(staged_rows)} POSCARs + {out_dir}/staged_systems.csv")
    print("Next: for f in pbe pbe-d3 r2scan beef-vdw; do "
          f"python setup_vasp_jobs.py --poscar-dir {out_dir} --functional $f; done")


if __name__ == "__main__":
    main()
