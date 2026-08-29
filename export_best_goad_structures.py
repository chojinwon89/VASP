#!/usr/bin/env python3
"""Export the best (lowest-energy) GOAD+SevenNet structure per system.

Walks a GOAD results tree, and for every ``<batch>/<mol>_on_<metal><facet>/``
system directory it takes the ``final_adsorbed`` structure (the GOAD-converged
global minimum). When the same system appears in several batches, the one with
the lowest single-point adsorption energy (from ``ga_history.txt``) is kept.

Outputs, into ``--out`` (default ``export_best``):
  * ``<Metal><Facet>_<molecule>_sevennet_omni.cif``  -- gallery-named structure
  * ``<Metal><Facet>_<molecule>_sevennet_omni.png``  -- rendered image
  * ``best_goad_index.csv``                           -- what was picked + energy

Download the ``.cif`` / ``.png`` files into your local
``GOAD+Sevennet_Structures`` folder.

Usage on Perlmutter:
    python export_best_goad_structures.py --results results --out export_best
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ase.io import read, write

# mol_on_Cu100  /  mol_on_Cu111  etc.  -> (molecule, Metal, facet)
SYS_RE = re.compile(r"^(?P<mol>.+?)_on_(?P<metal>[A-Z][a-z]?)(?P<facet>\d{3})$")


def gallery_name(molecule: str, metal: str, facet: str) -> str:
    """Return the gallery filename stem, e.g. ``Cu100_isopropanol_sevennet_omni``."""
    return f"{metal}{facet}_{molecule}_sevennet_omni"


def best_energy(system_dir: Path) -> float | None:
    """Lowest adsorption energy for a system from ga_history.txt, if available."""
    hist = system_dir / "ga_history.txt"
    if not hist.exists():
        return None
    vals = []
    for line in hist.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            vals.append(float(line))
        except ValueError:
            continue
    return min(vals) if vals else None


def find_structure(system_dir: Path) -> Path | None:
    """Return a readable final structure path for the system, or None."""
    for cand in ("final_adsorbed.cif", "final_adsorbed.traj", "final_adsorbed"):
        p = system_dir / cand
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results",
                    help="GOAD results root (holds batch_* dirs).")
    ap.add_argument("--out", default="export_best",
                    help="Output directory for cif/png files.")
    ap.add_argument("--rotation", default="-70x,20y,10z",
                    help="ASE rotation string for the png render.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render even if outputs already exist.")
    args = ap.parse_args()

    results = Path(args.results)
    if not results.is_dir():
        raise SystemExit(f"results dir not found: {results.resolve()}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # system_key -> (energy, structure_path, molecule, metal, facet)
    best: dict[str, tuple[float, Path, str, str, str]] = {}

    for system_dir in sorted(results.glob("*/*")):
        if not system_dir.is_dir():
            continue
        m = SYS_RE.match(system_dir.name)
        if not m:
            continue
        struct = find_structure(system_dir)
        if struct is None:
            continue
        mol, metal, facet = m["mol"], m["metal"], m["facet"]
        key = f"{metal}{facet}_{mol}"
        e = best_energy(system_dir)
        e_cmp = e if e is not None else float("inf")
        if key not in best or e_cmp < best[key][0]:
            best[key] = (e_cmp, struct, mol, metal, facet)

    if not best:
        raise SystemExit(f"No GOAD systems matched '<mol>_on_<Metal><facet>' under {results}.")

    index_rows = ["system,molecule,metal,facet,energy_eV,source,cif,png"]
    n_ok = n_fail = 0
    for key in sorted(best):
        energy, struct, mol, metal, facet = best[key]
        stem = gallery_name(mol, metal, facet)
        cif_path = out / f"{stem}.cif"
        png_path = out / f"{stem}.png"

        if cif_path.exists() and png_path.exists() and not args.overwrite:
            print(f"[skip] {stem} (already exported)")
            e_str = "" if energy == float("inf") else f"{energy:.6f}"
            index_rows.append(f"{key},{mol},{metal},{facet},{e_str},{struct},{cif_path.name},{png_path.name}")
            n_ok += 1
            continue

        try:
            atoms = read(struct)
            write(cif_path, atoms)
            write(png_path, atoms, rotation=args.rotation, show_unit_cell=2)
            e_str = "" if energy == float("inf") else f"{energy:.6f}"
            print(f"[ok]   {stem} | E={e_str or 'n/a'} eV | {struct}")
            index_rows.append(f"{key},{mol},{metal},{facet},{e_str},{struct},{cif_path.name},{png_path.name}")
            n_ok += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[err]  {stem}: {exc}")
            n_fail += 1

    index_path = out / "best_goad_index.csv"
    index_path.write_text("\n".join(index_rows) + "\n")

    print("\nDone.")
    print(f"  exported : {n_ok}")
    print(f"  failed   : {n_fail}")
    print(f"  output   : {out.resolve()}")
    print(f"  index    : {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
