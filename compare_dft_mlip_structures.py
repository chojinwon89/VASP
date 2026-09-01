#!/usr/bin/env python
"""
compare_dft_mlip_structures.py
==============================
Compare *relaxed* DFT final geometries against the MLIP (GOAD + SevenNet/
MatterSim) relaxed geometries for the same surface+molecule systems.

For every matched (surface, molecule, functional) it reports:

  * ``min_dist_dft`` / ``min_dist_mlip`` - nearest metal<->adsorbate contact
    (minimum-image PBC), the same metric used by the gallery bond-distance
    review, plus the element pair for each.
  * ``d_min_dist``  = min_dist_dft - min_dist_mlip  (positive = DFT further).
  * ``rmsd``        - per-atom RMSD between the two geometries after removing
    the centre-of-mass shift and matching atoms by index (both structures come
    from the same starting POSCAR, so atom ordering is preserved).
  * ``max_disp``    - largest single-atom displacement (Angstrom).
  * ``max_disp_atom`` - element+index of that atom.

Inputs
------
--dft-jobs   Root of the finished DFT relaxations.  Expected layout
             (Perlmutter):  <root>/<surface>_<molecule>/<FUNC>/CONTCAR
             also accepts   <root>/<system>/<FUNC>/CONTCAR
             and bucketed   <root>/C<n>/<system>/<FUNC>/CONTCAR
             FUNC in {PBE, PBE_D3, r2scan, beef_vdw} (case-insensitive).
--mlip-dir   Directory holding the MLIP relaxed structures as
             <surface>_<molecule>[_<tag>].cif  (the gallery .cif files).
--out        Output CSV path (default dft_mlip_structure_compare.csv).

Requires ASE.  On Perlmutter the base ``python`` already has ASE.

Usage
-----
    python compare_dft_mlip_structures.py \
        --dft-jobs dft_jobs \
        --mlip-dir /path/to/bond-distance-review \
        --out dft_mlip_structure_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

try:
    from ase.io import read as ase_read
    from ase import Atoms
except ImportError:
    sys.exit("ASE is required. On Perlmutter use the base `python` (has ASE).")


# --- metals we treat as the "surface" side of a contact -------------------
METALS = {
    "Cu", "Pt", "Pd", "Ni", "Ag", "Au", "Co", "Fe", "Cr", "Mn", "Mo",
    "W", "V", "Ti", "Zn", "Ru", "Rh", "Ir", "Al",
}

FUNC_NORMALISE = {
    "pbe": "pbe", "pbe_d3": "pbe_d3", "pbe+d3": "pbe_d3", "pbed3": "pbe_d3",
    "r2scan": "r2scan", "beef_vdw": "beef_vdw", "beef-vdw": "beef_vdw",
    "beefvdw": "beef_vdw",
}

FUNC_DIRS = {"pbe": "PBE", "pbe_d3": "PBE_D3", "r2scan": "r2scan",
             "beef_vdw": "beef_vdw"}

KNOWN_FACETS = ["0001", "111", "110", "100", "001"]

# The MLIP .cif gallery uses common molecule names (Ag100_ethanol.cif) while the
# dft_jobs directories use formula-style tokens (C2H5OH_Ag100). Normalise both
# sides to a canonical key so DFT and MLIP structures can be matched. Add pairs
# here as new molecules appear.
MOLECULE_CANON = {
    "c2h5oh": "ethanol", "ch3ch2oh": "ethanol", "ethanol": "ethanol",
    "ch3oh": "methanol", "methanol": "methanol",
    "c2h6": "ethane", "ethane": "ethane",
    "c2h4": "ethene", "ethene": "ethene", "ethylene": "ethene",
    "ch4": "methane", "methane": "methane",
    "c2h2": "acetylene", "acetylene": "acetylene",
    "ch3cho": "acetaldehyde", "acetaldehyde": "acetaldehyde",
    "ch3cooh": "acetic_acid", "acetic_acid": "acetic_acid",
    "hcooh": "formic_acid", "formic_acid": "formic_acid",
    "ch3och3": "dme", "dme": "dme",
    "h2co": "formaldehyde", "formaldehyde": "formaldehyde",
    "h2o": "water", "water": "water",
    "co2": "co2", "co": "co", "no": "no", "n2": "n2", "nh3": "nh3",
    "h2s": "h2s", "so2": "so2", "ch3": "ch3",
    "ch3o": "methoxy", "methoxy": "methoxy",
    "oh": "hydroxyl", "hydroxyl": "hydroxyl",
    "hcn": "hcn",
    "h": "atomich", "atomich": "atomich",
    "o": "atomico", "atomico": "atomico",
    "n": "atomicn", "atomicn": "atomicn",
    "c": "atomicc", "atomicc": "atomicc",
    "s": "atomics", "atomics": "atomics",
}


def normalise_func(name: str) -> str:
    s = name.strip().lower().replace("+", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return FUNC_NORMALISE.get(s, s)


def canon_molecule(name: str) -> str:
    """Map a molecule token (formula or common name) to a canonical key."""
    return MOLECULE_CANON.get(name.strip().lower(), name.strip().lower())


def parse_surface_molecule(name: str, molecule_first: bool = False):
    """Split into (surface, molecule), auto-detecting the name order.

    Surface-first  'Ag100_ethanol'                    -> ('Ag100', 'ethanol').
    Molecule-first 'C2H5OH_Ag100' / 'H2O_Au111' / 'CH3OH_Pd111_bri'
                                                       -> ('Ag100'/'Au111'/'Pd111', ...).

    The order is detected from where the known <metal><facet> token sits, so
    surface-first (poscar/best) and molecule-first (dft_jobs) trees mix freely
    in one run. ``molecule_first`` only steers the no-known-surface fallback.
    """
    metals = sorted(METALS, key=len, reverse=True)
    # 1) Surface-first: the name starts with a known <metal><facet> token.
    for metal in metals:
        for facet in KNOWN_FACETS:
            surf = f"{metal}{facet}"
            if name.startswith(surf + "_"):
                return surf, name[len(surf) + 1:].split("_seed")[0]
    # 2) Molecule-first: a known surface token appears after an underscore
    #    (drops any trailing adsorption-site suffix, e.g. '_top', '_bri').
    for metal in metals:
        for facet in KNOWN_FACETS:
            surf = f"{metal}{facet}"
            idx = name.find("_" + surf)
            if idx != -1:
                return surf, name[:idx]
    parts = name.split("_", 1)
    if len(parts) != 2:
        return name, "unknown"
    return (parts[1].split("_")[0], parts[0]) if molecule_first else (parts[0], parts[1])


def surface_metal(surface: str) -> str:
    m = re.match(r"[A-Za-z]+", surface or "")
    letters = m.group() if m else surface
    if letters[:2] in METALS:
        return letters[:2]
    if letters[:1] in METALS:
        return letters[:1]
    return letters[:2]


def min_metal_adsorbate_contact(atoms: "Atoms", metal: str):
    """Nearest metal<->non-metal distance using minimum-image PBC.

    Returns (min_dist, "El1-El2", i_metal, j_ads) or (None, "", -1, -1).
    """
    syms = atoms.get_chemical_symbols()
    metal_idx = [i for i, s in enumerate(syms) if s == metal]
    ads_idx = [i for i, s in enumerate(syms) if s != metal]
    if not metal_idx or not ads_idx:
        return None, "", -1, -1
    best = None
    best_pair = ("", "")
    best_ij = (-1, -1)
    for i in metal_idx:
        # ASE handles PBC minimum-image when the cell is periodic.
        d = atoms.get_distances(i, ads_idx, mic=True)
        j_local = int(np.argmin(d))
        dij = float(d[j_local])
        if best is None or dij < best:
            best = dij
            j = ads_idx[j_local]
            best_pair = (syms[i], syms[j])
            best_ij = (i, j)
    pair = f"{best_pair[0]}-{best_pair[1]}"
    return best, pair, best_ij[0], best_ij[1]


def per_atom_rmsd(a: "Atoms", b: "Atoms"):
    """RMSD and max single-atom displacement between two same-ordered sets.

    Both structures derive from the same POSCAR, so atoms match by index.
    Removes the mean (centre-of-mass-free) shift before comparing so a rigid
    translation of the slab does not inflate the RMSD. Returns
    (rmsd, max_disp, max_disp_atom_label) or (None, None, "").
    """
    if len(a) != len(b):
        return None, None, ""
    if a.get_chemical_symbols() != b.get_chemical_symbols():
        return None, None, ""
    pa = a.get_positions()
    pb = b.get_positions()
    pa = pa - pa.mean(axis=0)
    pb = pb - pb.mean(axis=0)
    disp = np.linalg.norm(pa - pb, axis=1)
    rmsd = float(np.sqrt(np.mean(disp ** 2)))
    k = int(np.argmax(disp))
    label = f"{a.get_chemical_symbols()[k]}{k}"
    return rmsd, float(disp[k]), label


def _read(path: Path):
    try:
        return ase_read(str(path))
    except Exception as exc:  # noqa: BLE001 - report and skip
        print(f"  ! could not read {path}: {exc}", file=sys.stderr)
        return None


def discover_dft_contcars(root: Path, molecule_first: bool = True):
    """Yield (surface, molecule, canon, func_key, contcar_path).

    Walks up from each CONTCAR to find the functional directory, so both
    ``<system>/<FUNC>/CONTCAR`` and ``<system>/<FUNC>/<sub>/CONTCAR`` (e.g. a
    trailing ``fully_relaxed`` or ``single-point`` directory) are discovered.
    """
    for contcar in root.rglob("CONTCAR"):
        parts = contcar.parent.parts
        func_key = system = None
        for i in range(len(parts) - 1, -1, -1):
            fk = normalise_func(parts[i])
            if fk in FUNC_DIRS:
                func_key = fk
                system = parts[i - 1] if i - 1 >= 0 else None
                break
        if func_key is None or not system:
            continue
        surface, molecule = parse_surface_molecule(system, molecule_first)
        if molecule == "unknown":
            continue
        yield surface, molecule, canon_molecule(molecule), func_key, contcar


def index_mlip_cifs(mlip_dir: Path):
    """Map (surface, canon_molecule) -> preferred .cif path.

    Filenames look like Ag100_ethanol_sevennet_omni.cif or Ag100_ethanol_5m.cif
    or Ag100_ethanol.cif. Prefer sevennet_omni, then plain, then 5m. Keys are
    canonicalised so formula-named DFT jobs match common-named .cif files.
    """
    def rank(name: str) -> int:
        if "sevennet_omni" in name:
            return 0
        if name.count("_") == 1:  # Surface_molecule.cif (base)
            return 1
        if "_5m" in name:
            return 2
        return 3

    best: dict[tuple[str, str], tuple[int, Path]] = {}
    for cif in mlip_dir.glob("*.cif"):
        stem = cif.stem
        # strip known method tags to recover surface_molecule
        core = stem
        for tag in ("_sevennet_omni", "_5m_d3", "_5m", "_1m"):
            if core.endswith(tag):
                core = core[: -len(tag)]
                break
        surface, molecule = parse_surface_molecule(core)
        if molecule == "unknown":
            continue
        r = rank(stem)
        key = (surface, canon_molecule(molecule))
        if key not in best or r < best[key][0]:
            best[key] = (r, cif)
    return {k: v[1] for k, v in best.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dft-jobs", action="append", default=None, metavar="ROOT",
                    help="Root of finished DFT relaxations. Repeatable, e.g. "
                         "--dft-jobs poscar/best --dft-jobs poscar/best2 "
                         "--dft-jobs dft_jobs. Surface-first ('Au111_H2O') and "
                         "molecule-first ('H2O_Au111') trees are auto-detected. "
                         "Default: dft_jobs.")
    ap.add_argument("--mlip-dir", required=True,
                    help="Directory with MLIP relaxed .cif structures "
                         "(the bond-distance-review gallery).")
    ap.add_argument("--out", default="dft_mlip_structure_compare.csv",
                    help="Output CSV path.")
    args = ap.parse_args()

    dft_roots = [Path(d) for d in (args.dft_jobs or ["dft_jobs"])]
    mlip_dir = Path(args.mlip_dir)
    missing = [str(d) for d in dft_roots if not d.is_dir()]
    if missing:
        sys.exit("--dft-jobs not found: " + ", ".join(missing))
    if not mlip_dir.is_dir():
        sys.exit(f"--mlip-dir not found: {mlip_dir}")

    print(f"Indexing MLIP .cif files in {mlip_dir} ...")
    mlip_index = index_mlip_cifs(mlip_dir)
    print(f"  found {len(mlip_index)} MLIP structures.")

    rows = []
    n_seen = n_matched = 0
    seen_keys: set[tuple[str, str, str]] = set()
    for dft_root in dft_roots:
        for surface, molecule, canon, func, contcar in discover_dft_contcars(dft_root):
            n_seen += 1
            # Prefer the first root that provides a given (surface, molecule,
            # functional) -- pass poscar/best before dft_jobs to prioritise it.
            dedup_key = (surface, canon, func)
            if dedup_key in seen_keys:
                continue
            cif = mlip_index.get((surface, canon))
            if cif is None:
                continue
            dft = _read(contcar)
            mlip = _read(cif)
            if dft is None or mlip is None:
                continue

            metal = surface_metal(surface)
            d_dft, pair_dft, _, _ = min_metal_adsorbate_contact(dft, metal)
            d_ml, pair_ml, _, _ = min_metal_adsorbate_contact(mlip, metal)
            rmsd, max_disp, disp_atom = per_atom_rmsd(dft, mlip)

            seen_keys.add(dedup_key)
            n_matched += 1
            rows.append({
                "surface": surface,
                "molecule": molecule,
                "functional": func,
                "min_dist_dft": None if d_dft is None else round(d_dft, 3),
                "pair_dft": pair_dft,
                "min_dist_mlip": None if d_ml is None else round(d_ml, 3),
                "pair_mlip": pair_ml,
                "d_min_dist": (None if (d_dft is None or d_ml is None)
                               else round(d_dft - d_ml, 3)),
                "rmsd": None if rmsd is None else round(rmsd, 3),
                "max_disp": None if max_disp is None else round(max_disp, 3),
                "max_disp_atom": disp_atom,
                "mlip_cif": cif.name,
            })

    if not rows:
        sys.exit("No matched DFT/MLIP structure pairs found. Check paths.")

    rows.sort(key=lambda r: (r["functional"], r["surface"], r["molecule"]))
    fields = ["surface", "molecule", "functional", "min_dist_dft", "pair_dft",
              "min_dist_mlip", "pair_mlip", "d_min_dist", "rmsd", "max_disp",
              "max_disp_atom", "mlip_cif"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nScanned {n_seen} DFT CONTCARs, matched {n_matched} to MLIP.")
    print(f"Wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
