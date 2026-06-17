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

Output:

    poscar/
        Cu111_isopropanol_seed0/POSCAR
        Cu110_isopropanol_seed0/POSCAR
        Cu001_isopropanol_seed0/POSCAR
        ...
    poscar/best/
        Cu111_isopropanol/POSCAR    <- lowest E_ads across all seeds per surface
        Cu110_isopropanol/POSCAR
        Cu001_isopropanol/POSCAR

Usage
-----
    python extract_poscar.py                          # default: reads ./runs, writes ./poscar
    python extract_poscar.py --runs-dir /path/to/runs
    python extract_poscar.py --best-only              # only write best-seed per surface
    python extract_poscar.py --verbose
"""

import argparse
import json
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

    # Write to a temporary string buffer first, then prepend the comment line.
    # ASE writes the system name as the first line of the POSCAR;
    # we overwrite it with our informative comment.
    import io
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

def collect_runs(runs_dir: Path) -> list:
    """
    Walk runs_dir and collect all task entries that have a final_adsorbed
    geometry.
    """
    entries = []

    for task_dir in sorted(runs_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        # Directory name pattern: <surface>_<adsorbate>_seed<N>_<calc>
        # e.g.  Cu111_isopropanol_seed0_sevennet_omni
        name = task_dir.name
        parts = name.split("_seed")
        if len(parts) != 2:
            continue

        surface_adsorbate = parts[0]       # e.g. Cu111_isopropanol
        seed_calc         = parts[1]       # e.g. 0_sevennet_omni

        seed_str, *calc_parts = seed_calc.split("_")
        calculator = "_".join(calc_parts)

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
            "calculator":    calculator,
            "atoms":         atoms,
            "E_ads_eV":      meta.get("E_ads_eV",      None),
            "E_total_eV":    meta.get("E_total_eV",    None),
            "E_surface_eV":  meta.get("E_surface_eV",  None),
            "E_molecule_eV": meta.get("E_molecule_eV", None),
            "timestamp":     meta.get("timestamp",     ""),
        })

    return entries


def select_best_per_system(entries: list) -> dict:
    """
    For each unique (surface, adsorbate) pair, return the entry with the
    lowest E_ads_eV across all seeds.
    """
    best = {}
    for e in entries:
        key = f"{e['surface']}_{e['adsorbate']}"
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
    parser.add_argument("--runs-dir", default="runs",
                        help="Path to the GOAD runs/ directory (default: ./runs)")
    parser.add_argument("--out-dir",  default="poscar",
                        help="Output directory for POSCAR files (default: ./poscar)")
    parser.add_argument("--best-only", action="store_true",
                        help="Only write the best-seed POSCAR per (surface, adsorbate)")
    parser.add_argument("--no-sort", action="store_true",
                        help="Do NOT sort atoms by species (default: sort, VASP convention)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print one line per POSCAR written")
    args = parser.parse_args()

    runs_dir     = Path(args.runs_dir)
    out_dir      = Path(args.out_dir)
    sort_species = not args.no_sort

    if not runs_dir.exists():
        print(f"ERROR: runs directory not found: {runs_dir}")
        raise SystemExit(1)

    entries = collect_runs(runs_dir)

    if not entries:
        print(f"No completed runs with final_adsorbed.cif/.traj found in {runs_dir}")
        raise SystemExit(0)

    print(f"Found {len(entries)} completed run(s) in {runs_dir}")

    written = 0

    # ---- Per-seed POSCARs --------------------------------------------------
    if not args.best_only:
        for e in entries:
            label   = f"{e['surface']}_{e['adsorbate']}_seed{e['seed']}"
            poscar  = out_dir / label / "POSCAR"
            e_str   = f"{e['E_ads_eV']:.4f} eV" if e["E_ads_eV"] is not None else "E_ads unknown"
            comment = (f"{e['surface']} + {e['adsorbate']} | seed={e['seed']} | "
                       f"calc={e['calculator']} | E_ads={e_str}")
            write_poscar(e["atoms"], poscar, sort_species=sort_species, comment=comment)
            if args.verbose:
                print(f"  [per-seed]  {poscar}   E_ads={e_str}")
            written += 1

    # ---- Best-seed POSCARs -------------------------------------------------
    best     = select_best_per_system(entries)
    best_dir = out_dir / "best"

    for key, e in sorted(best.items()):
        label   = f"{e['surface']}_{e['adsorbate']}"
        poscar  = best_dir / label / "POSCAR"
        e_str   = f"{e['E_ads_eV']:.4f} eV" if e["E_ads_eV"] is not None else "E_ads unknown"
        comment = (f"{e['surface']} + {e['adsorbate']} | BEST seed={e['seed']} | "
                   f"calc={e['calculator']} | E_ads={e_str}")
        write_poscar(e["atoms"], poscar, sort_species=sort_species, comment=comment)
        print(f"  [best]  {poscar}   seed={e['seed']}  E_ads={e_str}")
        written += 1

    # ---- Summary -----------------------------------------------------------
    print(f"\nWrote {written} POSCAR file(s) under {out_dir}/")
    print()
    print("Per-seed layout:   poscar/<surface>_<adsorbate>_seed<N>/POSCAR")
    print("Best-seed layout:  poscar/best/<surface>_<adsorbate>/POSCAR  <- use for DFT")
    print()
    print("Next steps:")
    print("  1. cp poscar/best/<system>/POSCAR  ~/vasp-jobs/<system>/")
    print("  2. Add INCAR, KPOINTS, POTCAR")
    print("  3. NSW=0 (single-point) or NSW=20 IBRION=2 (short relax)")
    print("  4. Compare VASP E_ads with GOAD E_ads_eV in result.json")


if __name__ == "__main__":
    main()
