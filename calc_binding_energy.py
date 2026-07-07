#!/usr/bin/env python
"""
calc_binding_energy.py
======================
Calculate DFT adsorption (binding) energies from VASP OUTCARs.

Directory layout expected
-------------------------
    poscar/best/<surface>_<molecule>/OUTCAR   ← slab + adsorbed molecule
    vasp_slab/<surface>/OUTCAR                ← clean slab reference
    vasp_mol/<molecule>/OUTCAR                ← gas-phase molecule reference

Formula
-------
    E_ads = E(slab+mol) - E(slab) - E(mol)

Usage
-----
    # Auto-discover everything under poscar/best/
    python calc_binding_energy.py

    # Custom paths
    python calc_binding_energy.py \
        --best-dirs poscar/best poscar/best2 \
        --slab-dir vasp_slab \
        --mol-dir  vasp_mol

    # Save results to CSV
    python calc_binding_energy.py --output dft_binding_energies.csv
"""

import argparse
import csv
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# OUTCAR parser
# ---------------------------------------------------------------------------

def read_energy_from_outcar(outcar_path: Path) -> float:
    """
    Extract the final total energy (free energy, sigma->0) from a VASP OUTCAR.
    Reads the last occurrence of:
        free  energy   TOTEN  =   -123.456 eV
    Raises FileNotFoundError if OUTCAR missing, ValueError if no energy found.
    """
    if not outcar_path.exists():
        raise FileNotFoundError(f"OUTCAR not found: {outcar_path}")

    energy = None
    with outcar_path.open() as f:
        for line in f:
            if "free  energy   TOTEN" in line:
                try:
                    energy = float(line.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    pass

    if energy is None:
        raise ValueError(
            f"No 'free  energy   TOTEN' line found in {outcar_path}.\n"
            "Check that the VASP job completed successfully."
        )
    return energy


# ---------------------------------------------------------------------------
# Directory name parsing
# ---------------------------------------------------------------------------

def parse_surface_molecule(dir_name: str):
    """
    Parse a directory name like 'Cu111_isopropanol' or 'Pt111_glycerol_seed0'
    into (surface, molecule).

    Tries known surface names first (e.g. Cu111, Pt110, Au100 ...),
    then falls back to splitting on the first underscore.
    """
    KNOWN_METALS = ["Cu", "Pt", "Pd", "Ni", "Ag", "Au", "Fe", "Co", "Zn", "Al"]
    KNOWN_FACETS = ["111", "110", "100", "001"]

    for metal in KNOWN_METALS:
        for facet in KNOWN_FACETS:
            surface = f"{metal}{facet}"
            if dir_name.startswith(surface + "_"):
                remainder = dir_name[len(surface) + 1:]
                # Strip trailing _seed<N> if present
                molecule = remainder.split("_seed")[0]
                return surface, molecule

    # Fallback: split on first underscore
    parts = dir_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1].split("_seed")[0]

    return dir_name, "unknown"


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def calc_binding_energies(best_dirs, slab_dir: Path, mol_dir: Path):
    """
    Walk one or more best directories, compute E_ads for each system.
    Returns list of dicts with keys:
        system, surface, molecule, source_dir,
        E_slab_mol, E_slab, E_mol, E_ads,
        status, note
    """
    results = []

    for best_dir in best_dirs:
        job_dirs = sorted(
            d for d in best_dir.iterdir()
            if d.is_dir() and (d / "OUTCAR").exists()
        )

        if not job_dirs:
            print(f"No OUTCAR-containing directories found under {best_dir}")
            continue

        for job_dir in job_dirs:
            system = job_dir.name
            surface, molecule = parse_surface_molecule(system)

            row = {
                "system":     system,
                "surface":    surface,
                "molecule":   molecule,
                "source_dir": str(best_dir),
                "E_slab_mol": None,
                "E_slab":     None,
                "E_mol":      None,
                "E_ads":      None,
                "status":     "ok",
                "note":       "",
            }

            notes = []

            # 1) Slab + molecule energy
            try:
                row["E_slab_mol"] = read_energy_from_outcar(job_dir / "OUTCAR")
            except (FileNotFoundError, ValueError) as e:
                notes.append(f"slab+mol: {e}")
                row["status"] = "error"

            # 2) Clean slab energy
            slab_outcar = slab_dir / surface / "OUTCAR"
            try:
                row["E_slab"] = read_energy_from_outcar(slab_outcar)
            except (FileNotFoundError, ValueError) as e:
                notes.append(f"slab: {e}")
                row["status"] = "error"

            # 3) Gas molecule energy
            mol_outcar = mol_dir / molecule / "OUTCAR"
            try:
                row["E_mol"] = read_energy_from_outcar(mol_outcar)
            except (FileNotFoundError, ValueError) as e:
                notes.append(f"mol: {e}")
                row["status"] = "error"

            # 4) E_ads
            if row["status"] == "ok":
                row["E_ads"] = row["E_slab_mol"] - row["E_slab"] - row["E_mol"]
            else:
                row["note"] = "; ".join(notes)

            results.append(row)

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_table(results):
    """Pretty-print results to stdout."""
    header = (
        f"{'System':<35} {'Source':<18} {'E(slab+mol)':>14} {'E(slab)':>14} "
        f"{'E(mol)':>12} {'E_ads(eV)':>12}  Status"
    )
    print()
    print(header)
    print("-" * len(header))

    for r in results:
        if r["status"] == "ok":
            print(
                f"{r['system']:<35} "
                f"{Path(r['source_dir']).name:<18} "
                f"{r['E_slab_mol']:>14.6f} "
                f"{r['E_slab']:>14.6f} "
                f"{r['E_mol']:>12.6f} "
                f"{r['E_ads']:>12.4f}  ok"
            )
        else:
            print(
                f"{r['system']:<35} {Path(r['source_dir']).name:<18} {'---':>14} {'---':>14} "
                f"{'---':>12} {'---':>12}  ERROR: {r['note']}"
            )

    print()

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        print(f"Summary: {len(ok_results)}/{len(results)} systems computed successfully")
        eads_vals = [r["E_ads"] for r in ok_results]
        print(f"  E_ads range : {min(eads_vals):.4f} to {max(eads_vals):.4f} eV")
        print(
            f"  Most stable : "
            f"{min(ok_results, key=lambda x: x['E_ads'])['system']} "
            f"({min(eads_vals):.4f} eV)"
        )
    print()


def write_csv(results, output_path: Path):
    """Write results to CSV."""
    fields = [
        "system", "surface", "molecule", "source_dir",
        "E_slab_mol", "E_slab", "E_mol", "E_ads",
        "status", "note",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate DFT adsorption energies from VASP OUTCARs."
    )
    parser.add_argument(
        "--best-dirs", nargs="+", default=["poscar/best"],
        help="One or more directories containing slab+molecule VASP jobs "
             "(default: poscar/best)"
    )
    parser.add_argument(
        "--slab-dir", default="vasp_slab",
        help="Directory containing clean slab VASP jobs (default: vasp_slab)"
    )
    parser.add_argument(
        "--mol-dir", default="vasp_mol",
        help="Directory containing gas molecule VASP jobs (default: vasp_mol)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results to this CSV file (default: print to screen only)"
    )
    args = parser.parse_args()

    best_dirs = [Path(d) for d in args.best_dirs]
    slab_dir = Path(args.slab_dir)
    mol_dir  = Path(args.mol_dir)

    # Validate directories
    missing = [str(d) for d in [*best_dirs, slab_dir, mol_dir] if not d.exists()]
    if missing:
        print("ERROR: The following directories were not found:")
        for m in missing:
            print(f"  {m}")
        print()
        print("Make sure VASP jobs have been run and paths are correct.")
        print("Use --best-dirs / --slab-dir / --mol-dir to override defaults.")
        sys.exit(1)

    for d in best_dirs:
        print(f"slab+mol jobs : {d}")
    print(f"slab refs     : {slab_dir}")
    print(f"molecule refs : {mol_dir}")

    results = calc_binding_energies(best_dirs, slab_dir, mol_dir)

    if not results:
        print("No results computed.")
        sys.exit(0)

    print_table(results)

    if args.output:
        write_csv(results, Path(args.output))


if __name__ == "__main__":
    main()
