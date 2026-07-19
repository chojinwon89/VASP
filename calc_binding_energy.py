#!/usr/bin/env python
"""
calc_binding_energy.py
======================
Calculate DFT adsorption (binding) energies from VASP OUTCARs.

Directory layouts supported
---------------------------
Flat layout (original):
    poscar/best/<surface>_<molecule>/OUTCAR
    vasp_slab/<surface>/OUTCAR
    vasp_mol/<molecule>/OUTCAR

Per-functional subdirectory layout:
    poscar/best/<surface>_<molecule>/<functional>/OUTCAR
    vasp_slab/<surface>/<functional>/OUTCAR
    vasp_mol/<molecule>/<functional>/OUTCAR

Bucketed slab+molecule layout (supported for --best-dirs):
    poscar/best/C<n>/<surface>_<molecule>/<functional>/OUTCAR

Single-point slab+molecule layout (supported with --calc-type single-point):
    poscar/best/C<n>/<surface>_<molecule>/singlepoint/<functional>/OUTCAR

best_dir discovery for slab+molecule jobs mirrors setup_vasp_jobs.py:
  (1) bucketed root: DIR/C<n>/<system>/...
  (2) single bucket: DIR/<system>/...
  (3) direct system: DIR is itself one <system> directory

Reference assumptions confirmed from current setup_slab_jobs.py and
setup_molecule_jobs.py:
  - vasp_slab and vasp_mol are not carbon-bucketed.
  - single-point mode in those scripts changes INCAR settings only; it does not
    add a singlepoint/ subdirectory.
  - Therefore this script applies --calc-type path switching only to slab+mol
    jobs under --best-dirs.

Formula
-------
    E_ads = E(slab+mol) - E(slab) - E(mol)

Usage
-----
    # Single functional (subdirectory layout)
    python calc_binding_energy.py --functional PBE_D3 \\
        --output dft_binding_energies_pbe_d3.csv

    # All four functionals at once → dft_binding_energies_all.csv
    python calc_binding_energy.py --all-functionals \\
        --functionals beef_vdw PBE PBE_D3 r2scan \\
        --output dft_binding_energies_all.csv

    # Single-point DFT extraction across all functionals
    python calc_binding_energy.py --best-dirs poscar/best --calc-type single-point \\
        --all-functionals --functionals PBE PBE_D3 r2scan beef_vdw \\
        --output dft_binding_energies_all.csv

    # Flat layout (original behaviour)
    python calc_binding_energy.py --output dft_binding_energies.csv
"""

import argparse
import csv
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Functional name normalisation  (directory name → canonical key)
# ---------------------------------------------------------------------------

FUNC_NORMALISE = {
    "pbe":       "pbe",
    "PBE":       "pbe",
    "pbe_d3":    "pbe_d3",
    "PBE_D3":    "pbe_d3",
    "pbe+d3":    "pbe_d3",
    "PBE+D3":    "pbe_d3",
    "r2scan":    "r2scan",
    "R2SCAN":    "r2scan",
    "beef_vdw":  "beef_vdw",
    "BEEF_VDW":  "beef_vdw",
    "beef-vdw":  "beef_vdw",
    "BEEF-vdW":  "beef_vdw",
}


def normalise_func(name: str) -> str:
    return FUNC_NORMALISE.get(name, name.lower().replace("+", "_").replace("-", "_"))


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
                molecule = remainder.split("_seed")[0]
                return surface, molecule

    # Fallback: split on first underscore
    parts = dir_name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1].split("_seed")[0]

    return dir_name, "unknown"


# ---------------------------------------------------------------------------
# Main calculation (one functional)
# ---------------------------------------------------------------------------

def discover_system_dirs(best_dir: Path):
    """
    Discover system directories from a best-dir root, mirroring setup_vasp_jobs.py:
      (1) DIR itself is a system directory (contains POSCAR)
      (2) DIR contains <system>/POSCAR
      (3) DIR contains C<n>/<system>/POSCAR
    """
    system_dirs = []
    if (best_dir / "POSCAR").exists():
        system_dirs.append(best_dir)
    for first_level_dir in sorted(best_dir.iterdir()):
        if not first_level_dir.is_dir():
            continue
        if (first_level_dir / "POSCAR").exists():
            system_dirs.append(first_level_dir)
            continue
        for second_level_dir in sorted(first_level_dir.iterdir()):
            if second_level_dir.is_dir() and (second_level_dir / "POSCAR").exists():
                system_dirs.append(second_level_dir)
    return system_dirs


def calc_binding_energies(best_dirs, slab_dir: Path, mol_dir: Path,
                          functional: str = None, calc_type: str = "relax"):
    """
    Walk one or more best directories, compute E_ads for each system.

    If *functional* is given, OUTCARs are read from:
        <job_dir>/<functional>/OUTCAR      (slab+mol)
        <slab_dir>/<surface>/<functional>/OUTCAR
        <mol_dir>/<molecule>/<functional>/OUTCAR

    Returns list of dicts with keys:
        functional, system, surface, molecule, source_dir,
        E_slab_mol, E_slab, E_mol, E_ads, status, note
    """
    results = []
    func_key = normalise_func(functional) if functional else None

    for best_dir in best_dirs:
        job_dirs = discover_system_dirs(best_dir)

        if not job_dirs:
            print(f"No system directories (with POSCAR) found under {best_dir}")
            print("Expected one of:")
            print("  - a root containing C<n>/<system>/POSCAR")
            print("  - a single bucket containing <system>/POSCAR")
            print("  - a single system directory containing POSCAR")
            continue

        for job_dir in job_dirs:
            system = job_dir.name
            surface, molecule = parse_surface_molecule(system)

            row = {
                "functional": func_key or "default",
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

            # Build OUTCAR paths — flat or with functional subdirectory
            if functional:
                if calc_type == "single-point":
                    slab_mol_outcar = job_dir / "singlepoint" / functional / "OUTCAR"
                else:
                    slab_mol_outcar = job_dir / functional / "OUTCAR"
                slab_outcar     = slab_dir / surface  / functional / "OUTCAR"
                mol_outcar      = mol_dir  / molecule / functional / "OUTCAR"
            else:
                if calc_type == "single-point":
                    slab_mol_outcar = job_dir / "singlepoint" / "OUTCAR"
                else:
                    slab_mol_outcar = job_dir / "OUTCAR"
                slab_outcar     = slab_dir / surface  / "OUTCAR"
                mol_outcar      = mol_dir  / molecule / "OUTCAR"

            # 1) Slab + molecule energy
            try:
                row["E_slab_mol"] = read_energy_from_outcar(slab_mol_outcar)
            except (FileNotFoundError, ValueError) as e:
                notes.append(f"slab+mol: {e}")
                row["status"] = "error"

            # 2) Clean slab energy
            try:
                row["E_slab"] = read_energy_from_outcar(slab_outcar)
            except (FileNotFoundError, ValueError) as e:
                notes.append(f"slab: {e}")
                row["status"] = "error"

            # 3) Gas molecule energy
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

def print_table(results, functional=None):
    """Pretty-print results to stdout."""
    func_label = f"  [{functional}]" if functional else ""
    header = (
        f"{'System':<35} {'Functional':<10} {'E(slab+mol)':>14} {'E(slab)':>14} "
        f"{'E(mol)':>12} {'E_ads(eV)':>12}  Status"
    )
    print()
    print(f"{'─'*len(header)}")
    if func_label:
        print(f"Functional: {functional}{func_label}")
    print(header)
    print("─" * len(header))

    for r in results:
        if r["status"] == "ok":
            print(
                f"{r['system']:<35} "
                f"{r['functional']:<10} "
                f"{r['E_slab_mol']:>14.6f} "
                f"{r['E_slab']:>14.6f} "
                f"{r['E_mol']:>12.6f} "
                f"{r['E_ads']:>12.4f}  ok"
            )
        else:
            print(
                f"{r['system']:<35} {r['functional']:<10} {'---':>14} {'---':>14} "
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


def write_csv(results, output_path: Path, include_functional: bool = True):
    """Write results to CSV."""
    fields = [
        "functional", "system", "surface", "molecule", "source_dir",
        "E_slab_mol", "E_slab", "E_mol", "E_ads",
        "status", "note",
    ]
    if not include_functional:
        fields = [f for f in fields if f != "functional"]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {output_path}  ({len(results)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calculate DFT adsorption energies from VASP OUTCARs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Single functional (subdirectory layout)
  python calc_binding_energy.py --functional PBE_D3 \\
      --output dft_binding_energies_pbe_d3.csv

  # All functionals at once → dft_binding_energies_all.csv
  python calc_binding_energy.py --all-functionals \\
      --functionals beef_vdw PBE PBE_D3 r2scan \\
      --output dft_binding_energies_all.csv

  # Single-point DFT extraction across all functionals
  python calc_binding_energy.py --best-dirs poscar/best --calc-type single-point \\
      --all-functionals --functionals PBE PBE_D3 r2scan beef_vdw \\
      --output dft_binding_energies_all.csv

  # Flat layout (no functional subdir)
  python calc_binding_energy.py --output dft_binding_energies.csv
"""
    )
    parser.add_argument(
        "--best-dirs", nargs="+", default=["poscar/best"],
        help="Directories containing slab+molecule VASP jobs (default: poscar/best)"
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
        "--functional", default=None,
        metavar="NAME",
        help=(
            "Functional subdirectory name to append (e.g. PBE_D3, r2scan, "
            "beef_vdw, PBE). With --calc-type single-point, slab+mol jobs are "
            "read from <system>/singlepoint/<functional>/OUTCAR. "
            "Omit for flat layout."
        )
    )
    parser.add_argument(
        "--calc-type",
        choices=["relax", "single-point"],
        default="relax",
        help=(
            "Where to read slab+mol OUTCARs from: relax -> <system>/<functional>/OUTCAR "
            "(or <system>/OUTCAR in flat mode), single-point -> "
            "<system>/singlepoint/<functional>/OUTCAR "
            "(or <system>/singlepoint/OUTCAR in flat mode). "
            "Reference slabs/molecules remain under --slab-dir/--mol-dir."
        )
    )
    parser.add_argument(
        "--all-functionals", action="store_true",
        help=(
            "Run for all functionals listed in --functionals and merge "
            "into a single CSV with a 'functional' column."
        )
    )
    parser.add_argument(
        "--functionals", nargs="+",
        default=["beef_vdw", "PBE", "PBE_D3", "r2scan"],
        metavar="NAME",
        help=(
            "Functional subdirectory names to use with --all-functionals. "
            "Default: beef_vdw PBE PBE_D3 r2scan"
        )
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results to this CSV file (default: print to screen only)"
    )
    args = parser.parse_args()

    best_dirs = [Path(d) for d in args.best_dirs]
    slab_dir  = Path(args.slab_dir)
    mol_dir   = Path(args.mol_dir)

    # Validate base directories
    missing = [str(d) for d in [*best_dirs, slab_dir, mol_dir] if not d.exists()]
    if missing:
        print("ERROR: The following directories were not found:")
        for m in missing:
            print(f"  {m}")
        print()
        print("Make sure VASP jobs have been run and paths are correct.")
        print("Use --best-dirs / --slab-dir / --mol-dir to override defaults.")
        sys.exit(1)

    # ── All-functionals mode ────────────────────────────────────────────────
    if args.all_functionals:
        all_results = []
        for func in args.functionals:
            print(f"\n{'='*60}")
            print(f"Processing functional: {func}  →  {normalise_func(func)}")
            print(f"{'='*60}")
            results = calc_binding_energies(best_dirs, slab_dir, mol_dir,
                                            functional=func,
                                            calc_type=args.calc_type)
            print_table(results, functional=func)
            all_results.extend(results)

        print(f"\nTotal rows across all functionals: {len(all_results)}")
        if args.output:
            write_csv(all_results, Path(args.output), include_functional=True)
        return

    # ── Single-functional or flat mode ─────────────────────────────────────
    for d in best_dirs:
        if args.calc_type == "single-point":
            suffix = (f"/singlepoint/{args.functional}" if args.functional
                      else "/singlepoint")
        else:
            suffix = (f"/{args.functional}" if args.functional else "")
        print(f"slab+mol jobs : {d}{suffix}")
    print(f"slab refs     : {slab_dir}"
          + (f"/<surface>/{args.functional}" if args.functional else ""))
    print(f"molecule refs : {mol_dir}"
          + (f"/<molecule>/{args.functional}" if args.functional else ""))

    results = calc_binding_energies(best_dirs, slab_dir, mol_dir,
                                    functional=args.functional,
                                    calc_type=args.calc_type)

    if not results:
        print("No results computed.")
        sys.exit(0)

    print_table(results, functional=args.functional)

    if args.output:
        write_csv(results, Path(args.output),
                  include_functional=bool(args.functional))


if __name__ == "__main__":
    main()
