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

Single-point reference layout (also supported with --calc-type single-point):
    vasp_slab/<surface>/singlepoint/<functional>/OUTCAR
    vasp_mol/<molecule>/singlepoint/<functional>/OUTCAR

best_dir discovery for slab+molecule jobs mirrors setup_vasp_jobs.py:
  (1) bucketed root: DIR/C<n>/<system>/...
  (2) single bucket: DIR/<system>/...
  (3) direct system: DIR is itself one <system> directory

Path resolution notes:
  - vasp_slab and vasp_mol are not carbon-bucketed.
  - With --calc-type single-point, slab+mol AND the slab/molecule references are
    read from a singlepoint/ subdirectory when present, falling back to the
    plain <functional>/ (or flat) layout for older runs that lack it.

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
    # (writes dft_binding_energies_singlepoint_all.csv by default)
    python calc_binding_energy.py --best-dirs poscar/best --calc-type single-point \\
        --all-functionals --functionals PBE PBE_D3 r2scan beef_vdw

    # Flat layout (original behaviour)
    python calc_binding_energy.py --output dft_binding_energies.csv
"""

import argparse
import csv
import os
import re
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

def _resolve_outcar(outcar_path: Path) -> Path:
    """
    Given an intended OUTCAR path, return the actual file to read.

    Handles the common cluster cases where the directory clearly contains an
    OUTCAR but the literal path is not directly readable:
      - the OUTCAR has been gzip-compressed to OUTCAR.gz
      - the OUTCAR is a symlink (resolved transparently by open())

    Preference order: OUTCAR, then OUTCAR.gz. Returns the original path
    unchanged if neither exists, so callers can raise a clear FileNotFoundError.
    """
    if outcar_path.exists():
        return outcar_path
    gz = outcar_path.with_name(outcar_path.name + ".gz")
    if gz.exists():
        return gz
    return outcar_path


def read_energy_from_outcar(outcar_path: Path) -> float:
    """
    Extract the final total energy (free energy, sigma->0) from a VASP OUTCAR.
    Reads the last occurrence of:
        free  energy   TOTEN  =   -123.456 eV
    Transparently reads a gzip-compressed OUTCAR.gz when the plain OUTCAR is
    absent. Raises FileNotFoundError if neither is present (with a hint when a
    dangling symlink is detected), ValueError if no energy found.
    """
    import gzip

    resolved = _resolve_outcar(outcar_path)

    if not resolved.exists():
        abs_hint = ""
        try:
            abs_path = outcar_path.resolve()
            if str(abs_path) != str(outcar_path):
                abs_hint = f" (resolved: {abs_path})"
        except OSError:
            pass
        # Distinguish a broken symlink (ls shows it, but target is missing)
        # from a genuinely absent file for a clearer message.
        if outcar_path.is_symlink():
            raise FileNotFoundError(
                f"OUTCAR not found: {outcar_path}{abs_hint} "
                f"(broken symlink -> {os.readlink(outcar_path)})"
            )
        raise FileNotFoundError(f"OUTCAR not found: {outcar_path}{abs_hint}")

    opener = gzip.open if resolved.suffix == ".gz" else open

    energy = None
    with opener(resolved, "rt") as f:
        for line in f:
            if "free  energy   TOTEN" in line:
                try:
                    energy = float(line.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    pass

    if energy is None:
        raise ValueError(
            f"No 'free  energy   TOTEN' line found in {resolved}.\n"
            "Check that the VASP job completed successfully."
        )
    return energy


# ---------------------------------------------------------------------------
# Directory name parsing
# ---------------------------------------------------------------------------

# Molecule token aliases: dft_jobs uses formula-style tokens that do not
# always match the vasp_mol/ reference directory names. Map job-dir token ->
# vasp_mol/ directory name. Tokens with no gas-phase reference (radicals /
# atoms not computed as isolated molecules) are intentionally left unmapped so
# the run reports them as missing E_mol rather than silently mispairing.
MOLECULE_ALIASES = {
    "C2H5OH": "CH3CH2OH",
}

KNOWN_METALS = ["Cu", "Pt", "Pd", "Ni", "Ag", "Au", "Fe", "Co", "Zn", "Al",
                "Rh", "Ir", "Ru", "Mo", "Mn", "Cr", "Ti", "V", "W"]
KNOWN_FACETS = ["0001", "111", "110", "100", "001"]


def parse_surface_molecule(dir_name: str, molecule_first: bool = False):
    """
    Parse a directory name into (surface, molecule).

    Default (surface-first) handles 'Cu111_isopropanol' / 'Pt111_glycerol_seed0'.
    With ``molecule_first=True`` handles the dft_jobs layout 'C2H2_Ag100' or
    'CH3OH_Pd111_bri' where the molecule token comes first and the surface may
    be followed by an adsorption-site suffix (e.g. '_bri', '_top').

    Tries known surface names anywhere in the name; falls back to splitting on
    the first underscore.
    """
    # Locate a known <metal><facet> token anywhere in the name.
    for metal in KNOWN_METALS:
        for facet in KNOWN_FACETS:
            surface = f"{metal}{facet}"
            if molecule_first:
                needle = "_" + surface
                idx = dir_name.find(needle)
                if idx != -1:
                    molecule = dir_name[:idx]
                    return surface, molecule
            else:
                if dir_name.startswith(surface + "_"):
                    remainder = dir_name[len(surface) + 1:]
                    molecule = remainder.split("_seed")[0]
                    return surface, molecule

    # Fallback: split on first underscore (order depends on molecule_first).
    parts = dir_name.split("_", 1)
    if len(parts) == 2:
        if molecule_first:
            return parts[1].split("_")[0], parts[0]
        return parts[0], parts[1].split("_seed")[0]

    return dir_name, "unknown"


# ---------------------------------------------------------------------------
# Main calculation (one functional)
# ---------------------------------------------------------------------------

def discover_system_dirs(best_dir: Path):
    """
    Discover system directories from a best-dir root, mirroring setup_vasp_jobs.py:
      (1) DIR itself is a system directory (contains POSCAR)
      (2) DIR contains C<n>/<system>/POSCAR
      (3) fallback (no C<n> buckets): DIR contains <system>/POSCAR
    """
    def is_bucket_dir(path: Path) -> bool:
        return path.is_dir() and re.fullmatch(r"C\d+", path.name) is not None

    system_dirs = []
    if (best_dir / "POSCAR").exists():
        system_dirs.append(best_dir)
        return system_dirs

    first_level_dirs = [d for d in sorted(best_dir.iterdir()) if d.is_dir()]
    bucket_dirs = [d for d in first_level_dirs if is_bucket_dir(d)]

    if bucket_dirs:
        seen_systems = set()
        for bucket_dir in bucket_dirs:
            for second_level_dir in sorted(bucket_dir.iterdir()):
                if second_level_dir.is_dir() and (second_level_dir / "POSCAR").exists():
                    system_dirs.append(second_level_dir)
                    seen_systems.add(second_level_dir.name)

        # Also include non-bucketed system dirs sitting alongside the C<n>/
        # buckets, unless the same system already exists in a bucket (in which
        # case the bucketed copy wins to avoid double-counting).
        for extra_dir in first_level_dirs:
            if is_bucket_dir(extra_dir):
                continue
            if not (extra_dir / "POSCAR").exists():
                continue
            if extra_dir.name in seen_systems:
                print(
                    "NOTE: non-bucketed system directory "
                    f"'{extra_dir}' duplicates a bucketed copy; using the "
                    "bucketed one and skipping this duplicate."
                )
                continue
            print(
                "NOTE: including non-bucketed system directory "
                f"'{extra_dir}' alongside the bucketed C<n>/ directories."
            )
            system_dirs.append(extra_dir)
            seen_systems.add(extra_dir.name)

        return system_dirs

    for first_level_dir in first_level_dirs:
        if (first_level_dir / "POSCAR").exists():
            system_dirs.append(first_level_dir)
            continue
        for second_level_dir in sorted(first_level_dir.iterdir()):
            if second_level_dir.is_dir() and (second_level_dir / "POSCAR").exists():
                system_dirs.append(second_level_dir)
    return system_dirs


def calc_binding_energies(best_dirs, slab_dir: Path, mol_dir: Path,
                          functional: str = None, calc_type: str = "relax",
                          molecule_first: bool = False):
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
            surface, molecule = parse_surface_molecule(system, molecule_first)
            mol_ref = MOLECULE_ALIASES.get(molecule, molecule)

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

            # Build OUTCAR paths — flat or with functional subdirectory.
            # For --calc-type single-point, slab+mol and the slab/molecule
            # references may live under a singlepoint/ subdirectory; prefer that
            # layout but fall back to the plain layout for older runs.
            def _pick(*candidates: Path) -> Path:
                """Return the first candidate with a readable OUTCAR/.gz,
                else the first candidate (so errors show the primary path)."""
                for cand in candidates:
                    if _resolve_outcar(cand).exists():
                        return cand
                return candidates[0]

            if functional:
                if calc_type == "single-point":
                    slab_mol_outcar = _pick(
                        job_dir / "singlepoint" / functional / "OUTCAR",
                        job_dir / functional / "OUTCAR",
                    )
                    slab_outcar = _pick(
                        slab_dir / surface / "singlepoint" / functional / "OUTCAR",
                        slab_dir / surface / functional / "OUTCAR",
                    )
                    mol_outcar = _pick(
                        mol_dir / mol_ref / "singlepoint" / functional / "OUTCAR",
                        mol_dir / mol_ref / functional / "OUTCAR",
                    )
                else:
                    slab_mol_outcar = job_dir / functional / "OUTCAR"
                    slab_outcar     = slab_dir / surface  / functional / "OUTCAR"
                    mol_outcar      = mol_dir  / mol_ref / functional / "OUTCAR"
            else:
                if calc_type == "single-point":
                    slab_mol_outcar = _pick(
                        job_dir / "singlepoint" / "OUTCAR",
                        job_dir / "OUTCAR",
                    )
                    slab_outcar = _pick(
                        slab_dir / surface / "singlepoint" / "OUTCAR",
                        slab_dir / surface / "OUTCAR",
                    )
                    mol_outcar = _pick(
                        mol_dir / mol_ref / "singlepoint" / "OUTCAR",
                        mol_dir / mol_ref / "OUTCAR",
                    )
                else:
                    slab_mol_outcar = job_dir / "OUTCAR"
                    slab_outcar     = slab_dir / surface  / "OUTCAR"
                    mol_outcar      = mol_dir  / mol_ref / "OUTCAR"

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


def default_output_name(calc_type: str, all_functionals: bool,
                        functional: str = None) -> str:
    """Derive the default CSV filename from the run configuration.

    Single-point runs get a ``_singlepoint`` marker so they don't clobber the
    relaxation results, e.g. ``dft_binding_energies_singlepoint_all.csv``.
    """
    sp = "_singlepoint" if calc_type == "single-point" else ""
    if all_functionals:
        return f"dft_binding_energies{sp}_all.csv"
    if functional:
        return f"dft_binding_energies{sp}_{normalise_func(functional)}.csv"
    return f"dft_binding_energies{sp}.csv"


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
  # (auto-writes dft_binding_energies_singlepoint_all.csv)
  python calc_binding_energy.py --best-dirs poscar/best --calc-type single-point \\
      --all-functionals --functionals PBE PBE_D3 r2scan beef_vdw

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
        "--molecule-first", action="store_true",
        help=(
            "Job directories are named <molecule>_<surface>[_site] (the "
            "dft_jobs layout, e.g. C2H2_Ag100, CH3OH_Pd111_bri) instead of "
            "the default <surface>_<molecule>."
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
        help=(
            "Save results to this CSV file. If omitted, a name is derived "
            "from the run type, e.g. dft_binding_energies_all.csv (relax) or "
            "dft_binding_energies_singlepoint_all.csv (single-point). "
            "Pass --no-output to only print to screen."
        )
    )
    parser.add_argument(
        "--no-output", action="store_true",
        help="Print results to screen only; do not write a CSV file."
    )
    args = parser.parse_args()

    # Resolve the output path: explicit --output wins, otherwise derive a
    # sensible default (unless --no-output was requested).
    if args.no_output:
        output_path = None
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(default_output_name(
            args.calc_type, args.all_functionals, args.functional))

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
                                            calc_type=args.calc_type,
                                            molecule_first=args.molecule_first)
            print_table(results, functional=func)
            all_results.extend(results)

        print(f"\nTotal rows across all functionals: {len(all_results)}")
        if output_path:
            write_csv(all_results, output_path, include_functional=True)
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
                                    calc_type=args.calc_type,
                                    molecule_first=args.molecule_first)

    if not results:
        print("No results computed.")
        sys.exit(0)

    print_table(results, functional=args.functional)

    if output_path:
        write_csv(results, output_path,
                  include_functional=bool(args.functional))


if __name__ == "__main__":
    main()
