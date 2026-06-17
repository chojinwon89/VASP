#!/usr/bin/env python
"""
setup_vasp_jobs.py
==================
After running extract_poscar.py, this script populates each POSCAR directory
with the remaining VASP input files needed for DFT verification:

    INCAR        -- your settings from the GOAD-MLIP meeting
    KPOINTS      -- Monkhorst-Pack 2x2x1
    POTCAR       -- concatenated PBE PAW potentials, species order from POSCAR
    slm.vasp.kestrel -- Kestrel Slurm submission script

Usage
-----
    # Generate inputs for the best-seed POSCARs (recommended):
    python setup_vasp_jobs.py

    # Generate inputs for ALL per-seed POSCARs as well:
    python setup_vasp_jobs.py --all-seeds

    # Specify a different poscar/ directory:
    python setup_vasp_jobs.py --poscar-dir /path/to/poscar

    # Dry-run: print what would be done without writing anything:
    python setup_vasp_jobs.py --dry-run

Prerequisites
-------------
The VASP PBE PAW pseudopotential library must be available on the machine.
Set the environment variable VASP_PP_PATH to the root of your POTCAR library,
e.g. in your ~/.bashrc:

    export VASP_PP_PATH=/projects/vasp/pps/PBE54

The script looks for pseudopotentials in:
    $VASP_PP_PATH/<Element>/POTCAR        (preferred)
    $VASP_PP_PATH/<Element>_pv/POTCAR     (high-valence variant)
    $VASP_PP_PATH/<Element>_sv/POTCAR     (semi-core variant)

POTCAR element mapping (PBE recommended PAW sets for common elements):
    Cu  -> Cu   (or Cu_pv for more accurate d-bands)
    C   -> C
    H   -> H
    O   -> O
"""

import argparse
import os
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# INCAR template  (settings from your research group)
# ---------------------------------------------------------------------------
INCAR_TEMPLATE = """\
SYSTEM = {system}

! Startparameter
NWRITE = 2
ISTART = 0
ISPIN  = 2

! Electronic Relaxation
ENCUT  = 450
NELM   = 150
NELMIN = 4
EDIFF  = 1E-05
EDIFFG = -5E-02

! Exchange-correlation
GGA = RP

! Ionic Relaxation
NSW    = 1000
IBRION = 2
POTIM  = 0.3

! DOS
ISMEAR = 1
SIGMA  = 0.05

! Algorithmic
IALGO  = 48
LDIAG  = .TRUE.
LREAL  = A
LWAVE  = .FALSE.
"""

# ---------------------------------------------------------------------------
# KPOINTS template  (Monkhorst-Pack 2x2x1 for slab)
# ---------------------------------------------------------------------------
KPOINTS_TEMPLATE = """\
Monkhorst-Pack 2x2x1
 0
Monkhorst-Pack
  2  2  1
  0  0  0
"""

# ---------------------------------------------------------------------------
# Slurm template  (Kestrel, your exact header)
# ---------------------------------------------------------------------------
SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=17
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --account=ccpc
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err

export OMP_NUM_THREADS=1
export OMP_STACKSIZE=3G
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
ulimit -s unlimited

module load vasp/6.3.2_openMP+tpc

srun vasp_std
"""

# ---------------------------------------------------------------------------
# POTCAR element name mapping
# Keys: chemical symbol  Values: preferred PAW-PBE folder names (in order of
# preference — first one that exists on disk is used)
# ---------------------------------------------------------------------------
POTCAR_MAP = {
    "Cu": ["Cu_pv", "Cu"],
    "C":  ["C"],
    "H":  ["H"],
    "O":  ["O"],
    "N":  ["N"],
    "S":  ["S"],
    "Pt": ["Pt_pv", "Pt"],
    "Pd": ["Pd_pv", "Pd"],
    "Ni": ["Ni_pv", "Ni"],
    "Ag": ["Ag_pv", "Ag"],
    "Au": ["Au_pv", "Au"],
    "Fe": ["Fe_pv", "Fe"],
    "Co": ["Co_pv", "Co"],
    "Zn": ["Zn_pv", "Zn"],
    "Al": ["Al"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_species_from_poscar(poscar_path: Path) -> list:
    """
    Parse the element species line from a VASP5 POSCAR.
    Line 6 (0-indexed: line 5) lists element symbols.
    Returns e.g. ['Cu', 'C', 'H', 'O']
    """
    lines = poscar_path.read_text().splitlines()
    if len(lines) < 6:
        raise ValueError(f"POSCAR too short: {poscar_path}")
    species = lines[5].split()
    # Validate: should be element symbols (start with uppercase letter)
    if not all(re.match(r"^[A-Z][a-z]?$", s) for s in species):
        raise ValueError(
            f"Could not parse species from POSCAR line 6: '{lines[5]}'\n"
            f"Check that vasp5=True was used when writing the POSCAR."
        )
    return species


def find_potcar(element: str, pp_root: Path) -> Path:
    """
    Find the POTCAR file for a given element under pp_root.
    Tries each folder in POTCAR_MAP[element] in order.
    Raises FileNotFoundError if none found.
    """
    candidates = POTCAR_MAP.get(element, [element])
    tried = []
    for folder in candidates:
        p = pp_root / folder / "POTCAR"
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        f"POTCAR for element '{element}' not found under {pp_root}.\n"
        f"Tried: {tried}\n"
        f"Set VASP_PP_PATH correctly or add '{element}' to POTCAR_MAP."
    )


def build_potcar(species: list, pp_root: Path, out_path: Path,
                 dry_run: bool = False) -> bool:
    """
    Concatenate individual element POTCAR files into a combined POTCAR.
    Returns True on success, False if any element POTCAR is missing.
    """
    potcar_parts = []
    for el in species:
        try:
            p = find_potcar(el, pp_root)
            potcar_parts.append(p)
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}")
            return False

    if not dry_run:
        with out_path.open("w") as fout:
            for p in potcar_parts:
                fout.write(p.read_text())

    return True


def setup_job_dir(job_dir: Path, system_name: str,
                  pp_root: Path, dry_run: bool = False) -> dict:
    """
    Write INCAR, KPOINTS, POTCAR, slm.vasp.kestrel into job_dir.
    job_dir must already contain a POSCAR.
    Returns a status dict.
    """
    poscar = job_dir / "POSCAR"
    if not poscar.exists():
        return {"status": "skipped", "reason": "no POSCAR"}

    status = {"dir": str(job_dir), "status": "ok", "warnings": []}

    # ---- Parse species from POSCAR ------------------------------------------
    try:
        species = read_species_from_poscar(poscar)
    except ValueError as e:
        return {"status": "error", "reason": str(e)}

    status["species"] = species

    # ---- INCAR ---------------------------------------------------------------
    incar_path = job_dir / "INCAR"
    if not dry_run:
        incar_path.write_text(INCAR_TEMPLATE.format(system=system_name))

    # ---- KPOINTS -------------------------------------------------------------
    kpoints_path = job_dir / "KPOINTS"
    if not dry_run:
        kpoints_path.write_text(KPOINTS_TEMPLATE)

    # ---- POTCAR -------------------------------------------------------------
    potcar_path = job_dir / "POTCAR"
    if pp_root is not None:
        ok = build_potcar(species, pp_root, potcar_path, dry_run=dry_run)
        if not ok:
            status["warnings"].append(
                "POTCAR not written — one or more element POTCARs not found. "
                "Run: cat $VASP_PP_PATH/<El>/POTCAR ... > POTCAR manually."
            )
            status["status"] = "partial"
    else:
        # No pp_root set — write a helper script instead
        cat_cmd = " ".join(
            f"$VASP_PP_PATH/{POTCAR_MAP.get(el, [el])[0]}/POTCAR"
            for el in species
        )
        note = (
            f"# POTCAR not generated automatically.\n"
            f"# Set VASP_PP_PATH and run:\n"
            f"cat {cat_cmd} > {potcar_path}\n"
        )
        if not dry_run:
            (job_dir / "make_potcar.sh").write_text(note)
        status["warnings"].append(
            "VASP_PP_PATH not set. Written make_potcar.sh instead — "
            "run it to build POTCAR."
        )
        status["status"] = "partial"

    # ---- Slurm script -------------------------------------------------------
    job_name = system_name.replace(" ", "_").replace("+", "")[:40]
    slurm_path = job_dir / "slm.vasp.kestrel"
    if not dry_run:
        slurm_path.write_text(
            SLURM_TEMPLATE.format(job_name=job_name)
        )
        slurm_path.chmod(0o755)

    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Populate VASP job directories (from extract_poscar.py output) "
            "with INCAR, KPOINTS, POTCAR, and slm.vasp.kestrel."
        )
    )
    parser.add_argument(
        "--poscar-dir", default="poscar",
        help="Root of the poscar/ directory written by extract_poscar.py (default: ./poscar)"
    )
    parser.add_argument(
        "--all-seeds", action="store_true",
        help="Also set up per-seed directories, not just poscar/best/"
    )
    parser.add_argument(
        "--pp-path", default=None,
        help="Path to VASP PBE PAW pseudopotential library root "
             "(overrides VASP_PP_PATH env var)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files"
    )
    args = parser.parse_args()

    poscar_root = Path(args.poscar_dir)
    if not poscar_root.exists():
        print(f"ERROR: poscar directory not found: {poscar_root}")
        print("Run extract_poscar.py first.")
        raise SystemExit(1)

    # Resolve POTCAR library path
    pp_path_str = args.pp_path or os.environ.get("VASP_PP_PATH", "")
    pp_root = Path(pp_path_str) if pp_path_str else None
    if pp_root is None:
        print("WARNING: VASP_PP_PATH is not set and --pp-path not given.")
        print("         POTCAR will NOT be built automatically.")
        print("         A make_potcar.sh helper will be written instead.")
        print()
    else:
        print(f"Using POTCAR library: {pp_root}")

    # Collect job directories to process
    job_dirs = []

    # Always include poscar/best/
    best_dir = poscar_root / "best"
    if best_dir.exists():
        for d in sorted(best_dir.iterdir()):
            if d.is_dir() and (d / "POSCAR").exists():
                job_dirs.append(d)

    # Optionally include per-seed directories
    if args.all_seeds:
        for d in sorted(poscar_root.iterdir()):
            if d.is_dir() and d.name != "best" and (d / "POSCAR").exists():
                job_dirs.append(d)

    if not job_dirs:
        print(f"No POSCAR directories found under {poscar_root}.")
        print("Run extract_poscar.py first to generate POSCAR files.")
        raise SystemExit(0)

    print(f"Setting up {len(job_dirs)} VASP job director(y/ies):")
    print()

    all_ok = True
    for job_dir in job_dirs:
        system_name = job_dir.name   # e.g. Cu111_isopropanol
        action = "[DRY-RUN]" if args.dry_run else "writing"
        print(f"  {action}: {job_dir}/")

        result = setup_job_dir(
            job_dir, system_name,
            pp_root=pp_root,
            dry_run=args.dry_run,
        )

        species_str = " ".join(result.get("species", []))
        print(f"    species:  {species_str}")

        if not args.dry_run:
            files = ["INCAR", "KPOINTS", "slm.vasp.kestrel"]
            if result["status"] == "ok":
                files.append("POTCAR")
            print(f"    written:  {', '.join(files)}")

        for w in result.get("warnings", []):
            print(f"    WARNING:  {w}")
            all_ok = False

        print()

    # ---- Final instructions --------------------------------------------------
    print("=" * 65)
    print("NEXT STEPS")
    print("=" * 65)
    print()
    print("1. If POTCAR was not built automatically, run make_potcar.sh")
    print("   in each job directory, or set VASP_PP_PATH and re-run:")
    print()
    print("     export VASP_PP_PATH=/projects/vasp/pps/PBE54")
    print("     python setup_vasp_jobs.py")
    print()
    print("2. Submit each job:")
    print()
    print("     cd poscar/best/Cu111_isopropanol")
    print("     sbatch slm.vasp.kestrel")
    print()
    print("3. After VASP finishes, compute E_ads from OUTCAR:")
    print()
    print("     grep 'free  energy' OUTCAR | tail -1")
    print()
    print("   Compare that total energy with E_surf + E_mol")
    print("   from your GOAD result.json to validate E_ads.")
    print()
    if not all_ok:
        print("Some warnings were raised — see above.")


if __name__ == "__main__":
    main()
