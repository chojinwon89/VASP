#!/usr/bin/env python
"""
setup_vasp_jobs.py
==================
After running extract_poscar.py, this script populates each POSCAR directory
with the remaining VASP input files needed for DFT verification:

    INCAR            -- relaxation settings
    KPOINTS          -- Monkhorst-Pack 2x2x1
    POTCAR           -- concatenated PBE PAW potentials (species order from POSCAR)
    slm.vasp.kestrel -- Kestrel Slurm submission script

Usage
-----
    # Point --poscar-dir at whichever folder holds the POSCAR subdirectories:
    python setup_vasp_jobs.py --poscar-dir /scratch/jcho5/.../poscar/best
    python setup_vasp_jobs.py --poscar-dir /scratch/jcho5/.../poscar/best2
    python setup_vasp_jobs.py --poscar-dir /scratch/jcho5/.../poscar/best3

    # Dry-run: see what would be written without touching any files:
    python setup_vasp_jobs.py --poscar-dir /scratch/jcho5/.../poscar/best --dry-run

Prerequisites
-------------
Set VASP_PP_PATH to the root of your POTCAR library, e.g.:

    export VASP_PP_PATH=/projects/vasp/pps/PBE54

Or pass it explicitly:

    python setup_vasp_jobs.py --poscar-dir ... --pp-path /projects/vasp/pps/PBE54
"""

import argparse
import os
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# INCAR template
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
GGA = PE

! Ionic Relaxation
NSW    = 1000
IBRION = 2

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
# Slurm template  (Kestrel)
# ---------------------------------------------------------------------------
SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=26
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --account=ccpc
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err

export OMP_NUM_THREADS=8
export OMP_STACKSIZE=1G
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
ulimit -s unlimited

module load vasp/6.3.2_openMP+tpc

srun vasp_std
"""

# ---------------------------------------------------------------------------
# POTCAR element name mapping
# ---------------------------------------------------------------------------
POTCAR_MAP = {
    "Cu": ["Cu"],
    "C":  ["C"],
    "H":  ["H"],
    "O":  ["O"],
    "N":  ["N"],
    "S":  ["S"],
    "Pt": ["Pt"],
    "Pd": ["Pd"],
    "Ni": ["Ni"],
    "Ag": ["Ag"],
    "Au": ["Au"],
    "Fe": ["Fe"],
    "Co": ["Co"],
    "Zn": ["Zn"],
    "Al": ["Al"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_species_from_poscar(poscar_path: Path) -> list:
    """Return the list of element symbols from line 6 of a VASP5 POSCAR."""
    lines = poscar_path.read_text().splitlines()
    if len(lines) < 6:
        raise ValueError(f"POSCAR too short: {poscar_path}")
    species = lines[5].split()
    if not all(re.match(r"^[A-Z][a-z]?$", s) for s in species):
        raise ValueError(
            f"Could not parse species from POSCAR line 6: '{lines[5]}'\n"
            "Check that vasp5=True was used when writing the POSCAR."
        )
    return species


def find_potcar(element: str, pp_root: Path) -> Path:
    candidates = POTCAR_MAP.get(element, [element])
    tried = []
    for folder in candidates:
        p = pp_root / folder / "POTCAR"
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        f"POTCAR for '{element}' not found under {pp_root}.\n"
        f"Tried: {tried}\n"
        "Set VASP_PP_PATH correctly or add the element to POTCAR_MAP."
    )


def build_potcar(species: list, pp_root: Path, out_path: Path,
                 dry_run: bool = False) -> bool:
    parts = []
    for el in species:
        try:
            parts.append(find_potcar(el, pp_root))
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}")
            return False
    if not dry_run:
        with out_path.open("w") as fout:
            for p in parts:
                fout.write(p.read_text())
    return True


def setup_job_dir(job_dir: Path, system_name: str,
                  pp_root: Path, dry_run: bool = False) -> dict:
    """Write INCAR, KPOINTS, POTCAR, slm.vasp.kestrel into job_dir."""
    poscar = job_dir / "POSCAR"
    if not poscar.exists():
        return {"status": "skipped", "reason": "no POSCAR"}

    status = {"dir": str(job_dir), "status": "ok", "warnings": []}

    try:
        species = read_species_from_poscar(poscar)
    except ValueError as e:
        return {"status": "error", "reason": str(e)}

    status["species"] = species

    if not dry_run:
        (job_dir / "INCAR").write_text(INCAR_TEMPLATE.format(system=system_name))
        (job_dir / "KPOINTS").write_text(KPOINTS_TEMPLATE)

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
        cat_cmd = " ".join(
            f"$VASP_PP_PATH/{POTCAR_MAP.get(el, [el])[0]}/POTCAR"
            for el in species
        )
        note = (
            "# POTCAR not generated automatically.\n"
            "# Set VASP_PP_PATH and run:\n"
            f"cat {cat_cmd} > {potcar_path}\n"
        )
        if not dry_run:
            (job_dir / "make_potcar.sh").write_text(note)
        status["warnings"].append(
            "VASP_PP_PATH not set. Written make_potcar.sh instead — "
            "run it to build POTCAR."
        )
        status["status"] = "partial"

    job_name = system_name.replace(" ", "_").replace("+", "")[:40]
    slurm_path = job_dir / "slm.vasp.kestrel"
    if not dry_run:
        slurm_path.write_text(SLURM_TEMPLATE.format(job_name=job_name))
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
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--poscar-dir",
        required=True,
        metavar="DIR",
        help=(
            "Directory whose immediate subdirectories each contain a POSCAR. "
            "Point this directly at the folder you want to process, e.g.:\n"
            "  poscar/best    poscar/best2    poscar/best3    poscar/best4"
        ),
    )
    parser.add_argument(
        "--pp-path", default=None,
        help="Path to VASP PBE PAW pseudopotential library root "
             "(overrides VASP_PP_PATH env var)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    poscar_dir = Path(args.poscar_dir)
    if not poscar_dir.exists():
        print(f"ERROR: directory not found: {poscar_dir}")
        raise SystemExit(1)

    print(f"Processing POSCAR subdirectories in: {poscar_dir}")

    # ---- Resolve POTCAR library path -----------------------------------------
    pp_path_str = args.pp_path or os.environ.get("VASP_PP_PATH", "")
    pp_root = Path(pp_path_str) if pp_path_str else None
    if pp_root is None:
        print("WARNING: VASP_PP_PATH is not set and --pp-path not given.")
        print("         POTCAR will NOT be built automatically.")
        print("         A make_potcar.sh helper will be written instead.")
        print()
    else:
        print(f"Using POTCAR library: {pp_root}")

    # ---- Collect job directories (all immediate subdirs with a POSCAR) -------
    job_dirs = [
        d for d in sorted(poscar_dir.iterdir())
        if d.is_dir() and (d / "POSCAR").exists()
    ]

    if not job_dirs:
        print(f"No subdirectories containing a POSCAR found under {poscar_dir}.")
        print("Make sure --poscar-dir points at the folder with the POSCAR subdirs,")
        print("e.g. poscar/best  or  poscar/best2")
        raise SystemExit(0)

    print(f"Setting up {len(job_dirs)} VASP job director(y/ies):")
    print()

    all_ok = True
    for job_dir in job_dirs:
        system_name = job_dir.name
        action = "[DRY-RUN]" if args.dry_run else "writing"
        print(f"  {action}: {job_dir}/")

        result = setup_job_dir(
            job_dir, system_name,
            pp_root=pp_root,
            dry_run=args.dry_run,
        )

        print(f"    species:  {' '.join(result.get('species', []))}")

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
    print(f"     python setup_vasp_jobs.py --poscar-dir {poscar_dir}")
    print()
    print("2. Submit each job:")
    print()
    print(f"     for d in {poscar_dir}/*/; do")
    print("       (cd \"$d\" && sbatch slm.vasp.kestrel)")
    print("     done")
    print()
    print("3. After VASP finishes, compute E_ads from OUTCAR:")
    print()
    print("     grep 'free  energy' OUTCAR | tail -1")
    print()
    if not all_ok:
        print("Some warnings were raised — see above.")


if __name__ == "__main__":
    main()
