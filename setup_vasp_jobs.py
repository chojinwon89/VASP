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

A functional-named subfolder is created inside each system directory so that
multiple functionals can coexist side-by-side:

    poscar/best/Cu001_CO2/PBE/          <- --functional pbe
    poscar/best/Cu001_CO2/PBE_D3/       <- --functional pbe-d3
    poscar/best/Cu001_CO2/r2scan/       <- --functional r2scan
    poscar/best/Cu001_CO2/beef_vdw/     <- --functional beef-vdw

Usage
-----
    python setup_vasp_jobs.py --poscar-dir /scratch/jcho5/.../poscar/best \\
                               --functional pbe

    python setup_vasp_jobs.py --poscar-dir /scratch/jcho5/.../poscar/best \\
                               --functional pbe-d3

    python setup_vasp_jobs.py --poscar-dir /scratch/jcho5/.../poscar/best \\
                               --functional r2scan

    python setup_vasp_jobs.py --poscar-dir /scratch/jcho5/.../poscar/best \\
                               --functional beef-vdw

    # Dry-run: see what would be written without touching any files:
    python setup_vasp_jobs.py --poscar-dir /scratch/jcho5/.../poscar/best \\
                               --functional pbe --dry-run

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
# Supported functionals  ->  (subfolder_name, INCAR_xc_block)
# ---------------------------------------------------------------------------
FUNCTIONAL_CONFIGS = {
    "pbe": {
        "subfolder": "PBE",
        "xc_block": """\
! Exchange-correlation
GGA = PE
""",
    },
    "pbe-d3": {
        "subfolder": "PBE_D3",
        "xc_block": """\
! Exchange-correlation
GGA    = PE
IVDW   = 11
VDW_S6 = 1.0
VDW_SR = 1.217
""",
    },
    "r2scan": {
        "subfolder": "r2scan",
        "xc_block": """\
! Exchange-correlation
METAGGA = R2SCAN
LASPH   = .TRUE.
""",
    },
    "beef-vdw": {
        "subfolder": "beef_vdw",
        "xc_block": """\
! Exchange-correlation
GGA  = BF
LUSE_VDW  = .TRUE.
AGGAC     = 0.0000
""",
    },
}

# ---------------------------------------------------------------------------
# INCAR template  (xc_block is injected per-functional)
# ---------------------------------------------------------------------------
INCAR_BASE = """\
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

{xc_block}
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
                  pp_root: Path, xc_block: str,
                  dry_run: bool = False) -> dict:
    """Write INCAR, KPOINTS, POTCAR, slm.vasp.kestrel into job_dir.

    job_dir is the functional subfolder, e.g. poscar/best/Cu001_CO2/PBE.
    The POSCAR is read from the parent system directory.
    """
    poscar = job_dir.parent / "POSCAR"
    if not poscar.exists():
        return {"status": "skipped", "reason": "no POSCAR"}

    status = {"dir": str(job_dir), "status": "ok", "warnings": []}

    try:
        species = read_species_from_poscar(poscar)
    except ValueError as e:
        return {"status": "error", "reason": str(e)}

    status["species"] = species

    if not dry_run:
        job_dir.mkdir(parents=True, exist_ok=True)
        # Copy POSCAR into the functional subfolder
        (job_dir / "POSCAR").write_bytes(poscar.read_bytes())
        incar_text = INCAR_BASE.format(system=system_name, xc_block=xc_block.rstrip())
        (job_dir / "INCAR").write_text(incar_text)
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
            "with INCAR, KPOINTS, POTCAR, and slm.vasp.kestrel.\n\n"
            "A functional-named subfolder is created inside each system directory:\n"
            "  poscar/best/Cu001_CO2/PBE/\n"
            "  poscar/best/Cu001_CO2/PBE_D3/\n"
            "  poscar/best/Cu001_CO2/r2scan/\n"
            "  poscar/best/Cu001_CO2/beef_vdw/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--poscar-dir",
        required=True,
        metavar="DIR",
        help=(
            "Directory whose immediate subdirectories each contain a POSCAR. "
            "E.g.: poscar/best  poscar/best2  poscar/best3"
        ),
    )
    parser.add_argument(
        "--functional",
        required=True,
        choices=list(FUNCTIONAL_CONFIGS.keys()),
        metavar="FUNCTIONAL",
        help=(
            "Exchange-correlation functional to use. "
            "Choices: " + ", ".join(FUNCTIONAL_CONFIGS.keys()) + ". "
            "Controls the subfolder name and INCAR XC settings."
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

    # ---- Resolve functional config -------------------------------------------
    func_cfg   = FUNCTIONAL_CONFIGS[args.functional]
    subfolder  = func_cfg["subfolder"]
    xc_block   = func_cfg["xc_block"]

    print(f"Processing POSCAR subdirectories in: {poscar_dir}")
    print(f"Functional : {args.functional}  ->  subfolder name: {subfolder}")
    print()

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

    # ---- Collect system directories (immediate subdirs with a POSCAR) --------
    system_dirs = [
        d for d in sorted(poscar_dir.iterdir())
        if d.is_dir() and (d / "POSCAR").exists()
    ]

    if not system_dirs:
        print(f"No subdirectories containing a POSCAR found under {poscar_dir}.")
        print("Make sure --poscar-dir points at the folder with the POSCAR subdirs,")
        print("e.g. poscar/best  or  poscar/best2")
        raise SystemExit(0)

    print(f"Setting up {len(system_dirs)} system(s) with functional '{args.functional}':")
    print()

    all_ok = True
    for sys_dir in system_dirs:
        system_name = sys_dir.name
        job_dir     = sys_dir / subfolder           # e.g. Cu001_CO2/PBE
        action = "[DRY-RUN]" if args.dry_run else "writing"
        print(f"  {action}: {job_dir}/")

        result = setup_job_dir(
            job_dir, system_name,
            pp_root=pp_root,
            xc_block=xc_block,
            dry_run=args.dry_run,
        )

        print(f"    species:  {' '.join(result.get('species', []))}")

        if not args.dry_run:
            files = ["POSCAR", "INCAR", "KPOINTS", "slm.vasp.kestrel"]
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
    print(f"     python setup_vasp_jobs.py --poscar-dir {poscar_dir} --functional {args.functional}")
    print()
    print("2. Submit each job:")
    print()
    print(f"     for d in {poscar_dir}/*/{subfolder}/; do")
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
