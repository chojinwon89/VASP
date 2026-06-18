#!/usr/bin/env python
"""
setup_slab_jobs.py
==================
Generate VASP input files (POSCAR, INCAR, KPOINTS, POTCAR, slm.vasp.kestrel)
for bare-slab reference energy calculations.

These slab energies (E_surf) are needed to compute the DFT adsorption energy:
    E_ads = E_total(slab+mol) - E_surf(slab) - E_mol(gas)

The slabs are built from ASE using the same geometry as GOAD
(4x4x4, 15 Ang vacuum). Bottom N layers are frozen via Selective Dynamics,
matching the constraint used in batch_isopropanol.py.

Output layout
-------------
    vasp_slab/
        Cu111/
            POSCAR  INCAR  KPOINTS  POTCAR  slm.vasp.kestrel
        Cu110/
            ...
        Cu001/
            ...

Usage
-----
    python setup_slab_jobs.py
    python setup_slab_jobs.py --surfaces Cu111 Cu110 Cu001
    python setup_slab_jobs.py --out-dir /scratch/jcho5/slab_jobs
    python setup_slab_jobs.py --dry-run

Prerequisites
-------------
    conda activate goad
    export VASP_PP_PATH=/home/jcho5/project/paw64/potpaw_PBE_64
"""

import argparse
import os
from pathlib import Path

from ase.build import fcc111, fcc100, fcc110
from ase import Atoms


# ---------------------------------------------------------------------------
# INCAR — same settings as the adsorbed-system jobs
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
# KPOINTS — Monkhorst-Pack 2x2x1
# ---------------------------------------------------------------------------
KPOINTS_TEMPLATE = """\
Monkhorst-Pack 2x2x1
 0
Monkhorst-Pack
  2  2  1
  0  0  0
"""

# ---------------------------------------------------------------------------
# Slurm — Kestrel template
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
# POTCAR element map
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
# Slab builder registry
# ---------------------------------------------------------------------------
SLAB_BUILDERS = {
    "Cu111": (fcc111, {"symbol": "Cu", "size": (4, 4, 4), "vacuum": 15.0, "orthogonal": True}),
    "Cu110": (fcc110, {"symbol": "Cu", "size": (4, 4, 4), "vacuum": 15.0}),
    "Cu100": (fcc100, {"symbol": "Cu", "size": (4, 4, 4), "vacuum": 15.0}),
    "Cu001": (fcc100, {"symbol": "Cu", "size": (4, 4, 4), "vacuum": 15.0}),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_slab(surface_name):
    """Build and return the ASE slab for a given surface name."""
    if surface_name not in SLAB_BUILDERS:
        raise ValueError(
            f"Unknown surface '{surface_name}'. "
            f"Available: {list(SLAB_BUILDERS.keys())}"
        )
    func, kwargs = SLAB_BUILDERS[surface_name]
    return func(**kwargs)


def get_layer_z_values(atoms, tol=0.5):
    """
    Return sorted list of unique Z coordinates representing atomic layers,
    from bottom (lowest Z) to top (highest Z).
    """
    z_coords = sorted(atoms.get_positions()[:, 2])
    layers = []
    current = [z_coords[0]]
    for z in z_coords[1:]:
        if z - current[-1] < tol:
            current.append(z)
        else:
            layers.append(sum(current) / len(current))
            current = [z]
    layers.append(sum(current) / len(current))
    return layers


def make_selective_dynamics_poscar(atoms, n_fixed_bottom, comment=""):
    """
    Return a VASP5 POSCAR string with Selective Dynamics.
    Bottom n_fixed_bottom layers are frozen (F F F); top layers are free (T T T).
    """
    # Sort atoms by species (Cu only for bare slab, but keep general)
    symbols = atoms.get_chemical_symbols()
    seen = []
    for s in symbols:
        if s not in seen:
            seen.append(s)
    sorted_idx = []
    for el in seen:
        sorted_idx.extend(i for i, s in enumerate(symbols) if s == el)
    atoms = atoms[sorted_idx]

    layer_zs = get_layer_z_values(atoms)
    fixed_z_cutoff = layer_zs[n_fixed_bottom - 1] + 0.3

    positions = atoms.get_positions()
    cell = atoms.get_cell()

    lines = []
    lines.append(comment or atoms.get_chemical_formula() + " slab")
    lines.append("   1.00000000000000")

    for vec in cell:
        lines.append(f"  {vec[0]:20.16f}  {vec[1]:20.16f}  {vec[2]:20.16f}")

    # Species and counts
    species_list = []
    counts = []
    for el in seen:
        n = sum(1 for s in atoms.get_chemical_symbols() if s == el)
        species_list.append(el)
        counts.append(str(n))
    lines.append("  " + "  ".join(species_list))
    lines.append("  " + "  ".join(counts))

    lines.append("Selective dynamics")
    lines.append("Cartesian")

    for pos in positions:
        flag = "F  F  F" if pos[2] <= fixed_z_cutoff else "T  T  T"
        lines.append(
            f"  {pos[0]:20.16f}  {pos[1]:20.16f}  {pos[2]:20.16f}  {flag}"
        )

    return "\n".join(lines) + "\n"


def find_potcar(element, pp_root):
    candidates = POTCAR_MAP.get(element, [element])
    tried = []
    for folder in candidates:
        p = pp_root / folder / "POTCAR"
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        f"POTCAR for '{element}' not found under {pp_root}.\nTried: {tried}"
    )


def build_potcar(species, pp_root, out_path, dry_run=False):
    parts = []
    for el in species:
        try:
            parts.append(find_potcar(el, pp_root))
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}")
            return False
    if not dry_run:
        with out_path.open("w") as f:
            for p in parts:
                f.write(p.read_text())
    return True


# ---------------------------------------------------------------------------
# Per-surface setup
# ---------------------------------------------------------------------------

def setup_slab_dir(surface_name, out_dir, pp_root, n_fixed, dry_run=False):
    """Build slab and write all VASP input files into out_dir/surface_name/."""

    job_dir = out_dir / surface_name
    if not dry_run:
        job_dir.mkdir(parents=True, exist_ok=True)

    try:
        slab = build_slab(surface_name)
    except ValueError as e:
        return {"status": "error", "reason": str(e)}

    n_atoms = len(slab)
    species = list(dict.fromkeys(slab.get_chemical_symbols()))
    layer_zs = get_layer_z_values(slab)
    n_layers = len(layer_zs)

    status = {
        "surface":  surface_name,
        "n_atoms":  n_atoms,
        "n_layers": n_layers,
        "species":  species,
        "status":   "ok",
        "warnings": [],
    }

    # POSCAR
    comment = (
        f"{surface_name} slab | {n_atoms} atoms | "
        f"{n_layers} layers | bottom {n_fixed} fixed"
    )
    poscar_text = make_selective_dynamics_poscar(slab, n_fixed, comment)
    if not dry_run:
        (job_dir / "POSCAR").write_text(poscar_text)

    # INCAR
    if not dry_run:
        (job_dir / "INCAR").write_text(INCAR_TEMPLATE.format(system=surface_name))

    # KPOINTS
    if not dry_run:
        (job_dir / "KPOINTS").write_text(KPOINTS_TEMPLATE)

    # POTCAR
    potcar_path = job_dir / "POTCAR"
    if pp_root is not None:
        ok = build_potcar(species, pp_root, potcar_path, dry_run=dry_run)
        if not ok:
            status["warnings"].append("POTCAR not written — element POTCARs not found.")
            status["status"] = "partial"
    else:
        cat_cmd = " ".join(
            f"$VASP_PP_PATH/{POTCAR_MAP.get(el, [el])[0]}/POTCAR"
            for el in species
        )
        helper = (
            f"# Build POTCAR for {surface_name}\n"
            f"# Set VASP_PP_PATH first, then run:\n"
            f"cat {cat_cmd} > POTCAR\n"
        )
        if not dry_run:
            (job_dir / "make_potcar.sh").write_text(helper)
        status["warnings"].append(
            "VASP_PP_PATH not set — written make_potcar.sh instead."
        )
        status["status"] = "partial"

    # Slurm
    slurm_path = job_dir / "slm.vasp.kestrel"
    if not dry_run:
        slurm_path.write_text(SLURM_TEMPLATE.format(job_name=surface_name[:40]))
        slurm_path.chmod(0o755)

    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate VASP inputs for bare-slab reference energy calculations."
    )
    parser.add_argument(
        "--surfaces", nargs="+",
        default=["Cu111", "Cu110", "Cu001"],
        help="Surface names to set up (default: Cu111 Cu110 Cu001)"
    )
    parser.add_argument(
        "--out-dir", default="vasp_slab",
        help="Output root directory (default: ./vasp_slab)"
    )
    parser.add_argument(
        "--pp-path", default=None,
        help="Path to VASP PBE PAW library (overrides VASP_PP_PATH env var)"
    )
    parser.add_argument(
        "--n-fixed", type=int, default=2,
        help="Number of bottom layers to freeze in POSCAR (default: 2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files"
    )
    args = parser.parse_args()

    # Resolve n_fixed as a plain local variable — no global needed
    n_fixed = args.n_fixed
    out_dir = Path(args.out_dir)

    # Resolve POTCAR library
    pp_path_str = args.pp_path or os.environ.get("VASP_PP_PATH", "")
    pp_root = Path(pp_path_str) if pp_path_str else None

    if pp_root is None:
        print("WARNING: VASP_PP_PATH not set — POTCAR will not be built automatically.")
        print("         make_potcar.sh will be written in each job directory instead.")
        print()
    else:
        print(f"Using POTCAR library: {pp_root}")
        print()

    print(f"Surfaces:          {args.surfaces}")
    print(f"Output directory:  {out_dir}/")
    print(f"Fixed bottom layers: {n_fixed}")
    print()

    all_ok = True
    for surface in args.surfaces:
        action = "[DRY-RUN]" if args.dry_run else "writing"
        print(f"  {action}: {out_dir / surface}/")

        result = setup_slab_dir(
            surface, out_dir,
            pp_root=pp_root,
            n_fixed=n_fixed,
            dry_run=args.dry_run,
        )

        if result["status"] == "error":
            print(f"    ERROR: {result['reason']}")
            all_ok = False
            continue

        print(f"    atoms:    {result['n_atoms']}")
        print(f"    layers:   {result['n_layers']}  (bottom {n_fixed} fixed)")
        print(f"    species:  {' '.join(result['species'])}")

        if not args.dry_run:
            files = ["POSCAR (SD)", "INCAR", "KPOINTS", "slm.vasp.kestrel"]
            if result["status"] == "ok":
                files.insert(3, "POTCAR")
            print(f"    written:  {', '.join(files)}")

        for w in result.get("warnings", []):
            print(f"    WARNING:  {w}")
            all_ok = False

        print()

    # Summary
    print("=" * 65)
    print("NEXT STEPS")
    print("=" * 65)
    print()
    if not pp_root:
        print("1. Build POTCARs:")
        print()
        print("     export VASP_PP_PATH=/home/jcho5/project/paw64/potpaw_PBE_64")
        print("     python setup_slab_jobs.py    # re-run to build POTCARs")
        print()
        step = 2
    else:
        step = 1

    print(f"{step}. Submit slab jobs:")
    print()
    for s in args.surfaces:
        print(f"     cd {out_dir}/{s} && sbatch slm.vasp.kestrel && cd -")
    print()
    step += 1
    print(f"{step}. After jobs finish, extract E_surf from OUTCAR:")
    print()
    for s in args.surfaces:
        print(f"     grep 'free  energy' {out_dir}/{s}/OUTCAR | tail -1")
    print()
    step += 1
    print(f"{step}. Compute DFT adsorption energy:")
    print()
    print("     E_ads = E_total(slab+mol) - E_surf(slab) - E_mol(gas)")
    print()
    if not all_ok:
        print("Some warnings were raised — see above.")


if __name__ == "__main__":
    main()
