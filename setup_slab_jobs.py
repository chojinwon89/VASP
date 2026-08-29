#!/usr/bin/env python
"""
setup_slab_jobs.py
==================
Generate VASP input files (POSCAR, INCAR, KPOINTS, POTCAR, and a
cluster-specific Slurm script) for bare-slab reference energy calculations.

Use --cluster kestrel        -> writes slm.vasp.kestrel   (--account=ccpc)
Use --cluster perlmutter-cpu -> writes slm.vasp.perlmutter (-A m5281 -C cpu)

These slab energies (E_surf) are needed to compute the DFT adsorption energy:
    E_ads = E_total(slab+mol) - E_surf(slab) - E_mol(gas)

The slabs are built from ASE using the SAME per-facet geometry as
generate_surface_cifs.py (the slabs GOAD/MLIP and dft_jobs are built from),
so the clean-slab reference matches the adsorption-cell metal count:
    (111)/(0001) : (4, 4, 4) -> 64 atoms
    (110)        : (3, 2, 4) -> 24 atoms
    (100)        : (3, 3, 4) -> 36 atoms
15 Ang vacuum. Bottom N layers are frozen via Selective Dynamics,
matching the constraint used in batch_isopropanol.py.

Supported metals and facets (matches generate_surface_cifs.py):
  FCC : Cu, Pd, Pt, Ni, Ag, Au, Ir, Rh  ->  111, 110, 100
  BCC : Fe, Cr, Mo                        ->  110, 100, 111
  HCP : Ru, Co, Ti, Zn                    ->  0001

Output layout
-------------
    vasp_slab/
        Cu111/
            PBE/
                POSCAR  INCAR  KPOINTS  POTCAR  slm.vasp.<cluster>
            PBE_D3/
                ...
            r2scan/
                ...
            beef_vdw/
                ...
        Cu110/
            ...

Usage
-----
    python setup_slab_jobs.py --functional pbe
    python setup_slab_jobs.py --functional pbe-d3
    python setup_slab_jobs.py --functional r2scan
    python setup_slab_jobs.py --functional beef-vdw
    python setup_slab_jobs.py --functional pbe --surfaces Cu111 Cu110 Cu001
    python setup_slab_jobs.py --functional r2scan --out-dir /scratch/jcho5/slab_jobs
    python setup_slab_jobs.py --functional pbe --dry-run
    python setup_slab_jobs.py --functional pbe --force

Prerequisites
-------------
    conda activate goad
    export VASP_PP_PATH=/home/jcho5/project/paw64/potpaw_PBE_64

For beef-vdw jobs, this script also tries to copy:
    /projects/2dmgcat/vdw_kernel.bindat
"""

import argparse
import os
import shutil
from pathlib import Path

from ase.build import fcc111, fcc100, fcc110, bcc110, bcc100, bcc111, hcp0001
from ase import Atoms

# Default POTCAR / vdW-kernel paths, per cluster.
# Priority chain (highest first):
#   --pp-path / --vdw-kernel-path flag
#   > VASP_PP_PATH / VASP_VDW_KERNEL_PATH env var
#   > per-cluster default below (keyed by --cluster)
#   > generic fallback (DEFAULT_PP_PATH / DEFAULT_VDW_KERNEL_PATH)
DEFAULT_PP_PATH = "/projects/2dmgcat/paw64/potpaw_PBE_64"
DEFAULT_VDW_KERNEL_PATH = "/projects/2dmgcat/vdw_kernel.bindat"

CLUSTER_PP_PATH = {
    "kestrel":        "/projects/2dmgcat/paw64/potpaw_PBE_64",
    "perlmutter-cpu": "/pscratch/sd/j/jcho5/paw64/potpaw_PBE_64",
}
CLUSTER_VDW_KERNEL_PATH = {
    "kestrel":        "/projects/2dmgcat/vdw_kernel.bindat",
    "perlmutter-cpu": "/pscratch/sd/j/jcho5/vdw_kernel.bindat",
}


# ---------------------------------------------------------------------------
# Supported functionals  ->  (subfolder_name, INCAR_xc_block)
# (Matches setup_vasp_jobs.py conventions)
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

{xc_block}
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
# Slurm — Perlmutter template (NERSC, CPU nodes)
# ---------------------------------------------------------------------------
SLURM_TEMPLATE_PERLMUTTER_CPU = """\
#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -A m5281
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 12:00:00
#SBATCH -o {job_name}.out
#SBATCH -e {job_name}.err

# NERSC CPU VASP build (license-gated). If yours differs, run
# `module avail vasp` and edit this line.
module load vasp-tpc/6.4.2-cpu

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Full CPU node = 128 physical cores; pure MPI (OMP=1), no hyperthreading.
srun --cpu-bind=cores vasp_std
"""

# cluster name -> (slurm template, filename written into each job dir)
SLURM_TEMPLATES = {
    "kestrel":        (SLURM_TEMPLATE, "slm.vasp.kestrel"),
    "perlmutter-cpu": (SLURM_TEMPLATE_PERLMUTTER_CPU, "slm.vasp.perlmutter"),
}

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
    # Additional metals matching generate_surface_cifs.py
    "Ir": ["Ir_pv", "Ir"],
    "Rh": ["Rh_pv", "Rh"],
    "Cr": ["Cr_pv", "Cr"],
    "Mo": ["Mo_pv", "Mo"],
    "Ru": ["Ru_pv", "Ru"],
    "Ti": ["Ti_pv", "Ti"],
}

# ---------------------------------------------------------------------------
# Lattice constants — same values as generate_surface_cifs.py
# ---------------------------------------------------------------------------
_A_FCC = {
    "Cu": 3.615, "Pd": 3.890, "Pt": 3.924,
    "Ni": 3.524, "Au": 4.078, "Ag": 4.086,
    "Ir": 3.840, "Rh": 3.803,
}
_A_BCC = {
    "Fe": 2.870, "Cr": 2.885, "Mo": 3.147,
}
_A_HCP = {
    # (a, c)
    "Ru": (2.706, 4.282), "Co": (2.507, 4.069),
    "Ti": (2.951, 4.686), "Zn": (2.665, 4.947),
}

# ---------------------------------------------------------------------------
# Slab builder registry
# ---------------------------------------------------------------------------
SLAB_BUILDERS = {}

# Supercell sizes MUST match generate_surface_cifs.py (the slabs GOAD/MLIP and
# dft_jobs are built from) so the clean-slab reference has the same metal-atom
# count as the adsorption cell:
#   FCC/BCC (111), HCP (0001): (4, 4, 4) -> 64 atoms (orthogonal)
#   FCC/BCC (110):             (3, 2, 4) -> 24 atoms
#   FCC/BCC (100):             (3, 3, 4) -> 36 atoms
# --- FCC: 111 (orthogonal), 110, 100 ---
for _el, _a in _A_FCC.items():
    SLAB_BUILDERS[f"{_el}111"] = (fcc111, {"symbol": _el, "a": _a, "size": (4, 4, 4), "vacuum": 15.0, "orthogonal": True})
    SLAB_BUILDERS[f"{_el}110"] = (fcc110, {"symbol": _el, "a": _a, "size": (3, 2, 4), "vacuum": 15.0})
    SLAB_BUILDERS[f"{_el}100"] = (fcc100, {"symbol": _el, "a": _a, "size": (3, 3, 4), "vacuum": 15.0})

# Cu001 alias (backwards compatibility)
SLAB_BUILDERS["Cu001"] = (fcc100, {"symbol": "Cu", "a": _A_FCC["Cu"], "size": (3, 3, 4), "vacuum": 15.0})

# --- BCC: 110, 100, 111 (orthogonal) ---
for _el, _a in _A_BCC.items():
    SLAB_BUILDERS[f"{_el}110"] = (bcc110, {"symbol": _el, "a": _a, "size": (3, 2, 4), "vacuum": 15.0})
    SLAB_BUILDERS[f"{_el}100"] = (bcc100, {"symbol": _el, "a": _a, "size": (3, 3, 4), "vacuum": 15.0})
    SLAB_BUILDERS[f"{_el}111"] = (bcc111, {"symbol": _el, "a": _a, "size": (4, 4, 4), "vacuum": 15.0, "orthogonal": True})

# --- HCP: 0001 (orthogonal) ---
for _el, (_a, _c) in _A_HCP.items():
    SLAB_BUILDERS[f"{_el}0001"] = (hcp0001, {"symbol": _el, "a": _a, "c": _c, "size": (4, 4, 4), "vacuum": 15.0, "orthogonal": True})


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

def setup_slab_dir(surface_name, out_dir, pp_root, n_fixed, functional,
                   dry_run=False, force=False,
                   slurm_template=SLURM_TEMPLATE,
                   slurm_filename="slm.vasp.kestrel",
                   vdw_kernel_path=None):
    """Build slab and write all VASP input files into out_dir/surface_name/subfolder/."""

    func_cfg = FUNCTIONAL_CONFIGS[functional]
    subfolder = func_cfg["subfolder"]
    xc_block = func_cfg["xc_block"]

    job_dir = out_dir / surface_name / subfolder
    outcar_path = job_dir / "OUTCAR"
    if outcar_path.exists() and not force:
        return {
            "surface": surface_name,
            "status": "skipped",
            "reason": f"OUTCAR already exists in {job_dir}; skipping finished job (use --force to regenerate).",
            "warnings": [],
        }

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
        (job_dir / "INCAR").write_text(
            INCAR_TEMPLATE.format(system=surface_name, xc_block=xc_block)
        )

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
    slurm_path = job_dir / slurm_filename
    if not dry_run:
        slurm_path.write_text(slurm_template.format(job_name=surface_name[:40]))
        slurm_path.chmod(0o755)

    # vdw_kernel.bindat for beef-vdw
    if functional == "beef-vdw":
        kernel_path = Path(vdw_kernel_path) if vdw_kernel_path \
            else Path(DEFAULT_VDW_KERNEL_PATH)
        if kernel_path.exists():
            if not dry_run:
                shutil.copy2(kernel_path, job_dir / "vdw_kernel.bindat")
            status["vdw_kernel_written"] = True
        else:
            status["warnings"].append(
                f"vdw_kernel.bindat not found at: {kernel_path}. "
                "beef-vdw jobs will be missing this file; copy it manually."
            )
            status["status"] = "partial"

    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate VASP inputs for bare-slab reference energy calculations."
    )
    parser.add_argument(
        "--functional",
        required=True,
        choices=list(FUNCTIONAL_CONFIGS.keys()),
        metavar="FUNC",
        help=(
            "Exchange-correlation functional to use. "
            "Supported: pbe, pbe-d3, r2scan, beef-vdw. "
            "Creates a functional-specific subfolder under each surface, e.g. "
            "vasp_slab/Cu111/PBE/, vasp_slab/Cu111/PBE_D3/, etc."
        ),
    )
    parser.add_argument(
        "--surfaces", nargs="+",
        default=[
            # FCC
            "Cu111", "Cu110", "Cu001",
            "Pt111", "Pt110", "Pt100",
            "Pd111", "Pd110", "Pd100",
            "Ni111", "Ni110", "Ni100",
            "Ag111", "Ag110", "Ag100",
            "Au111", "Au110", "Au100",
            "Ir111", "Ir110", "Ir100",
            "Rh111", "Rh110", "Rh100",
            # BCC
            "Fe110", "Fe100", "Fe111",
            "Cr110", "Cr100", "Cr111",
            "Mo110", "Mo100", "Mo111",
            # HCP
            "Ru0001", "Co0001", "Ti0001", "Zn0001",
        ],
        help="Surface names to set up (default: all surfaces matching generate_surface_cifs.py)"
    )
    parser.add_argument(
        "--out-dir", default="vasp_slab",
        help="Output root directory (default: ./vasp_slab)"
    )
    parser.add_argument(
        "--cluster", default="kestrel",
        choices=sorted(SLURM_TEMPLATES.keys()),
        help=(
            "Which cluster's Slurm submit script to write into each job dir. "
            "'kestrel' (default) writes slm.vasp.kestrel (--account=ccpc); "
            "'perlmutter-cpu' writes slm.vasp.perlmutter (-A m5281 -C cpu "
            "-q regular). Also selects the per-cluster POTCAR / vdW-kernel "
            "default paths."
        ),
    )
    parser.add_argument(
        "--pp-path", default=None,
        help=(
            "Path to VASP PBE PAW library. Priority: --pp-path > VASP_PP_PATH "
            "env var > per-cluster default > generic fallback."
        ),
    )
    parser.add_argument(
        "--vdw-kernel-path", default=None,
        help=(
            "Path to vdw_kernel.bindat for beef-vdw jobs. Priority: "
            "--vdw-kernel-path > VASP_VDW_KERNEL_PATH env var > per-cluster "
            "default > generic fallback."
        ),
    )
    parser.add_argument(
        "--n-fixed", type=int, default=2,
        help="Number of bottom layers to freeze in POSCAR (default: 2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate job files even when OUTCAR already exists (default: skip finished jobs)."
    )
    args = parser.parse_args()

    functional = args.functional
    func_cfg = FUNCTIONAL_CONFIGS[functional]
    subfolder = func_cfg["subfolder"]

    # Resolve n_fixed as a plain local variable — no global needed
    n_fixed = args.n_fixed
    out_dir = Path(args.out_dir)

    # Resolve which Slurm script to write for this cluster.
    slurm_template, slurm_filename = SLURM_TEMPLATES[args.cluster]

    # Resolve POTCAR library: --pp-path > VASP_PP_PATH env > per-cluster
    # default > generic fallback.
    cluster_pp_default = CLUSTER_PP_PATH.get(args.cluster, DEFAULT_PP_PATH)
    pp_path_str = (args.pp_path
                   or os.environ.get("VASP_PP_PATH", "")
                   or cluster_pp_default)
    pp_root = Path(pp_path_str) if pp_path_str else None

    # Resolve vdW kernel: --vdw-kernel-path > VASP_VDW_KERNEL_PATH env >
    # per-cluster default > generic fallback.
    cluster_vdw_default = CLUSTER_VDW_KERNEL_PATH.get(args.cluster,
                                                      DEFAULT_VDW_KERNEL_PATH)
    vdw_kernel_path = (args.vdw_kernel_path
                       or os.environ.get("VASP_VDW_KERNEL_PATH", "")
                       or cluster_vdw_default)

    if pp_root is not None and not pp_root.exists():
        print(f"WARNING: POTCAR library not found at: {pp_root}")
        print("         POTCAR will NOT be built automatically; "
              "make_potcar.sh written instead.")
        print("         Override with --pp-path or VASP_PP_PATH env var.")
        print()
        pp_root = None
    elif pp_root is None:
        print("WARNING: VASP_PP_PATH not set — POTCAR will not be built automatically.")
        print("         make_potcar.sh will be written in each job directory instead.")
        print()
    else:
        source = ("--pp-path" if args.pp_path
                  else "VASP_PP_PATH" if os.environ.get("VASP_PP_PATH")
                  else f"{args.cluster} default")
        print(f"Using POTCAR library ({source}): {pp_root}")
        print()

    print(f"Functional:        {functional} -> subfolder '{subfolder}'")
    print(f"Cluster:           {args.cluster}  ->  submit script: {slurm_filename}")
    print(f"Surfaces:          {args.surfaces}")
    print(f"Output directory:  {out_dir}/")
    print(f"Fixed bottom layers: {n_fixed}")
    print(f"Force overwrite:   {args.force}")
    print()

    all_ok = True
    for surface in args.surfaces:
        action = "[DRY-RUN]" if args.dry_run else "writing"
        print(f"  {action}: {out_dir / surface / subfolder}/")

        result = setup_slab_dir(
            surface, out_dir,
            pp_root=pp_root,
            n_fixed=n_fixed,
            functional=functional,
            dry_run=args.dry_run,
            force=args.force,
            slurm_template=slurm_template,
            slurm_filename=slurm_filename,
            vdw_kernel_path=vdw_kernel_path,
        )

        if result["status"] == "skipped":
            print(f"    SKIPPED: {result['reason']}")
            print()
            continue

        if result["status"] == "error":
            print(f"    ERROR: {result['reason']}")
            all_ok = False
            continue

        print(f"    atoms:    {result['n_atoms']}")
        print(f"    layers:   {result['n_layers']}  (bottom {n_fixed} fixed)")
        print(f"    species:  {' '.join(result['species'])}")

        if not args.dry_run:
            files = ["POSCAR (SD)", "INCAR", "KPOINTS", slurm_filename]
            if result["status"] == "ok":
                files.insert(3, "POTCAR")
            if result.get("vdw_kernel_written"):
                files.append("vdw_kernel.bindat")
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
    print("Finished job directories containing OUTCAR are skipped by default.")
    print("Use --force to regenerate those directories when needed.")
    print()
    if not pp_root:
        print("1. Build POTCARs:")
        print()
        print("     export VASP_PP_PATH=/home/jcho5/project/paw64/potpaw_PBE_64")
        print(f"     python setup_slab_jobs.py --functional {functional}    # re-run to build POTCARs")
        print()
        step = 2
    else:
        step = 1

    print(f"{step}. Submit slab jobs:")
    print()
    for s in args.surfaces:
        print(f"     cd {out_dir}/{s}/{subfolder} && sbatch {slurm_filename} && cd -")
    print()
    step += 1
    print(f"{step}. After jobs finish, extract E_surf from OUTCAR:")
    print()
    for s in args.surfaces:
        print(f"     grep 'free  energy' {out_dir}/{s}/{subfolder}/OUTCAR | tail -1")
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
