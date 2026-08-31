#!/usr/bin/env python
"""
setup_slab_from_jobs.py
=======================
Generate clean-slab reference VASP jobs whose metal-atom count EXACTLY matches
the adsorption jobs under poscar/best/, by stripping the adsorbate atoms out of
each adsorption job's own POSCAR/CONTCAR.

Why: the adsorption jobs do not follow a fixed per-facet supercell size
(e.g. Au100=64, Ag100=36, Cu111=64, Ag110=24). A single ASE-built reference
per facet therefore never matches. Deriving the reference directly from the
job structure guarantees the same cell, facet and metal count, so
    E_ads = E(slab+mol) - E(slab) - E(mol)
is well defined.

For each unique <metal><facet> surface found among the job dirs, this picks one
representative job, removes every non-metal atom, re-freezes the bottom N layers
with Selective Dynamics, and writes a relaxation job under:
    <out-dir>/<surface>/<FUNC>/POSCAR INCAR KPOINTS POTCAR slm.vasp.<cluster>

Reuses the INCAR/KPOINTS/POTCAR/Slurm templates and writer helpers from
setup_slab_jobs.py so the reference settings match the adsorption jobs.

Usage
-----
    python setup_slab_from_jobs.py --functional pbe --best-dir poscar/best
    python setup_slab_from_jobs.py --functional pbe --surfaces Au110 Cu001
    python setup_slab_from_jobs.py --functional r2scan --dry-run
    python setup_slab_from_jobs.py --functional pbe --force
"""

import argparse
import os
import re
import shutil
from pathlib import Path

from ase.io import read

import setup_slab_jobs as ssj

KNOWN_METALS = sorted(ssj.POTCAR_MAP.keys(), key=len, reverse=True)
# Only elements that are actually slab metals (exclude adsorbate elements).
_ADSORBATE_ELEMENTS = {"C", "H", "O", "N", "S"}
METAL_ELEMENTS = {m for m in ssj.POTCAR_MAP if m not in _ADSORBATE_ELEMENTS}

_SURFACE_RE = re.compile(r"^([A-Z][a-z]?)(0001|100|110|111|001)")


def parse_surface(dir_name: str):
    """Return the <metal><facet> surface token from a job dir name, or None."""
    m = _SURFACE_RE.match(dir_name)
    if not m:
        return None
    metal, facet = m.group(1), m.group(2)
    if metal not in METAL_ELEMENTS:
        return None
    return f"{metal}{facet}", metal


def find_job_structure(job_dir: Path):
    """Return an ASE Atoms for the job slab+mol, preferring CONTCAR then POSCAR."""
    for name in ("CONTCAR", "POSCAR"):
        for cand in (job_dir / name, *job_dir.glob(f"**/{name}")):
            if cand.is_file():
                try:
                    return read(cand), cand
                except Exception:
                    continue
    return None, None


def collect_representatives(best_dir: Path, wanted_surfaces=None):
    """Map surface -> (Atoms slab-only, metal, source_path) using one job each.

    Picks, per surface, the job with the most metal atoms (the representative
    full slab), so partial/broken cells don't win.
    """
    reps = {}
    for job_dir in sorted(best_dir.glob("**/")):
        parsed = parse_surface(job_dir.name)
        if parsed is None:
            continue
        surface, metal = parsed
        if wanted_surfaces and surface not in wanted_surfaces:
            continue

        atoms, src = find_job_structure(job_dir)
        if atoms is None:
            continue

        # Keep only the slab metal atoms.
        keep = [i for i, s in enumerate(atoms.get_chemical_symbols())
                if s == metal]
        if not keep:
            continue
        slab = atoms[keep]
        n_metal = len(slab)

        prev = reps.get(surface)
        if prev is None or n_metal > prev[3]:
            reps[surface] = (slab, metal, src, n_metal)
    return reps


def main():
    ap = argparse.ArgumentParser(
        description=("Generate clean-slab references derived from the actual "
                     "adsorption-job POSCARs so metal counts match exactly."))
    ap.add_argument("--functional", required=True,
                    choices=list(ssj.FUNCTIONAL_CONFIGS.keys()))
    ap.add_argument("--best-dir", default="poscar/best",
                    help="Root containing adsorption job dirs (default: poscar/best)")
    ap.add_argument("--surfaces", nargs="+", default=None,
                    help="Restrict to these surfaces (e.g. Au110 Cu001).")
    ap.add_argument("--out-dir", default="vasp_slab")
    ap.add_argument("--cluster", default="kestrel",
                    choices=sorted(ssj.SLURM_TEMPLATES.keys()))
    ap.add_argument("--pp-path", default=None)
    ap.add_argument("--vdw-kernel-path", default=None)
    ap.add_argument("--n-fixed", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    functional = args.functional
    func_cfg = ssj.FUNCTIONAL_CONFIGS[functional]
    subfolder = func_cfg["subfolder"]
    xc_block = func_cfg["xc_block"]

    best_dir = Path(args.best_dir)
    out_dir = Path(args.out_dir)
    if not best_dir.exists():
        raise SystemExit(f"best-dir not found: {best_dir}")

    slurm_template, slurm_filename = ssj.SLURM_TEMPLATES[args.cluster]

    cluster_pp_default = ssj.CLUSTER_PP_PATH.get(args.cluster, ssj.DEFAULT_PP_PATH)
    pp_path_str = (args.pp_path or os.environ.get("VASP_PP_PATH", "")
                   or cluster_pp_default)
    pp_root = Path(pp_path_str) if pp_path_str else None

    wanted = set(args.surfaces) if args.surfaces else None
    reps = collect_representatives(best_dir, wanted)
    if not reps:
        raise SystemExit(f"No matching job structures found under {best_dir}")

    n_ok = n_skip = n_part = 0
    for surface in sorted(reps):
        slab, metal, src, n_metal = reps[surface]
        job_dir = out_dir / surface / subfolder
        outcar = job_dir / "OUTCAR"
        if outcar.exists() and not args.force:
            print(f"skip  {surface:8s} ({n_metal} {metal}) OUTCAR exists")
            n_skip += 1
            continue

        print(f"write {surface:8s} ({n_metal} {metal}) from {src}")
        if args.dry_run:
            continue

        job_dir.mkdir(parents=True, exist_ok=True)

        comment = (f"{surface} slab | {n_metal} {metal} | derived from job | "
                   f"bottom {args.n_fixed} fixed")
        poscar_text = ssj.make_selective_dynamics_poscar(
            slab, args.n_fixed, comment)
        (job_dir / "POSCAR").write_text(poscar_text)
        (job_dir / "INCAR").write_text(
            ssj.INCAR_TEMPLATE.format(system=surface, xc_block=xc_block))
        (job_dir / "KPOINTS").write_text(ssj.KPOINTS_TEMPLATE)

        if pp_root is not None:
            ok = ssj.build_potcar([metal], pp_root, job_dir / "POTCAR",
                                  dry_run=False)
            if not ok:
                print(f"  WARNING: POTCAR not written for {metal}")
                n_part += 1
        else:
            cat_cmd = f"$VASP_PP_PATH/{ssj.POTCAR_MAP.get(metal, [metal])[0]}/POTCAR"
            (job_dir / "make_potcar.sh").write_text(
                f"# Build POTCAR for {surface}\ncat {cat_cmd} > POTCAR\n")
            n_part += 1

        slurm_path = job_dir / slurm_filename
        slurm_path.write_text(slurm_template.format(job_name=surface[:40]))
        slurm_path.chmod(0o755)

        if functional == "beef-vdw":
            kernel = Path(args.vdw_kernel_path) if args.vdw_kernel_path \
                else Path(ssj.DEFAULT_VDW_KERNEL_PATH)
            if kernel.exists():
                shutil.copy2(kernel, job_dir / "vdw_kernel.bindat")
            else:
                print(f"  WARNING: vdw_kernel.bindat not found at {kernel}")
        n_ok += 1

    print(f"\nDone: {n_ok} written, {n_skip} skipped, {n_part} partial "
          f"(POTCAR/vdw pending). Surfaces: {len(reps)}")


if __name__ == "__main__":
    main()
