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
    """Map (surface, n_metal) -> (Atoms slab-only, metal, source_path, n_metal).

    Emits one representative per DISTINCT metal count per surface, because the
    same surface can appear with different supercell sizes depending on the
    adsorbate (e.g. Ag100 jobs come as both 36 and 64 metal atoms).
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

        key = (surface, n_metal)
        if key not in reps:
            reps[key] = (slab, metal, src, n_metal)
    return reps


def audit_counts(best_dir: Path, wanted_surfaces=None):
    """Report, per surface, the distinct metal counts across ALL jobs.

    Returns dict: surface -> {count: [job_dir_name, ...]}.
    """
    counts = {}
    for job_dir in sorted(best_dir.glob("**/")):
        parsed = parse_surface(job_dir.name)
        if parsed is None:
            continue
        surface, metal = parsed
        if wanted_surfaces and surface not in wanted_surfaces:
            continue
        atoms, _ = find_job_structure(job_dir)
        if atoms is None:
            continue
        n_metal = sum(1 for s in atoms.get_chemical_symbols() if s == metal)
        if n_metal == 0:
            continue
        counts.setdefault(surface, {}).setdefault(n_metal, []).append(job_dir.name)
    return counts


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
    ap.add_argument("--audit", action="store_true",
                    help="Only report distinct metal counts per surface; write nothing.")
    args = ap.parse_args()

    if args.audit:
        best_dir = Path(args.best_dir)
        if not best_dir.exists():
            raise SystemExit(f"best-dir not found: {best_dir}")
        wanted = set(args.surfaces) if args.surfaces else None
        counts = audit_counts(best_dir, wanted)
        n_mixed = 0
        for surface in sorted(counts):
            by_count = counts[surface]
            sizes = sorted(by_count)
            njobs = sum(len(v) for v in by_count.values())
            flag = ""
            if len(sizes) > 1:
                n_mixed += 1
                flag = "  <-- MIXED SIZES"
                examples = {c: by_count[c][:2] for c in sizes}
                detail = "; ".join(f"{c}:{examples[c]}" for c in sizes)
            else:
                detail = f"{sizes[0]} ({njobs} jobs)"
            print(f"{surface:8s} counts={sizes} {detail}{flag}")
        print(f"\n{len(counts)} surfaces, {n_mixed} with MIXED metal counts.")
        return

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

    # Which surfaces have more than one distinct count -> tag dir with _n<count>.
    surf_counts = {}
    for (surface, n_metal) in reps:
        surf_counts.setdefault(surface, set()).add(n_metal)

    n_ok = n_skip = n_part = 0
    for (surface, n_metal) in sorted(reps):
        slab, metal, src, _ = reps[(surface, n_metal)]
        # Only append _n<count> when the surface is ambiguous, so single-size
        # surfaces keep their plain name for backward compatibility.
        tag = f"{surface}_n{n_metal}" if len(surf_counts[surface]) > 1 else surface
        job_dir = out_dir / tag / subfolder
        outcar = job_dir / "OUTCAR"
        if outcar.exists() and not args.force:
            print(f"skip  {tag:14s} ({n_metal} {metal}) OUTCAR exists")
            n_skip += 1
            continue

        print(f"write {tag:14s} ({n_metal} {metal}) from {src}")
        if args.dry_run:
            continue

        job_dir.mkdir(parents=True, exist_ok=True)

        comment = (f"{tag} slab | {n_metal} {metal} | derived from job | "
                   f"bottom {args.n_fixed} fixed")
        poscar_text = ssj.make_selective_dynamics_poscar(
            slab, args.n_fixed, comment)
        (job_dir / "POSCAR").write_text(poscar_text)
        (job_dir / "INCAR").write_text(
            ssj.INCAR_TEMPLATE.format(system=tag, xc_block=xc_block))
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
                f"# Build POTCAR for {tag}\ncat {cat_cmd} > POTCAR\n")
            n_part += 1

        slurm_path = job_dir / slurm_filename
        slurm_path.write_text(slurm_template.format(job_name=tag[:40]))
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
