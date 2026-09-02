#!/usr/bin/env python
"""
setup_fully_relaxed.py
======================
Fix jobs under poscar/best/**/<functional>/ that were accidentally set up as
single-point (NSW=0). For each such functional job dir this creates a
`fully_relaxed/` subfolder holding a proper ionic-relaxation job:

    poscar/best/C2/CH3OH_Pd111/PBE_D3/            <- original (NSW=0)
        INCAR  KPOINTS  POTCAR  POSCAR  slm.vasp.kestrel  [OUTCAR CONTCAR ...]
        fully_relaxed/                            <- NEW (this script)
            INCAR    (NSW=1000, IBRION=2, EDIFFG=-5E-02, same XC block)
            POSCAR   (best available geometry: parent CONTCAR else parent POSCAR)
            KPOINTS  POTCAR  slm.vasp.kestrel  [vdw_kernel.bindat]  (copied)

Design goals
------------
- No POTCAR library needed: the parent's POTCAR/KPOINTS/slurm/vdw_kernel are
  reused as-is, so this runs anywhere the original jobs already live.
- The relaxation starts from the *best* geometry available in the parent:
  CONTCAR if the single-point run produced one, otherwise the parent POSCAR.
- The XC / vdW INCAR block is preserved by rewriting only the ionic-relaxation
  tags in the parent's INCAR (NSW, IBRION) and inserting EDIFFG; everything
  else (GGA/METAGGA/LUSE_VDW/ENCUT/ISMEAR/...) is copied verbatim.
- Idempotent: with default behaviour an existing fully_relaxed/ dir is left
  alone unless --force is given.

Usage
-----
    # dry-run the whole tree
    python setup_fully_relaxed.py --root poscar/best --dry-run

    # create fully_relaxed jobs for every NSW=0 functional dir
    python setup_fully_relaxed.py --root poscar/best

    # only a subset of functionals
    python setup_fully_relaxed.py --root poscar/best --functionals PBE PBE_D3

    # overwrite existing fully_relaxed dirs and emit a joblist to submit
    python setup_fully_relaxed.py --root poscar/best --force \\
        --emit-joblist fully_relaxed_jobs.txt

Submit the created jobs (kestrel example)
-----------------------------------------
    for d in poscar/best/*/*/*/fully_relaxed/; do
        (cd "$d" && sbatch slm.vasp.kestrel)
    done
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# Functional subfolder names as written by setup_vasp_jobs.py.
DEFAULT_FUNCTIONALS = ["PBE", "PBE_D3", "r2scan", "beef_vdw"]

# Files copied verbatim from the parent functional dir into fully_relaxed/.
COPY_FILES = ["POTCAR", "KPOINTS", "vdw_kernel.bindat",
              "slm.vasp.kestrel", "slm.vasp.perlmutter"]


def rewrite_incar_to_relax(incar_text: str) -> str:
    """Return the INCAR text with ionic tags forced to a full relaxation.

    Preserves every non-ionic line (XC block, ENCUT, smearing, algo, ...).
    Sets NSW=1000, IBRION=2 and ensures EDIFFG=-5E-02 is present.
    """
    lines = incar_text.splitlines()
    out = []
    have_nsw = have_ibrion = have_ediffg = False
    for line in lines:
        stripped = line.strip()
        # Skip pure comment/blank lines through unchanged.
        if not stripped or stripped.startswith("!"):
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip().upper()
        if key == "NSW":
            out.append("NSW    = 1000")
            have_nsw = True
        elif key == "IBRION":
            out.append("IBRION = 2")
            have_ibrion = True
        elif key == "EDIFFG":
            out.append("EDIFFG = -5E-02")
            have_ediffg = True
        else:
            out.append(line)

    # Append any missing ionic tags (single-point INCARs omit EDIFFG entirely).
    missing = []
    if not have_nsw:
        missing.append("NSW    = 1000")
    if not have_ibrion:
        missing.append("IBRION = 2")
    if not have_ediffg:
        missing.append("EDIFFG = -5E-02")
    if missing:
        out.append("")
        out.append("! Ionic Relaxation (added by setup_fully_relaxed.py)")
        out.extend(missing)

    return "\n".join(out) + "\n"


def incar_is_single_point(incar_text: str) -> bool:
    """True if the INCAR has NSW=0 (i.e. a single-point job)."""
    m = re.search(r"^\s*NSW\s*=\s*(\d+)", incar_text, re.MULTILINE)
    return m is not None and int(m.group(1)) == 0


def best_source_poscar(func_dir: Path) -> Path | None:
    """Prefer a produced CONTCAR (non-empty) over the input POSCAR."""
    contcar = func_dir / "CONTCAR"
    if contcar.exists() and contcar.stat().st_size > 0:
        return contcar
    poscar = func_dir / "POSCAR"
    if poscar.exists() and poscar.stat().st_size > 0:
        return poscar
    return None


def find_functional_dirs(root: Path, functionals: list[str]):
    """Yield every functional job dir under root whose name is a functional.

    A functional dir is any directory named like one of `functionals` that
    contains an INCAR. Its own `fully_relaxed` child is never treated as one.
    """
    fset = set(functionals)
    for incar in sorted(root.rglob("INCAR")):
        d = incar.parent
        if d.name in fset:
            yield d


def process(func_dir: Path, dry_run: bool, force: bool) -> dict:
    res = {"dir": str(func_dir), "action": None, "note": ""}
    incar_path = func_dir / "INCAR"
    incar_text = incar_path.read_text()

    if not incar_is_single_point(incar_text):
        res["action"] = "skip"
        res["note"] = "parent INCAR is not NSW=0 (already relaxation)"
        return res

    target = func_dir / "fully_relaxed"
    if target.exists() and not force:
        res["action"] = "exists"
        res["note"] = "fully_relaxed/ already present (use --force to overwrite)"
        return res

    src_poscar = best_source_poscar(func_dir)
    if src_poscar is None:
        res["action"] = "error"
        res["note"] = "no CONTCAR/POSCAR to seed the relaxation"
        return res

    res["action"] = "create"
    res["note"] = f"seed={src_poscar.name}"

    if dry_run:
        return res

    target.mkdir(parents=True, exist_ok=True)
    # 1) seed geometry
    (target / "POSCAR").write_bytes(src_poscar.read_bytes())
    # 2) relaxation INCAR (preserves XC block from parent)
    (target / "INCAR").write_text(rewrite_incar_to_relax(incar_text))
    # 3) copy supporting files
    copied = []
    for name in COPY_FILES:
        src = func_dir / name
        if src.exists():
            shutil.copy2(src, target / name)
            copied.append(name)
    res["note"] += " copied=" + ",".join(copied)
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default="poscar/best",
                    help="Root holding the job tree (default: poscar/best)")
    ap.add_argument("--functionals", nargs="+", default=DEFAULT_FUNCTIONALS,
                    help="Functional subfolder names to fix "
                         f"(default: {' '.join(DEFAULT_FUNCTIONALS)})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be created without writing files")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing fully_relaxed/ dir")
    ap.add_argument("--emit-joblist", default=None, metavar="PATH",
                    help="Write created fully_relaxed dirs (one per line) to PATH")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"ERROR: root not found: {root}")

    counts = {"create": 0, "exists": 0, "skip": 0, "error": 0}
    created_dirs = []
    for func_dir in find_functional_dirs(root, args.functionals):
        r = process(func_dir, dry_run=args.dry_run, force=args.force)
        counts[r["action"]] = counts.get(r["action"], 0) + 1
        tag = "[DRY-RUN] " if args.dry_run else ""
        print(f"{tag}{r['action']:7s} {r['dir']}"
              + (f"  ({r['note']})" if r["note"] else ""))
        if r["action"] == "create":
            created_dirs.append(str(func_dir / "fully_relaxed"))

    print("=" * 65)
    print("Summary: "
          + ", ".join(f"{k}={v}" for k, v in counts.items() if v))

    if args.emit_joblist and not args.dry_run and created_dirs:
        Path(args.emit_joblist).write_text(
            "".join(d + "\n" for d in created_dirs)
        )
        print(f"Wrote {len(created_dirs)} job dir(s) -> {args.emit_joblist}")

    if not args.dry_run and counts.get("create"):
        print()
        print("Submit the created relaxation jobs (kestrel):")
        print(f"    for d in {root}/*/*/*/fully_relaxed/; do")
        print('        (cd "$d" && sbatch slm.vasp.kestrel)')
        print("    done")


if __name__ == "__main__":
    main()
