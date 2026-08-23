#!/usr/bin/env python
"""
audit_dft_runs.py
=================
Scan a tree of VASP DFT jobs (as laid out by setup_vasp_jobs.py) and report,
for every system x functional, whether it is a CONVERGED full geometry
optimization, an incomplete/unconverged relaxation, only a single-point energy,
or absent.  Pure standard library -- run it anywhere (Kestrel, Perlmutter, laptop).

It does NOT trust folder names alone: functional and calc-type are detected from
each job's INCAR (GGA / METAGGA / IVDW / NSW / IBRION) and cross-checked against
the OUTCAR (ionic-step count + "reached required accuracy" marker).  Folder names
are only a fallback.

Layout it understands (setup_vasp_jobs.py):
    <root>/C<n>/<system>/<FUNC>/                 <- full relaxation
    <root>/C<n>/<system>/singlepoint/<FUNC>/     <- single-point
A "system root" is any directory that holds a POSCAR but no INCAR; each job dir
(holds INCAR and/or OUTCAR) is attributed to its nearest such ancestor.

Usage
-----
    python audit_dft_runs.py --root /scratch/jcho5/.../poscar/best
    python audit_dft_runs.py --root DIR1 --root DIR2 --out-prefix myaudit

Outputs (written next to where you run it, prefix configurable):
    dft_audit_jobs.csv     one row per VASP job found (full detail)
    dft_audit_matrix.csv   pivot: system x functional status codes
plus a printed summary with the actionable "missing full relax" lists.

Status codes (matrix)
    R+   full relax, completed, reached required accuracy   (DONE)
    R-   full relax, completed, but NOT converged           (needs continuation)
    R.   full relax present but no completion footer         (running / crashed)
    S    single-point completed (no full relax present)
    s    single-point present but incomplete
    .    nothing for this functional
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

# Canonical functional display order; extras (rpbe, scan, ...) appended as seen.
CANONICAL_FUNCS = ["pbe", "pbe-d3", "r2scan", "beef-vdw"]

# Folder-name -> functional fallback (used only if INCAR detection fails).
FOLDER_FUNC = {
    "pbe": "pbe", "pbe_d3": "pbe-d3", "pbed3": "pbe-d3",
    "rpbe": "rpbe", "r2scan": "r2scan", "scan": "scan",
    "beef_vdw": "beef-vdw", "beefvdw": "beef-vdw", "beef": "beef-vdw",
}

GGA_CODE = {
    "PE": "pbe", "RP": "rpbe", "BF": "beef-vdw", "RE": "revpbe",
    "PS": "pbesol", "91": "pw91", "AM": "am05",
}


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_incar(path: Path) -> dict:
    """Return a dict of upper-cased INCAR tags -> first value token (string)."""
    tags: dict[str, str] = {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return tags
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        # allow "A = 1 ; B = 2" style
        for chunk in line.split(";"):
            if "=" not in chunk:
                continue
            key, _, val = chunk.partition("=")
            key = key.strip().upper()
            val = val.strip().split()[0] if val.strip() else ""
            if key:
                tags[key] = val
    return tags


def _to_int(val: str):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def functional_from_incar(tags: dict) -> str | None:
    metagga = tags.get("METAGGA", "").upper()
    if metagga and metagga not in (".FALSE.", "NONE"):
        if "R2SCAN" in metagga:
            return "r2scan"
        if metagga == "SCAN":
            return "scan"
        return "metagga:" + metagga.lower()
    gga = tags.get("GGA", "").upper()
    ivdw = _to_int(tags.get("IVDW", ""))
    luse = tags.get("LUSE_VDW", "").upper().startswith(".T")
    if gga:
        base = GGA_CODE.get(gga, "gga:" + gga.lower())
        if base == "pbe" and ivdw in (11, 12):
            return "pbe-d3"
        if base == "pbe" and luse:
            return "pbe-vdw"
        return base
    if luse:
        return "vdw-df"
    return None


def calctype_from_incar(tags: dict) -> str | None:
    nsw = _to_int(tags.get("NSW", ""))
    ibrion = _to_int(tags.get("IBRION", ""))
    if ibrion is not None and ibrion == -1:
        return "single-point"
    if nsw is not None and nsw == 0:
        return "single-point"
    if nsw is not None and nsw > 0 and (ibrion is None or ibrion in (1, 2, 3)):
        return "relax"
    if ibrion in (1, 2, 3):
        return "relax"
    return None


_TOTEN = re.compile(r"free\s+energy\s+TOTEN\s*=\s*(-?\d+\.\d+)")
_SIGMA0 = re.compile(r"energy\(sigma->0\)\s*=\s*(-?\d+\.\d+)")
_NIONS = re.compile(r"NIONS\s*=\s*(\d+)")


def parse_outcar(path: Path) -> dict:
    """Stream the OUTCAR once, collecting completion / convergence / energy."""
    out = {
        "outcar": True, "completed": False, "n_ionic": 0,
        "ionic_converged": False, "ediff_reached": False,
        "energy_eV": None, "natoms": None, "empty": False,
    }
    last_sigma0 = None
    last_toten = None
    try:
        size = path.stat().st_size
    except OSError:
        return out
    if size == 0:
        out["empty"] = True
        return out
    try:
        with path.open("r", errors="replace") as fh:
            for line in fh:
                if "free  energy   TOTEN" in line or "free energy    TOTEN" in line:
                    m = _TOTEN.search(line)
                    if m:
                        last_toten = float(m.group(1))
                        out["n_ionic"] += 1
                elif "energy(sigma->0)" in line:
                    m = _SIGMA0.search(line)
                    if m:
                        last_sigma0 = float(m.group(1))
                elif "aborting loop because EDIFF is reached" in line:
                    out["ediff_reached"] = True
                elif "reached required accuracy" in line:
                    out["ionic_converged"] = True
                elif "General timing and accounting" in line:
                    out["completed"] = True
                elif out["natoms"] is None and "NIONS" in line:
                    m = _NIONS.search(line)
                    if m:
                        out["natoms"] = int(m.group(1))
    except OSError:
        return out
    out["energy_eV"] = last_sigma0 if last_sigma0 is not None else last_toten
    return out


# --------------------------------------------------------------------------- #
# Tree walking
# --------------------------------------------------------------------------- #
def find_jobs_and_systems(root: Path):
    """Return (job_dirs, system_roots) as sets of Path.

    job dir      = contains INCAR or OUTCAR
    system root  = contains POSCAR but no INCAR (source geometry, not a job)
    """
    job_dirs: set[Path] = set()
    system_only: set[Path] = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        fset = set(filenames)
        has_job = ("INCAR" in fset) or ("OUTCAR" in fset)
        if has_job:
            job_dirs.add(d)
        elif "POSCAR" in fset:
            system_only.add(d)
    return job_dirs, system_only


def system_root_for(job_dir: Path, root: Path) -> Path:
    """Nearest ancestor holding a POSCAR but no INCAR; fallback strips folder."""
    cur = job_dir.parent
    while (cur == root) or (root in cur.parents):
        if (cur / "POSCAR").exists() and not (cur / "INCAR").exists():
            return cur
        if cur == root:
            break
        cur = cur.parent
    # fallback: strip a trailing functional folder and optional 'singlepoint'
    parts = list(job_dir.relative_to(root).parts)
    if parts and parts[-1].lower() in FOLDER_FUNC:
        parts = parts[:-1]
    if parts and parts[-1].lower() == "singlepoint":
        parts = parts[:-1]
    return root.joinpath(*parts) if parts else job_dir


def classify_job(job_dir: Path, root: Path) -> dict:
    incar = parse_incar(job_dir / "INCAR") if (job_dir / "INCAR").exists() else {}
    outc = (parse_outcar(job_dir / "OUTCAR")
            if (job_dir / "OUTCAR").exists()
            else {"outcar": False, "completed": False, "n_ionic": 0,
                  "ionic_converged": False, "ediff_reached": False,
                  "energy_eV": None, "natoms": None, "empty": False})

    func = functional_from_incar(incar) or FOLDER_FUNC.get(job_dir.name.lower())
    calc = calctype_from_incar(incar)
    if calc is None:  # fall back to OUTCAR / path
        parts_lower = [p.lower() for p in job_dir.relative_to(root).parts]
        if "singlepoint" in parts_lower:
            calc = "single-point"
        elif outc["n_ionic"] > 1 or outc["ionic_converged"]:
            calc = "relax"
        elif outc["n_ionic"] == 1:
            calc = "single-point"
        else:
            calc = "unknown"

    sysroot = system_root_for(job_dir, root)
    try:
        sysid = sysroot.relative_to(root).as_posix()
        if sysid == ".":
            sysid = sysroot.name
    except ValueError:
        sysid = job_dir.name
    return {
        "system": sysid,
        "functional": func or "unknown",
        "calc_type": calc,
        "has_outcar": outc["outcar"],
        "completed": outc["completed"],
        "n_ionic": outc["n_ionic"],
        "ionic_converged": outc["ionic_converged"],
        "ediff_reached": outc["ediff_reached"],
        "energy_eV": outc["energy_eV"],
        "natoms": outc["natoms"],
        "rel_path": job_dir.relative_to(root).as_posix(),
    }


def cell_code(jobs: list) -> str:
    """Compact status for one (system, functional) group of jobs."""
    relax = [j for j in jobs if j["calc_type"] == "relax"]
    sp = [j for j in jobs if j["calc_type"] == "single-point"]
    if any(j["completed"] and j["ionic_converged"] for j in relax):
        return "R+"
    if any(j["completed"] and not j["ionic_converged"] for j in relax):
        return "R-"
    if relax:
        return "R."
    if any(j["completed"] for j in sp):
        return "S"
    if sp:
        return "s"
    return "."


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load_manifest_scope(manifest_path, include_groups=None, exclude_groups=None):
    """Read MANIFEST.csv and return (scope_stems, meta).

    scope_stems : set of gallery_cif basenames (no .cif extension) for in-scope
                  rows -- these are the Kestrel/gallery system directory names to
                  keep (e.g. 'Ag100_ethylene').
    A row is in scope if its `group` passes the include/exclude filters.  Rows
    with an empty `gallery_cif` are counted (no_cif) but cannot be mapped to a
    run directory -- they need the SevenNet gap-fill first.
    """
    inc = {g.strip() for g in include_groups} if include_groups else None
    exc = {g.strip() for g in exclude_groups} if exclude_groups else set()
    stems: set[str] = set()
    n_rows = n_scope = n_no_cif = 0
    with open(manifest_path, newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            grp = (row.get("group") or "").strip()
            if inc is not None and grp not in inc:
                continue
            if grp in exc:
                continue
            n_scope += 1
            cif = (row.get("gallery_cif") or "").strip()
            if not cif:
                n_no_cif += 1
                continue
            stem = cif[:-4] if cif.lower().endswith(".cif") else cif
            stems.add(stem)
    meta = {"rows": n_rows, "in_scope": n_scope, "no_cif": n_no_cif,
            "stems": len(stems)}
    return stems, meta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True, metavar="DIR",
                    help="DFT run tree root (repeatable).")
    ap.add_argument("--out-prefix", default="dft_audit",
                    help="Prefix for output CSVs (default: dft_audit).")
    ap.add_argument("--manifest", default=None, metavar="CSV",
                    help="Optional MANIFEST.csv; restrict the report to its "
                         "in-scope systems (matched via the gallery_cif column). "
                         "Use with --exclude-groups deoxy for the well-known set.")
    ap.add_argument("--exclude-groups", default="", metavar="G1,G2",
                    help="Comma list of manifest 'group' values to drop "
                         "(e.g. deoxy). Requires --manifest.")
    ap.add_argument("--include-groups", default="", metavar="G1,G2",
                    help="Comma list of manifest 'group' values to keep "
                         "(default: all but --exclude-groups). Requires --manifest.")
    args = ap.parse_args()

    scope_stems = None
    scope_meta = None
    if args.manifest:
        mpath = Path(args.manifest)
        if not mpath.is_file():
            ap.error(f"--manifest not found: {mpath}")
        inc = [g for g in args.include_groups.split(",") if g.strip()] or None
        exc = [g for g in args.exclude_groups.split(",") if g.strip()]
        scope_stems, scope_meta = load_manifest_scope(mpath, inc, exc)
        if not scope_stems:
            ap.error("Manifest scope is empty after filtering "
                     "(check --include-groups/--exclude-groups).")

    roots = [Path(r).resolve() for r in args.root]
    for r in roots:
        if not r.is_dir():
            ap.error(f"--root not a directory: {r}")

    all_jobs: list[dict] = []
    system_ids: set[str] = set()
    for root in roots:
        job_dirs, system_only = find_jobs_and_systems(root)
        for jd in sorted(job_dirs):
            rec = classify_job(jd, root)
            all_jobs.append(rec)
            system_ids.add(rec["system"])
        # prepared-but-unrun system roots (POSCAR, no jobs beneath detected here)
        for sd in system_only:
            sid = sd.relative_to(root).as_posix() if sd != root else sd.name
            system_ids.add(sid)

    if not all_jobs and not system_ids:
        print(f"No VASP jobs or POSCAR system dirs found under: "
              f"{', '.join(str(r) for r in roots)}", file=sys.stderr)
        sys.exit(1)

    # ---- optional scope filter: keep only in-scope (well-known) systems ----
    if scope_stems is not None:
        keep = []
        for j in all_jobs:
            base = j["system"].rsplit("/", 1)[-1]
            if base in scope_stems:
                j["system"] = base  # normalize to basename for grouping
                keep.append(j)
        all_jobs = keep
        # report on EVERY in-scope system, so never-run ones show as absent
        system_ids = set(scope_stems)

    # functional column order: always show the 4 canonical, then extras seen
    seen_funcs = {j["functional"] for j in all_jobs}
    funcs = list(CANONICAL_FUNCS)
    funcs += sorted(f for f in seen_funcs if f not in CANONICAL_FUNCS
                    and f != "unknown")
    if "unknown" in seen_funcs:
        funcs.append("unknown")

    # group jobs by (system, functional)
    grouped: dict[tuple, list] = {}
    for j in all_jobs:
        grouped.setdefault((j["system"], j["functional"]), []).append(j)

    systems = sorted(system_ids)

    # ---- write jobs CSV ----
    jobs_csv = Path(f"{args.out_prefix}_jobs.csv")
    with jobs_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "functional", "calc_type", "has_outcar",
                    "completed", "n_ionic", "ionic_converged", "ediff_reached",
                    "energy_eV", "natoms", "rel_path"])
        for j in sorted(all_jobs, key=lambda x: (x["system"], x["functional"],
                                                 x["calc_type"])):
            w.writerow([j["system"], j["functional"], j["calc_type"],
                        j["has_outcar"], j["completed"], j["n_ionic"],
                        j["ionic_converged"], j["ediff_reached"],
                        "" if j["energy_eV"] is None else f"{j['energy_eV']:.6f}",
                        "" if j["natoms"] is None else j["natoms"],
                        j["rel_path"]])

    # ---- write matrix CSV + compute gaps ----
    matrix_csv = Path(f"{args.out_prefix}_matrix.csv")
    missing = {f: [] for f in funcs}
    with matrix_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system"] + funcs)
        for s in systems:
            row = [s]
            for f in funcs:
                code = cell_code(grouped.get((s, f), []))
                row.append(code)
                if code != "R+":
                    missing[f].append(s)
            w.writerow(row)

    # ---- console summary ----
    print("=" * 70)
    print("DFT run audit")
    print("=" * 70)
    print(f"Roots scanned : {', '.join(str(r) for r in roots)}")
    if scope_meta is not None:
        matched = len({j["system"] for j in all_jobs})
        print(f"Scope filter  : {Path(args.manifest).name} "
              f"(excluded groups: {args.exclude_groups or 'none'})")
        print(f"  in-scope systems (mapped to a gallery name) : "
              f"{scope_meta['stems']}")
        print(f"    with >=1 Kestrel DFT job : {matched}")
        print(f"    with NO DFT job (all '.'): {scope_meta['stems'] - matched}")
        if scope_meta["no_cif"]:
            print(f"  (+{scope_meta['no_cif']} in-scope systems have no "
                  f"gallery_cif yet -> need SevenNet gap-fill; not shown)")
    print(f"Systems found : {len(systems)}")
    print(f"Jobs found    : {len(all_jobs)}")
    print()
    print("Legend: R+ relax-converged | R- relax-done-not-converged | "
          "R. relax-running/crashed | S single-point | s sp-incomplete | . none")
    print()
    hdr = "Per-functional coverage (of %d systems):" % len(systems)
    print(hdr)
    for f in funcs:
        counts = {"R+": 0, "R-": 0, "R.": 0, "S": 0, "s": 0, ".": 0}
        for s in systems:
            counts[cell_code(grouped.get((s, f), []))] += 1
        print(f"  {f:10s}  relax_ok={counts['R+']:4d}  "
              f"relax_incomplete={counts['R-'] + counts['R.']:4d}  "
              f"sp_only={counts['S'] + counts['s']:4d}  "
              f"absent={counts['.']:4d}")
    print()
    print("Systems MISSING a converged full relax (need to run on Perlmutter):")
    for f in funcs:
        n = len(missing[f])
        sample = ", ".join(missing[f][:8])
        more = f" ... (+{n - 8} more)" if n > 8 else ""
        print(f"  {f:10s} {n:4d} missing" + (f": {sample}{more}" if n else ""))
    print()
    print(f"Wrote: {jobs_csv}  and  {matrix_csv}")


if __name__ == "__main__":
    main()
