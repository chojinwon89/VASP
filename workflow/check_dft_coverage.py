#!/usr/bin/env python3
"""
check_dft_coverage.py
=====================
Answer "what DFT systems are missing?" by reconciling the in-scope MANIFEST
(the expected 478 well-known grid) against what is actually staged under
`dft_jobs/`.

Unlike resubmit_dft.py -- which only sees systems that were already staged
(have a job dir) -- this catches systems that never got a POSCAR at all, e.g.
CH3_Pt111. Each missing system is classified by *why*:

    missing_no_cif   in the MANIFEST but has no SevenNet gallery_cif yet
                     -> nothing to stage; needs a SevenNet gap-fill run
    missing_has_cif  has a gallery_cif but no staged POSCAR
                     -> a real staging bug: re-run stage_dft_poscars.py

Adsorption-site variants in the MANIFEST (e.g. CO_Pt111_atop, CO_Pt111_fcc,
CH3_Pt111_top) collapse to one physical system (CO_Pt111 / CH3_Pt111), because
the GOAD/SevenNet gallery yields one global-min structure per surface+molecule.

Usage
-----
    # full coverage report + write dft_coverage.csv:
    python workflow/check_dft_coverage.py

    # look up one system (site suffix optional):
    python workflow/check_dft_coverage.py --system CH3_Pt111

    # only print the physical systems still missing:
    python workflow/check_dft_coverage.py --missing-only

    # list the SevenNet gap-fill targets (no_cif systems):
    python workflow/check_dft_coverage.py --emit-gapfill
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

SURFACE_RE = re.compile(r'^[A-Z][a-z]?(?:100|110|111)$')


def strip_site(system_id: str) -> str:
    """CH3_Pt111_top -> CH3_Pt111 (drop a trailing adsorption-site token when
    the part before it is a <metal><facet> surface). Leaves plain ids alone."""
    parts = system_id.split("_")
    if len(parts) >= 3 and SURFACE_RE.match(parts[-2]):
        return "_".join(parts[:-1])
    return system_id


def load_manifest(path: Path, exclude_groups, include_groups):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if include_groups:
        rows = [r for r in rows if r["group"] in include_groups]
    if exclude_groups:
        rows = [r for r in rows if r["group"] not in exclude_groups]
    return rows


def staged_dir_names(jobs_dir: Path):
    """Names of staged systems = subdirs that actually contain a POSCAR."""
    if not jobs_dir.is_dir():
        sys.exit(f"ERROR: jobs dir not found: {jobs_dir}")
    return {d.name for d in jobs_dir.iterdir()
            if d.is_dir() and (d / "POSCAR").is_file()}


def build_coverage(rows, staged):
    """Collapse manifest rows to physical systems (base_id) and mark staged."""
    phys = {}
    for r in rows:
        base = strip_site(r["system_id"])
        p = phys.setdefault(base, {
            "base_id": base, "molecule": r["molecule"],
            "surface": r["metal"] + r["facet"], "metal": r["metal"],
            "facet": r["facet"], "groups": set(), "any_cif": False,
            "system_ids": [], "cifs": set()})
        p["groups"].add(r["group"])
        p["system_ids"].append(r["system_id"])
        cif = (r.get("gallery_cif") or "").strip()
        if cif:
            p["any_cif"] = True
            p["cifs"].add(cif)
    for base, p in phys.items():
        p["staged"] = (base in staged) or any(sid in staged
                                              for sid in p["system_ids"])
        if p["staged"]:
            p["status"] = "staged"
        elif p["any_cif"]:
            p["status"] = "missing_has_cif"     # staging bug
        else:
            p["status"] = "missing_no_cif"      # needs SevenNet
    return phys


def lookup(phys, staged, query):
    q = query.strip()
    base = strip_site(q)
    hit = phys.get(q) or phys.get(base)
    if not hit:
        # maybe they passed a molecule_metalfacet that only differs by alias;
        # try a loose contains match on base_id
        cands = [p for b, p in phys.items()
                 if b.lower() == base.lower()
                 or b.lower().startswith(base.lower() + "_")]
        hit = cands[0] if cands else None
    print("=" * 60)
    print(f"lookup: {query}")
    print("=" * 60)
    if not hit:
        print("  NOT in the in-scope MANIFEST.")
        if q in staged or base in staged:
            print("  ...but a staged POSCAR exists (out-of-scope / alias?).")
        else:
            print("  Check spelling, group scope (--include/--exclude-groups),")
            print("  or whether it belongs to the deoxy set (excluded here).")
        return
    print(f"  physical system : {hit['base_id']}")
    print(f"  surface / mol   : {hit['surface']} / {hit['molecule']}")
    print(f"  manifest ids    : {', '.join(hit['system_ids'])}")
    print(f"  group(s)        : {', '.join(sorted(hit['groups']))}")
    print(f"  gallery_cif     : {', '.join(sorted(hit['cifs'])) or '(none)'}")
    print(f"  staged POSCAR   : {'YES' if hit['staged'] else 'NO'}")
    print()
    if hit["status"] == "staged":
        print("  VERDICT: present. DFT job dir exists under dft_jobs/.")
    elif hit["status"] == "missing_no_cif":
        print("  VERDICT: MISSING - no SevenNet gallery structure yet.")
        print("           -> run the SevenNet gap-fill for this surface+molecule,")
        print("              collect the best CIF, then re-run stage_dft_poscars.py.")
    else:
        print("  VERDICT: MISSING but a gallery_cif EXISTS -> staging bug.")
        print("           -> re-run stage_dft_poscars.py (it should stage this).")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", default="dft_jobs")
    ap.add_argument("--manifest", default="DFT_results/MANIFEST.csv")
    ap.add_argument("--exclude-groups", default="deoxy",
                    help="Comma list of MANIFEST groups to drop (default: deoxy).")
    ap.add_argument("--include-groups", default="",
                    help="If set, keep ONLY these groups.")
    ap.add_argument("--system", help="Look up one system (site suffix optional).")
    ap.add_argument("--missing-only", action="store_true",
                    help="Print only the physical systems still missing.")
    ap.add_argument("--emit-gapfill", action="store_true",
                    help="Print the no_cif systems (SevenNet gap-fill targets).")
    ap.add_argument("--out", default="dft_coverage.csv",
                    help="Write full per-system status here (default: dft_coverage.csv).")
    args = ap.parse_args()

    man_path = Path(args.manifest)
    if not man_path.is_file():
        sys.exit(f"ERROR: manifest not found: {man_path}")
    excl = {g.strip() for g in args.exclude_groups.split(",") if g.strip()}
    incl = {g.strip() for g in args.include_groups.split(",") if g.strip()}

    rows = load_manifest(man_path, excl, incl)
    staged = staged_dir_names(Path(args.jobs_dir))
    phys = build_coverage(rows, staged)

    if args.system:
        lookup(phys, staged, args.system)
        return

    items = sorted(phys.values(), key=lambda p: (p["surface"], p["molecule"]))
    covered = [p for p in items if p["status"] == "staged"]
    miss_nocif = [p for p in items if p["status"] == "missing_no_cif"]
    miss_cif = [p for p in items if p["status"] == "missing_has_cif"]
    matched_dirs = {b for p in items if p["staged"]
                    for b in ([p["base_id"]] + p["system_ids"])}
    orphans = sorted(d for d in staged if d not in matched_dirs)

    if args.emit_gapfill:
        for p in miss_nocif:
            print(p["base_id"])
        return
    if args.missing_only:
        for p in miss_cif + miss_nocif:
            tag = "HAS_CIF(bug)" if p["status"] == "missing_has_cif" else "no_cif"
            print(f"{p['base_id']:24s} {tag}")
        return

    # ---------------------------------------------------------------- report
    scope = f"exclude={sorted(excl)}" + (f" include={sorted(incl)}" if incl else "")
    print("=" * 66)
    print("DFT staging coverage")
    print("=" * 66)
    print(f"  manifest : {man_path}   ({scope})")
    print(f"  jobs-dir : {args.jobs_dir}")
    print(f"  expected physical systems : {len(items)}")
    print(f"  staged (POSCAR present)   : {len(covered)}")
    print(f"  MISSING                   : {len(miss_nocif) + len(miss_cif)}")
    print(f"      - no gallery_cif (needs SevenNet) : {len(miss_nocif)}")
    print(f"      - has cif, not staged (BUG)       : {len(miss_cif)}")
    if orphans:
        print(f"  staged but not in-scope (orphans) : {len(orphans)}")

    # per-surface table
    per = defaultdict(lambda: [0, 0])  # surface -> [staged, expected]
    for p in items:
        per[p["surface"]][1] += 1
        if p["staged"]:
            per[p["surface"]][0] += 1
    print("\n  per-surface coverage (staged / expected):")
    for surf in sorted(per):
        s, e = per[surf]
        flag = "   <-- GAP" if s < e else ""
        print(f"    {surf:8s} {s:3d}/{e:3d}{flag}")

    if miss_cif:
        print(f"\n  !! {len(miss_cif)} system(s) HAVE a cif but were not staged "
              f"(re-run stage_dft_poscars.py):")
        for p in miss_cif:
            print(f"    {p['base_id']:24s} cif={sorted(p['cifs'])}")

    if miss_nocif:
        print(f"\n  {len(miss_nocif)} system(s) need a SevenNet structure "
              f"(showing up to 40):")
        for p in miss_nocif[:40]:
            print(f"    {p['base_id']:24s} ({','.join(sorted(p['groups']))})")
        if len(miss_nocif) > 40:
            print(f"    ... (+{len(miss_nocif) - 40} more; see {args.out})")

    if orphans:
        print(f"\n  orphan staged dirs (not matched to in-scope manifest):")
        for d in orphans[:40]:
            print(f"    {d}")

    # ------------------------------------------------------------------- csv
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["base_id", "surface", "molecule", "status", "staged",
                    "any_cif", "groups", "manifest_ids", "cifs"])
        for p in items:
            w.writerow([p["base_id"], p["surface"], p["molecule"], p["status"],
                        int(p["staged"]), int(p["any_cif"]),
                        "|".join(sorted(p["groups"])),
                        "|".join(p["system_ids"]), "|".join(sorted(p["cifs"]))])
    print(f"\n  wrote per-system status -> {args.out}")


if __name__ == "__main__":
    main()
