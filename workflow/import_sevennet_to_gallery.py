#!/usr/bin/env python3
"""
import_sevennet_to_gallery.py
=============================
The missing link between collect_missing_sevennet.py and stage_dft_poscars.py.

collect_missing_sevennet.py drops best-seed structures into
    collected/sevennet_missing/<surface>_<adsorbate>_sevennet_omni.cif
but stage_dft_poscars.py only stages a MANIFEST row whose `gallery_cif` column
is filled AND whose file lives in the `structure/` gallery. This script closes
that gap: for each collected CIF it

  1. maps <adsorbate> -> molecule formula (inverting the token->formula map that
     the MANIFEST already encodes in its filled gallery_cif entries, plus a small
     radical table for the gap-fill species that have no gallery row yet),
  2. finds the in-scope MANIFEST row(s) for that (surface, molecule),
  3. copies the CIF into structure/ (gallery dir), and
  4. writes the filename into that row's `gallery_cif` column.

Site variants of one physical system (CO_Pt111_atop / CO_Pt111_fcc) share the
same relaxed structure, so by default only ONE representative row per
(surface, molecule) is filled (avoids duplicate DFT jobs); pass --all-site-rows
to fill every matching row.

Usage
-----
    # preview what would be imported (no files touched):
    python workflow/import_sevennet_to_gallery.py --dry-run

    # do it (backs up MANIFEST.csv -> MANIFEST.csv.bak, copies CIFs, updates rows):
    python workflow/import_sevennet_to_gallery.py

Then re-stage and verify:
    python workflow/stage_dft_poscars.py --manifest DFT_results/MANIFEST.csv \
        --structure-dir structure --out-dir dft_jobs --fix-bottom-layers 2
    python workflow/check_dft_coverage.py
"""
import argparse
import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Gap-fill / open-shell species that have NO gallery row yet, so their
# adsorbate->formula mapping cannot be recovered from the MANIFEST. Keep in sync
# with make_tasks_missing_sevennet.py ADSORBATES.
RADICAL_TOKEN_TO_FORMULA = {
    "acetylene": "C2H2", "methoxy": "CH3O", "hydroxyl": "OH",
    "atomicH": "H", "atomicO": "O", "HCN": "HCN",
    # extra open-shell radicals (usually out of the 478 scope; harmless if unused)
    "CH": "CH", "atomicC": "C", "C2H5": "C2H5", "C2H3": "C2H3",
    "C2H": "C2H", "CH2OH": "CH2OH", "OOH": "OOH", "COOH": "COOH",
}

SURFACE_RE = re.compile(r"^[A-Z][a-z]?(?:100|110|111)$")


def token_from_gallery(gallery_cif: str, surface: str, calc: str) -> str:
    """Ag100_methanol.cif -> methanol ; Pt100_CH3_sevennet_omni.cif -> CH3."""
    stem = gallery_cif[:-4] if gallery_cif.lower().endswith(".cif") else gallery_cif
    if stem.startswith(surface + "_"):
        stem = stem[len(surface) + 1:]
    for suf in (f"_{calc}", "_sevennet_omni", "_sevennet", "_mattersim"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def parse_collected(fname: str, calc: str):
    """<surface>_<adsorbate>[_<calc>].cif -> (surface, adsorbate) or None."""
    if not fname.lower().endswith(".cif"):
        return None
    stem = fname[:-4]
    parts = stem.split("_", 1)
    if len(parts) != 2 or not SURFACE_RE.match(parts[0]):
        return None
    surface, rest = parts
    for suf in (f"_{calc}", "_sevennet_omni", "_sevennet", "_mattersim"):
        if rest.endswith(suf):
            rest = rest[: -len(suf)]
            break
    return surface, rest


def build_token_to_formula(rows, calc):
    """Recover adsorbate-token -> molecule-formula from filled MANIFEST rows,
    then layer the radical table on top (without overriding real gallery rows)."""
    t2f = {}
    for r in rows:
        g = (r.get("gallery_cif") or "").strip()
        if not g:
            continue
        surface = r["metal"] + r["facet"]
        tok = token_from_gallery(g, surface, calc)
        if tok:
            t2f.setdefault(tok, r["molecule"])
    for tok, formula in RADICAL_TOKEN_TO_FORMULA.items():
        t2f.setdefault(tok, formula)
    return t2f


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collected", default="collected/sevennet_missing",
                    help="Dir of collected CIFs (default: collected/sevennet_missing).")
    ap.add_argument("--structure-dir", default="structure",
                    help="Gallery dir to copy CIFs into (default: structure).")
    ap.add_argument("--manifest", default="DFT_results/MANIFEST.csv")
    ap.add_argument("--calc", default="sevennet_omni",
                    help="Calculator token in the filenames (default: sevennet_omni).")
    ap.add_argument("--exclude-groups", default="deoxy",
                    help="MANIFEST groups to treat as out of scope (default: deoxy).")
    ap.add_argument("--all-site-rows", action="store_true",
                    help="Fill every matching site row, not just one per system.")
    ap.add_argument("--overwrite-filled", action="store_true",
                    help="Also overwrite rows that already have a gallery_cif.")
    ap.add_argument("--out", default=None,
                    help="Write updated MANIFEST here (default: in place + .bak).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (REPO / p)

    collected = resolve(args.collected)
    struct_dir = resolve(args.structure_dir)
    manifest = resolve(args.manifest)
    if not collected.is_dir():
        sys.exit(f"ERROR: collected dir not found: {collected}")
    if not manifest.is_file():
        sys.exit(f"ERROR: manifest not found: {manifest}")
    excl = {g.strip() for g in args.exclude_groups.split(",") if g.strip()}

    rows = list(csv.DictReader(manifest.open()))
    fieldnames = rows[0].keys()
    t2f = build_token_to_formula(rows, args.calc)

    # index in-scope rows by (surface, formula)
    by_key = defaultdict(list)
    for r in rows:
        if r["group"] in excl:
            continue
        by_key[(r["metal"] + r["facet"], r["molecule"])].append(r)

    cifs = sorted(p for p in collected.iterdir()
                  if p.is_file() and p.suffix.lower() == ".cif")

    matched, filled_rows, copied = [], 0, 0
    unknown_token, no_manifest_row, already = [], [], []
    filled_systems = set()

    for cif in cifs:
        parsed = parse_collected(cif.name, args.calc)
        if not parsed:
            unknown_token.append((cif.name, "unparseable filename"))
            continue
        surface, ads = parsed
        formula = t2f.get(ads)
        if not formula:
            unknown_token.append((cif.name, f"no formula for adsorbate '{ads}'"))
            continue
        targets = by_key.get((surface, formula))
        if not targets:
            no_manifest_row.append((cif.name, f"{surface}/{formula} not in scope"))
            continue

        # choose which rows to fill
        rows_to_fill = targets if args.all_site_rows else targets[:1]
        did_fill = False
        for r in rows_to_fill:
            if (r.get("gallery_cif") or "").strip() and not args.overwrite_filled:
                already.append(r["system_id"])
                continue
            r["gallery_cif"] = cif.name
            filled_rows += 1
            did_fill = True

        if did_fill:
            filled_systems.add((surface, formula))
            matched.append((cif.name, surface, formula,
                            [r["system_id"] for r in rows_to_fill]))
            dest = struct_dir / cif.name
            if not args.dry_run:
                struct_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cif, dest)
            copied += 1

    # ---------------------------------------------------------------- report
    print("=" * 68)
    print("import_sevennet_to_gallery" + ("  [DRY RUN]" if args.dry_run else ""))
    print("=" * 68)
    print(f"  collected dir : {collected}")
    print(f"  structure dir : {struct_dir}")
    print(f"  manifest      : {manifest}")
    print(f"  collected CIFs        : {len(cifs)}")
    print(f"  matched systems       : {len(filled_systems)}")
    print(f"  MANIFEST rows filled  : {filled_rows}")
    print(f"  CIFs copied to gallery: {copied}")
    if already:
        print(f"  rows already had a cif (skipped): {len(already)} "
              f"(use --overwrite-filled to replace)")
    if unknown_token:
        print(f"\n  !! {len(unknown_token)} CIF(s) with unmapped adsorbate/name:")
        for n, why in unknown_token[:20]:
            print(f"     {n}: {why}")
        print("     -> add the token to RADICAL_TOKEN_TO_FORMULA if it's real.")
    if no_manifest_row:
        print(f"\n  {len(no_manifest_row)} CIF(s) matched no in-scope MANIFEST row "
              f"(out-of-scope surface/molecule, e.g. extra radicals):")
        for n, why in no_manifest_row[:20]:
            print(f"     {n}: {why}")

    if matched:
        print(f"\n  filled (showing up to 30):")
        for n, s, f, ids in matched[:30]:
            print(f"     {s:6s} {f:9s} <- {n}   -> {', '.join(ids)}")
        if len(matched) > 30:
            print(f"     ... (+{len(matched) - 30} more)")

    if args.dry_run:
        print("\n  DRY RUN - no files written. Re-run without --dry-run to apply.")
        return
    if filled_rows == 0:
        print("\n  Nothing to update.")
        return

    out = resolve(args.out) if args.out else manifest
    if out == manifest:
        bak = manifest.with_suffix(manifest.suffix + ".bak")
        shutil.copyfile(manifest, bak)
        print(f"\n  backed up MANIFEST -> {bak}")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote updated MANIFEST -> {out}")
    print("\n  Next: re-run stage_dft_poscars.py, then check_dft_coverage.py.")


if __name__ == "__main__":
    main()
