#!/usr/bin/env python3
"""
Cheminformatics descriptors + MLIP error stratification.

Builds a per-system feature table (adsorbate, metal, facet, element counts) and,
if given a per-system error CSV, joins it so you can see WHERE the MLIP is worst
-- which chemistries deserve the next DFT / active-learning budget.

This is deliberately dependency-light (stdlib + optional RDKit). Extend
``system_features`` with matminer / pymatgen surface descriptors as needed.

Examples
--------
    python descriptors.py --systems ml/data/valid.extxyz --out ml/desc.csv
    python descriptors.py --systems systems.txt --errors ml/per_system_err.csv \
        --out ml/desc.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("descriptors")


def parse_system(name):
    """Best-effort parse of '<Metal><facet>_<ADSORBATE>...' e.g. Pt111_C2H5OH_top."""
    d = {"system": name, "metal": "", "facet": "", "adsorbate": ""}
    m = re.search(r"([A-Z][a-z]?)(\d{3})[_/]([A-Za-z0-9]+)", name)
    if m:
        d.update(metal=m.group(1), facet=m.group(2), adsorbate=m.group(3))
    return d


def formula_counts(formula):
    """{'C':2,'H':6,'O':1} from 'C2H5OH' — accumulates repeated symbols."""
    counts = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula or ""):
        counts[el] = counts.get(el, 0) + int(n or 1)
    return counts


def system_features(name):
    feat = parse_system(name)
    counts = formula_counts(feat.get("adsorbate", ""))
    for el in ("C", "H", "O", "N"):
        feat["n_%s" % el] = counts.get(el, 0)
    # Optional RDKit hook (only if the adsorbate maps to a known SMILES).
    try:
        from rdkit import Chem                      # noqa: F401
        feat["rdkit"] = "available"
    except Exception:                               # noqa: BLE001
        feat["rdkit"] = ""
    return feat


def load_systems(path):
    if path.endswith(".extxyz") or path.endswith(".xyz"):
        from ase.io import read
        return sorted({a.info.get("system", "?") for a in read(path, index=":")})
    return [ln.strip() for ln in open(path) if ln.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", required=True,
                    help="extxyz (uses 'system' tags) or a text file of system ids")
    ap.add_argument("--errors", default=None,
                    help="CSV with a 'system' column + error columns to join")
    ap.add_argument("--out", default="descriptors.csv")
    args = ap.parse_args(argv)

    names = load_systems(args.systems)
    rows = [system_features(n) for n in names]

    if args.errors:
        err = {r.get("system"): r for r in csv.DictReader(open(args.errors))}
        for row in rows:
            extra = err.get(row["system"]) or {}
            row.update({k: v for k, v in extra.items() if k != "system"})

    cols = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d systems x %d features -> %s", len(rows), len(cols), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
