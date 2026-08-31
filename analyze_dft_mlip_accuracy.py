#!/usr/bin/env python
"""
analyze_dft_mlip_accuracy.py
============================
Consolidated accuracy report for GOAD+MLIP vs DFT.

Combines three views so you can decide how far to trust GOAD+SevenNet:

  1. ENERGY   - per-functional MAE / RMSE / bias / R2 of E_ads(ML) vs E_ads(DFT),
                plus breakdowns by molecule and by metal. Reads the matched-pair
                CSV emitted by plot_dft_vs_mlip.py (--csv-out).

  2. GEOMETRY - if the atom-by-atom structure comparison CSV is present
                (dft_mlip_structure_compare.csv from compare_dft_mlip_structures.py,
                which needs the DFT CONTCARs on Perlmutter/Kestrel), summarise
                min-contact-distance error, RMSD and max displacement, and the
                adsorption-site agreement (same nearest element pair = same site).

  3. SITE/GEOMETRY QUALITY - cross-reference every energy pair against the MLIP
                geometry category (reasonable / borderline / detached / gas_phase)
                from bond_distances.csv, to test whether the energy error is
                worse for the geometrically suspicious structures.

Everything degrades gracefully: whichever inputs are present are analysed.

Usage
-----
    python analyze_dft_mlip_accuracy.py \
        --pairs analysis_out/dft_vs_mlip_pairs.csv \
        --bond-distances /path/to/bond_distances.csv \
        --struct-compare dft_mlip_structure_compare.csv \
        --out-dir analysis_out
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_metal(surface: str) -> str:
    import re
    m = re.match(r"[A-Za-z]+", surface or "")
    letters = m.group() if m else ""
    return letters[:2] if len(letters) >= 2 else letters


def regression_stats(xy):
    """Return dict of n, mae, rmse, bias, r2 for a list of (dft, ml) pairs.

    bias = mean(ml - dft). r2 is the coefficient of determination of ml against
    dft about the y=x line (i.e. how well ML reproduces DFT, not a free fit).
    """
    xy = [(d, m) for d, m in xy if d is not None and m is not None]
    n = len(xy)
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "r2": None}
    diffs = [m - d for d, m in xy]
    mae = sum(abs(x) for x in diffs) / n
    rmse = math.sqrt(sum(x * x for x in diffs) / n)
    bias = sum(diffs) / n
    dft_mean = sum(d for d, _ in xy) / n
    ss_tot = sum((d - dft_mean) ** 2 for d, _ in xy)
    ss_res = sum((m - d) ** 2 for d, m in xy)   # residual about y=x
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else None
    return {"n": n, "mae": mae, "rmse": rmse, "bias": bias, "r2": r2}


def load_pairs(path: Path, max_diff=None, max_eads=None):
    """Load matched pairs, dropping unphysical DFT rows and outliers.

    A positive (or near-zero-positive) DFT E_ads means the "slab" reference was
    wrong or the relaxation failed (e.g. the corrupt Pd100 references), so those
    rows are excluded. ``max_diff`` drops pairs whose |ML-DFT| exceeds it, the
    same sanity cut used by plot_dft_vs_mlip.py.
    """
    rows = []
    for r in csv.DictReader(path.open()):
        r["E_ads_DFT"] = _f(r.get("E_ads_DFT"))
        r["E_ads_ML"] = _f(r.get("E_ads_ML"))
        r["diff_eV"] = _f(r.get("diff_eV"))
        d, m = r["E_ads_DFT"], r["E_ads_ML"]
        if d is None or m is None:
            continue
        if max_eads is not None and d > max_eads:
            continue                      # unphysical DFT reference
        if max_diff is not None and abs(m - d) > max_diff:
            continue                      # gross outlier
        rows.append(r)
    return rows


def load_categories(path: Path):
    """Map (surface, molecule) -> dict(category, min_dist, pair, ...)."""
    cat = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (r.get("surface", "").strip(), r.get("molecule", "").strip())
            cat[key] = {
                "category": r.get("category", "").strip(),
                "min_dist": _f(r.get("min_dist")),
                "pair": r.get("pair", "").strip(),
                "min_dist_heavy": _f(r.get("min_dist_heavy")),
                "pair_heavy": r.get("pair_heavy", "").strip(),
            }
    return cat


def load_struct_compare(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in ("min_dist_dft", "min_dist_mlip", "d_min_dist",
                      "rmsd", "max_disp"):
                if k in r:
                    r[k] = _f(r[k])
            rows.append(r)
    return rows


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def print_table(title, header, rows, lines):
    lines.append("")
    lines.append(title)
    lines.append("-" * len(title))
    widths = [len(h) for h in header]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))
    lines.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    lines.append("  ".join("-" * widths[i] for i in range(len(header))))
    for row in rows:
        lines.append("  ".join(str(c).ljust(widths[i])
                               for i, c in enumerate(row)))


def energy_report(pairs, lines):
    funcs = sorted({r["functional"] for r in pairs})

    # Overall per-functional.
    rows = []
    for fn in funcs:
        sub = [(r["E_ads_DFT"], r["E_ads_ML"]) for r in pairs
               if r["functional"] == fn]
        s = regression_stats(sub)
        rows.append([fn, s["n"], fmt(s["mae"]), fmt(s["rmse"]),
                     fmt(s["bias"]), fmt(s["r2"])])
    print_table("ENERGY: per-functional (E_ads ML vs DFT, eV)",
                ["functional", "n", "MAE", "RMSE", "bias(ML-DFT)", "R2"],
                rows, lines)

    # Choose the best-agreeing functional for the detailed breakdowns.
    best_fn, best_mae = None, None
    for fn in funcs:
        sub = [(r["E_ads_DFT"], r["E_ads_ML"]) for r in pairs
               if r["functional"] == fn]
        s = regression_stats(sub)
        if s["mae"] is not None and (best_mae is None or s["mae"] < best_mae):
            best_fn, best_mae = fn, s["mae"]
    if best_fn is None:
        return best_fn

    # Per-molecule (for the best functional).
    by_mol = defaultdict(list)
    for r in pairs:
        if r["functional"] == best_fn:
            by_mol[r["molecule"]].append((r["E_ads_DFT"], r["E_ads_ML"]))
    rows = []
    for mol in sorted(by_mol, key=lambda m: -(regression_stats(by_mol[m])["mae"] or 0)):
        s = regression_stats(by_mol[mol])
        rows.append([mol, s["n"], fmt(s["mae"]), fmt(s["bias"])])
    print_table(f"ENERGY: per-molecule (functional={best_fn})",
                ["molecule", "n", "MAE", "bias"], rows, lines)

    # Per-metal (for the best functional).
    by_metal = defaultdict(list)
    for r in pairs:
        if r["functional"] == best_fn:
            by_metal[parse_metal(r["surface"])].append(
                (r["E_ads_DFT"], r["E_ads_ML"]))
    rows = []
    for me in sorted(by_metal, key=lambda m: -(regression_stats(by_metal[m])["mae"] or 0)):
        s = regression_stats(by_metal[me])
        rows.append([me, s["n"], fmt(s["mae"]), fmt(s["bias"])])
    print_table(f"ENERGY: per-metal (functional={best_fn})",
                ["metal", "n", "MAE", "bias"], rows, lines)

    return best_fn


def category_cross_report(pairs, categories, best_fn, lines):
    if not categories:
        return
    by_cat = defaultdict(list)
    matched = 0
    for r in pairs:
        if best_fn and r["functional"] != best_fn:
            continue
        info = categories.get((r["surface"], r["molecule"]))
        if not info:
            continue
        matched += 1
        by_cat[info["category"] or "unknown"].append(
            (r["E_ads_DFT"], r["E_ads_ML"]))
    if matched == 0:
        lines.append("")
        lines.append("GEOMETRY-QUALITY x ENERGY: no (surface,molecule) overlap "
                     "between energy pairs and bond_distances.csv.")
        return
    rows = []
    order = ["reasonable", "borderline", "detached", "gas_phase", "unknown"]
    for cat in sorted(by_cat, key=lambda c: order.index(c) if c in order else 99):
        s = regression_stats(by_cat[cat])
        rows.append([cat, s["n"], fmt(s["mae"]), fmt(s["bias"])])
    print_table(
        f"GEOMETRY-QUALITY x ENERGY: energy error by MLIP geometry category "
        f"(functional={best_fn})",
        ["mlip_category", "n_pairs", "MAE", "bias"], rows, lines)


def geometry_report(struct_rows, lines):
    if not struct_rows:
        lines.append("")
        lines.append("GEOMETRY (atom-level): dft_mlip_structure_compare.csv not "
                     "provided — run compare_dft_mlip_structures.py on the host "
                     "that has the DFT CONTCARs (Perlmutter/Kestrel).")
        return
    # Contact-distance and RMSD summary, plus adsorption-site agreement.
    dd = [r["d_min_dist"] for r in struct_rows if r.get("d_min_dist") is not None]
    rmsd = [r["rmsd"] for r in struct_rows if r.get("rmsd") is not None]
    maxd = [r["max_disp"] for r in struct_rows if r.get("max_disp") is not None]

    def stats(v):
        if not v:
            return (0, None, None, None)
        n = len(v)
        mean_abs = sum(abs(x) for x in v) / n
        rms = math.sqrt(sum(x * x for x in v) / n)
        return (n, mean_abs, rms, max(abs(x) for x in v))

    rows = []
    for label, v in (("min-contact-dist error (DFT-MLIP, A)", dd),
                     ("per-atom RMSD (A)", rmsd),
                     ("max single-atom disp (A)", maxd)):
        n, ma, rms, mx = stats(v)
        rows.append([label, n, fmt(ma), fmt(rms), fmt(mx)])
    print_table("GEOMETRY (atom-level): displacement summary",
                ["metric", "n", "mean|.|", "rms", "max|.|"], rows, lines)

    # Adsorption-site agreement: same nearest element pair on both sides.
    same = tot = 0
    for r in struct_rows:
        pd, pm = r.get("pair_dft", ""), r.get("pair_mlip", "")
        if pd and pm:
            tot += 1
            if pd == pm:
                same += 1
    if tot:
        lines.append("")
        lines.append(f"ADSORPTION SITE: nearest element-pair agrees on "
                     f"{same}/{tot} systems ({100 * same / tot:.0f}%).")


def verdict(pairs, best_fn, lines):
    sub = [(r["E_ads_DFT"], r["E_ads_ML"]) for r in pairs
           if r["functional"] == best_fn]
    s = regression_stats(sub)
    lines.append("")
    lines.append("VERDICT")
    lines.append("=======")
    if s["n"] == 0:
        lines.append("Not enough matched pairs to judge.")
        return
    lines.append(f"Best-agreeing functional: {best_fn}  "
                 f"(MAE={fmt(s['mae'])} eV, bias={fmt(s['bias'])} eV, "
                 f"R2={fmt(s['r2'])}, n={s['n']}).")
    mae = s["mae"]
    if mae is not None:
        if mae <= 0.15:
            tier = ("TRUST for screening and ranking: MLIP E_ads is within "
                    "chemical-accuracy-ish range of DFT.")
        elif mae <= 0.30:
            tier = ("USABLE for coarse screening/ranking; expect ~0.2-0.3 eV "
                    "scatter. Confirm close calls with DFT.")
        else:
            tier = ("USE WITH CAUTION: energy error is large; rely on MLIP for "
                    "geometry generation but re-rank with DFT.")
        lines.append(tier)
    if s["bias"] is not None and abs(s["bias"]) > 0.1:
        sign = "over-binds" if s["bias"] < 0 else "under-binds"
        lines.append(f"Systematic offset: MLIP {sign} vs {best_fn} by "
                     f"{abs(s['bias']):.2f} eV on average (a constant shift you "
                     f"could correct for).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True,
                    help="Matched-pair CSV from plot_dft_vs_mlip.py --csv-out.")
    ap.add_argument("--bond-distances", default=None,
                    help="MLIP bond_distances.csv (for geometry-category cross-tab).")
    ap.add_argument("--struct-compare", default=None,
                    help="dft_mlip_structure_compare.csv (atom-level geometry).")
    ap.add_argument("--max-diff", type=float, default=5.0,
                    help="Drop pairs with |E_ads_ML - E_ads_DFT| > this (eV).")
    ap.add_argument("--max-eads", type=float, default=0.5,
                    help="Drop pairs whose DFT E_ads exceeds this (eV); "
                         "positive E_ads flags a broken reference.")
    ap.add_argument("--out-dir", default="analysis_out")
    args = ap.parse_args()

    pairs = load_pairs(Path(args.pairs), max_diff=args.max_diff,
                       max_eads=args.max_eads)
    categories = (load_categories(Path(args.bond_distances))
                  if args.bond_distances and Path(args.bond_distances).exists()
                  else {})
    struct_rows = (load_struct_compare(Path(args.struct_compare))
                   if args.struct_compare and Path(args.struct_compare).exists()
                   else [])

    lines = []
    lines.append("=" * 70)
    lines.append("GOAD+MLIP vs DFT — consolidated accuracy report")
    lines.append("=" * 70)
    lines.append(f"Energy pairs: {len(pairs)}   "
                 f"MLIP geometry rows: {len(categories)}   "
                 f"struct-compare rows: {len(struct_rows)}")

    best_fn = energy_report(pairs, lines)
    category_cross_report(pairs, categories, best_fn, lines)
    geometry_report(struct_rows, lines)
    verdict(pairs, best_fn, lines)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "dft_mlip_accuracy_report.txt"
    report_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nReport written: {report_path}")


if __name__ == "__main__":
    main()
