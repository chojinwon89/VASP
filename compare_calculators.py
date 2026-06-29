#!/usr/bin/env python
"""
compare_calculators.py
======================
Compare SevenNet-OMNI vs MatterSim-5M best adsorption energies from
workflow/summary.csv and rank surface+molecule pairs by disagreement.

Large disagreements flag systems where the two MLFFs predict very
different physics — these are exactly the cases most needing DFT
validation.  The script outputs:

  1. Ranked table to stdout (sorted by |E_ads_sevennet - E_ads_5m|)
  2. comparison.csv — full comparison for all matched pairs
  3. priority_dft_jobs.csv — top N pairs for DFT validation
  4. calc_comparison.png — parity plot sevennet_omni vs 5m

Usage
-----
    # Default: reads workflow/summary.csv, top 30 DFT priority pairs
    python compare_calculators.py

    # Custom summary CSV and top-N
    python compare_calculators.py \\
        --summary workflow/summary.csv \\
        --top 20 \\
        --out-csv results/comparison.csv \\
        --priority-csv results/priority_dft_jobs.csv \\
        --plot results/calc_comparison.png

    # Compare a different pair of calculators
    python compare_calculators.py --calc-a sevennet_omni --calc-b 5m_d3
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------
CALC_LABELS = {
    "sevennet_omni": "SevenNet-OMNI",
    "5m":            "MatterSim-5M",
    "5m_d3":         "MatterSim-5M+D3",
    "1m":            "MatterSim-1M",
}

METAL_COLORS = {
    "Cu": "#D55E00",
    "Pt": "#0072B2",
    "Pd": "#009E73",
    "Ni": "#CC79A7",
    "Ag": "#F0E442",
    "Au": "#56B4E9",
}

MOLECULE_MARKERS = {
    "glycerol":    "o",
    "isopropanol": "s",
    "propanol":    "^",
    "ethanol":     "D",
    "propane":     "v",
    "ethane":      "P",
    "propene":     "X",
    "ethene":      "*",
    "CO2":         "h",
}

# Disagreement thresholds for colouring the priority table
THRESH_HIGH   = 0.3   # eV — likely structural difference, high DFT priority
THRESH_MEDIUM = 0.15  # eV — worth checking


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_best(summary_path: Path, calc: str) -> dict:
    """
    Return {(surface, adsorbate): best_E_ads_eV} for finished rows
    matching `calc`, taking the minimum E_ads across all seeds.
    """
    best = defaultdict(lambda: float("inf"))
    skipped = 0

    with summary_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("calculator", "").strip() != calc:
                continue
            state = row.get("state", "").strip()
            if state != "finished":
                skipped += 1
                continue
            key = (row["surface"].strip(), row["adsorbate"].strip())
            try:
                e = float(row["E_ads_eV"])
                if e < best[key]:
                    best[key] = e
            except ValueError:
                pass

    if skipped:
        print(f"  [{calc}] skipped {skipped} non-finished rows")

    return {k: v for k, v in best.items() if v != float("inf")}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(pairs, a_vals, b_vals):
    a = np.array([a_vals[k] for k in pairs])
    b = np.array([b_vals[k] for k in pairs])
    diffs = b - a
    mae   = float(np.mean(np.abs(diffs)))
    rmse  = float(np.sqrt(np.mean(diffs ** 2)))
    bias  = float(np.mean(diffs))          # positive = B predicts stronger binding
    r2    = float(np.corrcoef(a, b)[0, 1] ** 2) if len(pairs) > 1 else float("nan")
    return mae, rmse, bias, r2


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(pairs, a_vals, b_vals, calc_a, calc_b, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    all_vals = []

    for (surf, mol) in pairs:
        metal  = surf[:2]
        color  = METAL_COLORS.get(metal, "grey")
        marker = MOLECULE_MARKERS.get(mol, "o")
        x = a_vals[(surf, mol)]
        y = b_vals[(surf, mol)]
        all_vals.extend([x, y])
        ax.scatter(x, y, color=color, marker=marker,
                   s=70, alpha=0.85, edgecolors="k", linewidths=0.5, zorder=3)

    if not all_vals:
        print("No data to plot.")
        plt.close(fig)
        return

    lo = min(all_vals) - 0.15
    hi = max(all_vals) + 0.15
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x", zorder=2)

    # ±0.15 eV band
    ax.fill_between([lo, hi], [lo - 0.15, hi - 0.15],
                    [lo + 0.15, hi + 0.15],
                    color="grey", alpha=0.10, zorder=1, label="±0.15 eV")

    mae, rmse, bias, r2 = compute_stats(pairs, a_vals, b_vals)
    label_a = CALC_LABELS.get(calc_a, calc_a)
    label_b = CALC_LABELS.get(calc_b, calc_b)

    stats_txt = (
        f"N = {len(pairs)}\n"
        f"MAE  = {mae:.3f} eV\n"
        f"RMSE = {rmse:.3f} eV\n"
        f"Bias = {bias:+.3f} eV\n"
        f"R²   = {r2:.3f}"
    )
    ax.text(0.04, 0.97, stats_txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    # Metal legend
    metals_present = sorted({s[:2] for s, _ in pairs})
    metal_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=METAL_COLORS.get(m, "grey"),
                   markeredgecolor="k", markersize=8, label=m)
        for m in metals_present
    ]
    leg1 = ax.legend(handles=metal_handles, title="Metal",
                     loc="lower right", fontsize=8, title_fontsize=8)

    # Molecule legend
    mols_present = sorted({mol for _, mol in pairs})
    mol_handles = [
        plt.Line2D([0], [0], marker=MOLECULE_MARKERS.get(mol, "o"),
                   color="w", markerfacecolor="grey",
                   markeredgecolor="k", markersize=8, label=mol)
        for mol in mols_present
    ]
    leg2 = ax.legend(handles=mol_handles, title="Molecule",
                     loc="upper right", fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)

    ax.set_xlabel(f"{label_a}  $E_{{ads}}$ (eV)", fontsize=12)
    ax.set_ylabel(f"{label_b}  $E_{{ads}}$ (eV)", fontsize=12)
    ax.set_title(f"{label_a} vs {label_b}  —  Adsorption Energies", fontsize=13)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rank surface+molecule pairs by SevenNet vs MatterSim disagreement "
            "to identify which DFT validation jobs are highest priority."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summary", default="workflow/summary.csv",
        help="GOAD summary CSV from collect_results.py (default: workflow/summary.csv)"
    )
    parser.add_argument(
        "--calc-a", default="sevennet_omni",
        help="First calculator (default: sevennet_omni)"
    )
    parser.add_argument(
        "--calc-b", default="5m",
        help="Second calculator (default: 5m)"
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of highest-disagreement pairs to highlight (default: 30)"
    )
    parser.add_argument(
        "--out-csv", default="results/comparison.csv",
        help="Full comparison CSV output (default: results/comparison.csv)"
    )
    parser.add_argument(
        "--priority-csv", default="results/priority_dft_jobs.csv",
        help="Top-N priority DFT jobs CSV (default: results/priority_dft_jobs.csv)"
    )
    parser.add_argument(
        "--plot", default="results/calc_comparison.png",
        help="Parity plot output (default: results/calc_comparison.png)"
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found.")
        print("Run collect_results.py first to generate the summary CSV.")
        raise SystemExit(1)

    label_a = CALC_LABELS.get(args.calc_a, args.calc_a)
    label_b = CALC_LABELS.get(args.calc_b, args.calc_b)

    print(f"Summary CSV : {summary_path}")
    print(f"Comparing   : {label_a}  vs  {label_b}")
    print()

    print(f"Loading {label_a} results...")
    a_vals = load_best(summary_path, args.calc_a)
    print(f"  {len(a_vals)} finished (surface, molecule) pairs")

    print(f"Loading {label_b} results...")
    b_vals = load_best(summary_path, args.calc_b)
    print(f"  {len(b_vals)} finished (surface, molecule) pairs")
    print()

    # Matched pairs only
    common = sorted(set(a_vals) & set(b_vals))
    only_a = sorted(set(a_vals) - set(b_vals))
    only_b = sorted(set(b_vals) - set(a_vals))

    print(f"Matched pairs : {len(common)}")
    print(f"Only in {label_a:<20}: {len(only_a)}")
    print(f"Only in {label_b:<20}: {len(only_b)}")
    print()

    if not common:
        print("No matched pairs found. Check that both calculators have finished runs.")
        raise SystemExit(1)

    # Build comparison rows sorted by |diff| descending
    rows = []
    for (surf, mol) in common:
        ea = a_vals[(surf, mol)]
        eb = b_vals[(surf, mol)]
        diff = eb - ea
        rows.append({
            "surface":       surf,
            "molecule":      mol,
            f"E_ads_{args.calc_a}_eV": f"{ea:.4f}",
            f"E_ads_{args.calc_b}_eV": f"{eb:.4f}",
            "diff_eV":       f"{diff:+.4f}",
            "abs_diff_eV":   f"{abs(diff):.4f}",
            "priority":      (
                "HIGH"   if abs(diff) >= THRESH_HIGH   else
                "MEDIUM" if abs(diff) >= THRESH_MEDIUM else
                "LOW"
            ),
        })

    rows.sort(key=lambda r: float(r["abs_diff_eV"]), reverse=True)

    # ---- Overall stats ----
    mae, rmse, bias, r2 = compute_stats(common, a_vals, b_vals)
    print(f"Overall statistics ({len(common)} pairs):")
    print(f"  MAE  = {mae:.4f} eV")
    print(f"  RMSE = {rmse:.4f} eV")
    print(f"  Bias = {bias:+.4f} eV  ({'5m stronger' if bias < 0 else 'sevennet stronger'} binding)")
    print(f"  R²   = {r2:.4f}")
    print()

    n_high   = sum(1 for r in rows if r["priority"] == "HIGH")
    n_medium = sum(1 for r in rows if r["priority"] == "MEDIUM")
    n_low    = sum(1 for r in rows if r["priority"] == "LOW")
    print(f"Priority breakdown (|diff| threshold: HIGH≥{THRESH_HIGH} eV, MEDIUM≥{THRESH_MEDIUM} eV):")
    print(f"  HIGH   (≥{THRESH_HIGH} eV) : {n_high:>4} pairs  ← run DFT first")
    print(f"  MEDIUM (≥{THRESH_MEDIUM} eV) : {n_medium:>4} pairs")
    print(f"  LOW    (<{THRESH_MEDIUM} eV) : {n_low:>4} pairs  ← calculators agree")
    print()

    # ---- Top-N table ----
    top_n = min(args.top, len(rows))
    print(f"Top {top_n} highest-disagreement pairs (DFT validation priority):")
    print()

    col_a = f"E({args.calc_a})"
    col_b = f"E({args.calc_b})"
    hdr = (f"  {'#':<4} {'Surface':<8} {'Molecule':<14} "
           f"{col_a:>14} {col_b:>14} {'Diff':>10} {'Priority':<8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for i, r in enumerate(rows[:top_n], 1):
        diff_f = float(r["diff_eV"])
        prio   = r["priority"]
        flag   = " ◄" if prio == "HIGH" else ""
        print(
            f"  {i:<4} {r['surface']:<8} {r['molecule']:<14} "
            f"{float(r[f'E_ads_{args.calc_a}_eV']):>14.4f} "
            f"{float(r[f'E_ads_{args.calc_b}_eV']):>14.4f} "
            f"{diff_f:>+10.4f} {prio:<8}{flag}"
        )
    print()

    # ---- Write CSVs ----
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Full comparison CSV : {args.out_csv}  ({len(rows)} pairs)")

    priority_rows = [r for r in rows if r["priority"] in ("HIGH", "MEDIUM")][:args.top]
    Path(args.priority_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.priority_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(priority_rows)
    print(f"Priority DFT jobs   : {args.priority_csv}  ({len(priority_rows)} pairs)")
    print()

    # ---- Plot ----
    make_plot(common, a_vals, b_vals, args.calc_a, args.calc_b, Path(args.plot))

    # ---- Hint ----
    print()
    print("Next steps:")
    print(f"  1. Review {args.priority_csv}")
    print(f"  2. Run VASP (PBE+D3) on the HIGH priority pairs first")
    print(f"  3. After VASP, run: python calc_binding_energy.py --output dft_binding_energies.csv")
    print(f"  4. Then: python plot_dft_vs_sevennet.py --calculators sevennet_omni 5m")
    print()


if __name__ == "__main__":
    main()
