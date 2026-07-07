#!/usr/bin/env python
"""
plot_dft_vs_sevennet.py
=======================
Compare GOAD+MLIP adsorption energies against DFT calculations across
multiple DFT functionals.

Produces a 2×2 parity-plot grid — one panel per DFT functional — each
showing all ML calculators overlaid, with per-panel MAE / RMSE / R² stats.

Reads
-----
  dft_binding_energies_all.csv   (from calc_binding_energy.py, merged,
                                   must have a 'functional' column)
  workflow/summary.csv           (from GOAD runs)

Output
------
  PNG : 2×2 subplot figure  (one panel per DFT functional)
  CSV : optional flat table of all matched pairs

Usage
-----
    # Default: SevenNet-OMNI vs all functionals found in the CSV
    python plot_dft_vs_sevennet.py

    # Both ML calculators, specific functionals, custom output
    python plot_dft_vs_sevennet.py \\
        --dft  dft_binding_energies_all.csv \\
        --ml   workflow/summary.csv \\
        --calculators sevennet_omni 5m \\
        --functionals pbe pbe_d3 r2scan beef_vdw \\
        --output results/dft_vs_mlip_all_functionals.png \\
        --csv-out results/dft_vs_mlip_all_functionals.csv

Functional name aliases accepted
---------------------------------
    pbe, PBE
    pbe_d3, pbe+d3, PBE+D3
    r2scan, r²scan
    beef_vdw, beef-vdw, BEEF-vdW
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Display name mappings
# ---------------------------------------------------------------------------

CALC_LABELS = {
    "sevennet_omni": "SevenNet-OMNI",
    "5m":            "MatterSim-5M",
    "5m_d3":         "MatterSim-5M+D3",
    "1m":            "MatterSim-1M",
}

CALC_FILL = {
    "sevennet_omni": "full",
    "5m":            "none",
    "5m_d3":         "none",
    "1m":            "none",
}

CALC_LINEWIDTH = {
    "sevennet_omni": 0.5,
    "5m":            1.2,
    "5m_d3":         1.2,
    "1m":            1.2,
}

# Color per ML calculator (for multi-calc panels)
CALC_COLORS = {
    "sevennet_omni": "#E05C00",   # orange
    "5m":            "#0072B2",   # blue
    "5m_d3":         "#009E73",   # green
    "1m":            "#CC79A7",   # pink
}

FUNC_LABELS = {
    "pbe":      "PBE",
    "pbe_d3":   "PBE+D3",
    "r2scan":   "r\u00b2SCAN",
    "beef_vdw": "BEEF-vdW",
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
    "propanol":    "^",
    "ethene":      "*",
    "CO2":         "h",
    "isobutene":   "+",
    "1-butene":    "x",
    "butadiene":   "<",
    "isoprene":    ">",
    "benzene":     "H",
    "toluene":     "8",
}

# 0.2 eV grey band on parity plots
BAND_WIDTH = 0.2


# ---------------------------------------------------------------------------
# Functional name normalisation
# ---------------------------------------------------------------------------

def normalise_func(name: str) -> str:
    s = name.strip().lower().replace("+", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    aliases = {
        "beef_dfw":  "beef_vdw",
        "beefvdw":   "beef_vdw",
        "pbed3":     "pbe_d3",
        "r2scan":    "r2scan",
        "r_2scan":   "r2scan",
    }
    return aliases.get(s, s)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_dft_all(path: Path) -> dict:
    """
    Returns {functional: {(surface, molecule): E_ads_eV}}
    Accepts both single-functional CSVs (no 'functional' column → key 'default')
    and multi-functional CSVs with a 'functional' column.
    """
    data = defaultdict(dict)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "ok").strip() != "ok":
                continue
            func = normalise_func(row.get("functional", "default"))
            key  = (row["surface"].strip(), row["molecule"].strip())
            try:
                data[func][key] = float(row["E_ads"])
            except (ValueError, KeyError):
                pass
    return dict(data)


def load_ml_best(path: Path, calculators: list) -> dict:
    """
    Returns {calc_name: {(surface, adsorbate): best_E_ads_eV}}
    Only includes rows with state == 'finished'.
    """
    best = {c: defaultdict(lambda: float("inf")) for c in calculators}
    skipped = 0

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            calc = row.get("calculator", "").strip()
            if calc not in calculators:
                continue
            state = row.get("state", row.get("status", "")).strip()
            if state != "finished":
                skipped += 1
                continue
            key = (row["surface"].strip(), row["adsorbate"].strip())
            try:
                e = float(row["E_ads_eV"])
                if e < best[calc][key]:
                    best[calc][key] = e
            except (ValueError, KeyError):
                pass

    if skipped:
        print(f"  Skipped {skipped} row(s) with state != finished.")

    return {c: {k: v for k, v in d.items() if v != float("inf")}
            for c, d in best.items()}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(dft_vals, ml_vals, pairs):
    dft_arr = np.array([dft_vals[k] for k in pairs])
    ml_arr  = np.array([ml_vals[k]  for k in pairs])
    mae  = float(np.mean(np.abs(ml_arr - dft_arr)))
    rmse = float(np.sqrt(np.mean((ml_arr - dft_arr) ** 2)))
    bias = float(np.mean(ml_arr - dft_arr))
    ss_res = np.sum((ml_arr - dft_arr) ** 2)
    ss_tot = np.sum((dft_arr - dft_arr.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, bias, r2


# ---------------------------------------------------------------------------
# Single panel plotter
# ---------------------------------------------------------------------------

def _plot_panel(ax, func, func_label, calc_pairs, dft_vals, ml_data):
    """
    Draw one parity panel on *ax*.
    calc_pairs : {calc: [(surf, mol), ...]}
    """
    all_vals = []

    for calc, pairs in calc_pairs.items():
        fill  = CALC_FILL.get(calc, "full")
        lw    = CALC_LINEWIDTH.get(calc, 0.5)
        # When only one ML calc → use metal colours; when multiple → use calc colour
        multi_calc = len([c for c, p in calc_pairs.items() if p]) > 1

        for (surf, mol) in pairs:
            metal  = surf[:2] if len(surf) >= 2 else surf
            color  = (CALC_COLORS.get(calc, "grey") if multi_calc
                      else METAL_COLORS.get(metal, "grey"))
            marker = MOLECULE_MARKERS.get(mol, "o")
            x = dft_vals[func].get((surf, mol))
            y = ml_data[calc].get((surf, mol))
            if x is None or y is None:
                continue
            all_vals.extend([x, y])

            if fill == "none":
                ax.scatter(x, y, facecolors="none", edgecolors=color,
                           marker=marker, s=70, linewidths=lw,
                           alpha=0.9, zorder=3)
            else:
                ax.scatter(x, y, color=color, marker=marker,
                           s=70, alpha=0.85, linewidths=lw,
                           edgecolors="k", zorder=3)

    if not all_vals:
        ax.set_title(f"DFT: {func_label}  (no data)", fontsize=11)
        return

    lo = min(all_vals) - 0.15
    hi = max(all_vals) + 0.15

    # Parity line
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, zorder=2)

    # ±0.2 eV grey band
    ax.fill_between([lo, hi],
                    [lo - BAND_WIDTH, hi - BAND_WIDTH],
                    [lo + BAND_WIDTH, hi + BAND_WIDTH],
                    color="grey", alpha=0.12, zorder=1)

    # Stats box (lower right)
    stat_lines = []
    for calc, pairs in calc_pairs.items():
        matched = [(s, m) for (s, m) in pairs
                   if (s, m) in dft_vals.get(func, {}) and (s, m) in ml_data.get(calc, {})]
        if not matched:
            continue
        mae, rmse, bias, r2 = compute_stats(dft_vals[func], ml_data[calc],
                                             matched)
        label = CALC_LABELS.get(calc, calc)
        stat_lines.append(
            f"[{label}]\n"
            f"  MAE={mae:.3f}  RM{chr(83)}E={rmse:.3f}\n"
            f"  R\u00b2={r2:.3f}  bias={bias:+.3f}"
        )

    ax.text(0.97, 0.03, "\n".join(stat_lines),
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=6.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"DFT: {func_label}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"DFT ({func_label})  $E_{{ads}}$ (eV)", fontsize=9)


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def make_figure(functionals, calc_pairs_per_func, dft_data, ml_data,
                calculators, output_path: Path):
    """
    functionals          : list of functional keys to plot (max 4 for 2×2)
    calc_pairs_per_func  : {func: {calc: [(surf, mol), ...]}}
    """
    n = len(functionals)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7.5 * ncols, 7 * nrows),
                             squeeze=False)

    # Determine a single shared ML calculator label for the y-axis
    ml_label = " / ".join(CALC_LABELS.get(c, c) for c in calculators)

    for idx, func in enumerate(functionals):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        func_label = FUNC_LABELS.get(func, func.upper())
        _plot_panel(ax, func, func_label,
                    calc_pairs_per_func.get(func, {}),
                    dft_data, ml_data)
        ax.set_ylabel(f"{ml_label}  $E_{{ads}}$ (eV)", fontsize=9)

    # Hide any unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # ── Shared legends ──────────────────────────────────────────────────────
    # Collect all pairs across all functionals for legend filtering
    all_pairs = [p for fp in calc_pairs_per_func.values()
                 for pairs in fp.values() for p in pairs]

    multi_calc = len(calculators) > 1

    # Calculator legend (only if >1 calc)
    if multi_calc:
        calc_handles = []
        for calc in calculators:
            fill  = CALC_FILL.get(calc, "full")
            color = CALC_COLORS.get(calc, "grey")
            label = CALC_LABELS.get(calc, calc)
            if fill == "none":
                h = plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="none", markeredgecolor=color,
                               markeredgewidth=1.4, markersize=9, label=label)
            else:
                h = plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=color, markeredgecolor="k",
                               markersize=9, label=label)
            calc_handles.append(h)
        fig.legend(handles=calc_handles, title="Calculator",
                   loc="upper center", ncol=len(calculators),
                   fontsize=9, title_fontsize=9,
                   bbox_to_anchor=(0.5, 1.01))
    else:
        # Single calc → metal colour legend at top
        metal_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=c, markeredgecolor="k",
                       markersize=9, label=m)
            for m, c in METAL_COLORS.items()
            if any(k[0].startswith(m) for k in all_pairs)
        ]
        fig.legend(handles=metal_handles, title="Metal",
                   loc="upper center", ncol=len(metal_handles),
                   fontsize=9, title_fontsize=9,
                   bbox_to_anchor=(0.5, 1.01))

    # Molecule marker legend (bottom centre)
    mol_handles = [
        plt.Line2D([0], [0], marker=mk, color="w",
                   markerfacecolor="grey", markeredgecolor="k",
                   markersize=9, label=mol)
        for mol, mk in MOLECULE_MARKERS.items()
        if any(k[1] == mol for k in all_pairs)
    ]
    if mol_handles:
        fig.legend(handles=mol_handles, title="Molecule",
                   loc="lower center", ncol=min(len(mol_handles), 6),
                   fontsize=8, title_fontsize=8,
                   bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        f"DFT vs {ml_label}  —  Adsorption Energies (all functionals)",
        fontsize=13, fontweight="bold", y=1.03
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "2×2 parity plot: GOAD+MLIP vs DFT, one panel per DFT functional.\n"
            "Reads dft_binding_energies_all.csv (needs 'functional' column)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dft", default="dft_binding_energies_all.csv",
        help="Multi-functional DFT CSV with 'functional' column "
             "(default: dft_binding_energies_all.csv)"
    )
    parser.add_argument(
        "--ml", default="workflow/summary.csv",
        help="GOAD summary CSV (default: workflow/summary.csv)"
    )
    parser.add_argument(
        "--calculators", nargs="+", default=["sevennet_omni"],
        metavar="CALC",
        help="ML calculator(s) to plot. Default: sevennet_omni"
    )
    parser.add_argument(
        "--functionals", nargs="+", default=None,
        metavar="FUNC",
        help=(
            "DFT functionals to include (default: all found in --dft CSV). "
            "Accepted: pbe  pbe_d3  r2scan  beef_vdw"
        )
    )
    parser.add_argument(
        "--output", default="results/dft_vs_mlip_all_functionals.png",
        help="Output PNG (default: results/dft_vs_mlip_all_functionals.png)"
    )
    parser.add_argument(
        "--csv-out", default=None,
        help="Also write all matched pairs to this CSV"
    )
    args = parser.parse_args()

    dft_path = Path(args.dft)
    ml_path  = Path(args.ml)

    for p in [dft_path, ml_path]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            raise SystemExit(1)

    print(f"DFT CSV    : {dft_path}")
    print(f"ML CSV     : {ml_path}")
    print(f"Calculators: {[CALC_LABELS.get(c, c) for c in args.calculators]}")
    print()

    dft_data = load_dft_all(dft_path)
    ml_data  = load_ml_best(ml_path, args.calculators)

    # Determine which functionals to plot
    if args.functionals:
        functionals = [normalise_func(f) for f in args.functionals]
    else:
        # Use whatever is in the CSV, in a preferred display order
        preferred = ["pbe", "pbe_d3", "r2scan", "beef_vdw"]
        functionals = [f for f in preferred if f in dft_data]
        extras = [f for f in dft_data if f not in functionals]
        functionals += sorted(extras)

    if not functionals:
        print("ERROR: No functionals found in DFT CSV.")
        raise SystemExit(1)

    print(f"Functionals: {[FUNC_LABELS.get(f, f) for f in functionals]}")
    print()

    # Build matched pairs per functional × calculator
    calc_pairs_per_func = {}
    all_csv_rows = []

    for func in functionals:
        func_dft = dft_data.get(func, {})
        calc_pairs = {}
        for calc in args.calculators:
            common = sorted(set(func_dft) & set(ml_data[calc]))
            calc_pairs[calc] = common
            ml_only  = len(ml_data[calc]) - len(common)
            dft_only = len(func_dft) - len(common)
            label    = CALC_LABELS.get(calc, calc)
            fl       = FUNC_LABELS.get(func, func)
            print(f"[{fl} | {label}]  matched={len(common)}"
                  f"  DFT-only={dft_only}  ML-only={ml_only}")

            if common:
                mae, rmse, bias, r2 = compute_stats(func_dft, ml_data[calc], common)
                print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}  "
                      f"bias={bias:+.4f}  R\u00b2={r2:.4f}")
                for (surf, mol) in common:
                    all_csv_rows.append({
                        "functional": func,
                        "calculator": calc,
                        "surface":    surf,
                        "molecule":   mol,
                        "E_ads_DFT":  f"{func_dft[(surf,mol)]:.6f}",
                        "E_ads_ML":   f"{ml_data[calc][(surf,mol)]:.6f}",
                        "diff_eV":    f"{ml_data[calc][(surf,mol)]-func_dft[(surf,mol)]:.6f}",
                    })
            print()

        calc_pairs_per_func[func] = calc_pairs

    if not any(p for cp in calc_pairs_per_func.values()
               for p in cp.values()):
        print("No matched pairs found for any functional/calculator combination.")
        raise SystemExit(1)

    make_figure(functionals, calc_pairs_per_func, dft_data, ml_data,
                args.calculators, Path(args.output))

    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        fields = ["functional", "calculator", "surface", "molecule",
                  "E_ads_DFT", "E_ads_ML", "diff_eV"]
        with Path(args.csv_out).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"CSV saved : {args.csv_out}  ({len(all_csv_rows)} rows)")


if __name__ == "__main__":
    main()
