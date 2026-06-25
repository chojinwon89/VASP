#!/usr/bin/env python
"""
plot_dft_vs_sevennet.py
=======================
Compare DFT adsorption energies with one or more ML calculator predictions.

Supported calculator names (as they appear in workflow/summary.csv):
    sevennet_omni   -- SevenNet-OMNI
    5m              -- MatterSim 5M
    5m_d3           -- MatterSim 5M + D3
    1m              -- MatterSim 1M

Reads:
  - dft_binding_energies.csv   (from calc_binding_energy.py)
  - workflow/summary.csv       (from GOAD runs)

For each surface+molecule pair, takes the best (lowest) E_ads across all
finished seeds per calculator and plots against the DFT reference.

Running jobs (state != finished) are automatically skipped.

Usage
-----
    # SevenNet-OMNI only (default)
    python plot_dft_vs_sevennet.py

    # MatterSim 5M only
    python plot_dft_vs_sevennet.py --calculators 5m --output dft_vs_5m.png

    # Both on one parity plot
    python plot_dft_vs_sevennet.py \\
        --calculators sevennet_omni 5m \\
        --output dft_vs_both.png \\
        --csv-out results/comparison_both.csv
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Display name mapping  (CSV value -> human-readable label)
# ---------------------------------------------------------------------------
CALC_LABELS = {
    "sevennet_omni": "SevenNet-OMNI",
    "5m":            "MatterSim-5M",
    "5m_d3":         "MatterSim-5M+D3",
    "1m":            "MatterSim-1M",
}

# Per-calculator fill style: SevenNet = filled, MatterSim = hollow
CALC_FILL = {
    "sevennet_omni": "full",
    "5m":            "none",
    "5m_d3":         "none",
    "1m":            "none",
}

# Hollow markers need a thicker edge to be visible
CALC_LINEWIDTH = {
    "sevennet_omni": 0.5,
    "5m":            1.2,
    "5m_d3":         1.2,
    "1m":            1.2,
}

# ---------------------------------------------------------------------------
# Color / marker dictionaries
# ---------------------------------------------------------------------------

METAL_COLORS = {
    "Cu": "#D55E00",   # orange-red
    "Pt": "#0072B2",   # blue
    "Pd": "#009E73",   # green
    "Ni": "#CC79A7",   # pink
    "Ag": "#F0E442",   # yellow
    "Au": "#56B4E9",   # light blue
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


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_dft(path: Path) -> dict:
    """Returns {(surface, molecule): E_ads_eV} for status=='ok' rows."""
    data = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "").strip() != "ok":
                continue
            key = (row["surface"].strip(), row["molecule"].strip())
            try:
                data[key] = float(row["E_ads"])
            except ValueError:
                pass
    return data


def load_ml_best(path: Path, calculators: list[str]) -> dict:
    """
    Returns {calc_name: {(surface, adsorbate): best_E_ads_eV}}
    Only includes rows with state == 'finished'.
    """
    best = {c: defaultdict(lambda: float("inf")) for c in calculators}
    skipped_running = 0

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            calc = row.get("calculator", "").strip()
            if calc not in calculators:
                continue
            state = row.get("state", row.get("status", "")).strip()
            if state != "finished":
                skipped_running += 1
                continue
            key = (row["surface"].strip(), row["adsorbate"].strip())
            try:
                e = float(row["E_ads_eV"])
                if e < best[calc][key]:
                    best[calc][key] = e
            except ValueError:
                pass

    if skipped_running:
        print(f"  Skipped {skipped_running} row(s) with state != finished (still running).")

    return {c: {k: v for k, v in d.items() if v != float("inf")}
            for c, d in best.items()}


# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def compute_stats(dft_vals, ml_vals, pairs):
    dft_arr = np.array([dft_vals[k] for k in pairs])
    ml_arr  = np.array([ml_vals[k]  for k in pairs])
    mae  = float(np.mean(np.abs(ml_arr - dft_arr)))
    rmse = float(np.sqrt(np.mean((ml_arr - dft_arr) ** 2)))
    r2   = float(np.corrcoef(dft_arr, ml_arr)[0, 1] ** 2)
    return mae, rmse, r2


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(calc_pairs: dict, dft_vals: dict, ml_data: dict,
              output_path: Path):
    """
    calc_pairs : {calc_name: [(surface, molecule), ...]}
    dft_vals   : {(surface, molecule): dft_E}
    ml_data    : {calc_name: {(surface, molecule): ml_E}}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    all_vals = []

    # ---- Scatter points ----
    for calc, pairs in calc_pairs.items():
        fill = CALC_FILL.get(calc, "full")
        lw   = CALC_LINEWIDTH.get(calc, 0.5)
        for (surf, mol) in pairs:
            metal  = surf[:2]
            color  = METAL_COLORS.get(metal, "grey")
            marker = MOLECULE_MARKERS.get(mol, "o")
            x = dft_vals[(surf, mol)]
            y = ml_data[calc][(surf, mol)]
            all_vals.extend([x, y])

            if fill == "none":
                ax.scatter(x, y, facecolors="none", edgecolors=color,
                           marker=marker, s=80, linewidths=lw,
                           alpha=0.9, zorder=3)
            else:
                ax.scatter(x, y, color=color, marker=marker,
                           s=80, alpha=0.85, linewidths=lw,
                           edgecolors="k", zorder=3)

    if not all_vals:
        print("No data to plot.")
        return

    # ---- Parity line ----
    lo = min(all_vals) - 0.1
    hi = max(all_vals) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, zorder=2)

    # ---- Stats annotation (top-right, one line per calculator) ----
    stat_lines = []
    for calc, pairs in calc_pairs.items():
        if not pairs:
            continue
        mae, rmse, r2 = compute_stats(dft_vals, ml_data[calc], pairs)
        label = CALC_LABELS.get(calc, calc)
        stat_lines.append(
            f"[{label}]  MAE={mae:.3f}  RMSE={rmse:.3f}  $R^2$={r2:.3f}"
        )

    ax.text(0.96, 0.96, "\n".join(stat_lines),
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # ---- Calculator fill-style legend (top-left) ----
    calc_handles = []
    for calc, pairs in calc_pairs.items():
        if not pairs:
            continue
        fill  = CALC_FILL.get(calc, "full")
        label = CALC_LABELS.get(calc, calc)
        if fill == "none":
            h = plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="none", markeredgecolor="grey",
                           markeredgewidth=1.2, markersize=9, label=label)
        else:
            h = plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="grey", markeredgecolor="k",
                           markersize=9, label=label)
        calc_handles.append(h)

    leg_calc = ax.legend(handles=calc_handles, title="Calculator",
                         loc="upper left", fontsize=8, title_fontsize=8)

    # ---- Metal color legend (lower right) ----
    all_pairs = [p for pairs in calc_pairs.values() for p in pairs]
    metal_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markeredgecolor="k",
                   markersize=8, label=m)
        for m, c in METAL_COLORS.items()
        if any(k[0].startswith(m) for k in all_pairs)
    ]
    leg_metal = ax.legend(handles=metal_handles, title="Metal",
                          loc="lower right", fontsize=8, title_fontsize=8)

    # ---- Molecule marker legend (lower left) ----
    mol_handles = [
        plt.Line2D([0], [0], marker=mk, color="w",
                   markerfacecolor="grey", markeredgecolor="k",
                   markersize=8, label=mol)
        for mol, mk in MOLECULE_MARKERS.items()
        if any(k[1] == mol for k in all_pairs)
    ]
    leg_mol = ax.legend(handles=mol_handles, title="Molecule",
                        loc="lower left", fontsize=8, title_fontsize=8)

    ax.add_artist(leg_calc)
    ax.add_artist(leg_metal)
    ax.add_artist(leg_mol)

    # Titles and formatting
    calc_labels_str = " vs ".join(CALC_LABELS.get(c, c) for c in calc_pairs)
    ax.set_xlabel("DFT  $E_{ads}$ (eV)", fontsize=12)
    ax.set_ylabel(f"{calc_labels_str}  $E_{{ads}}$ (eV)", fontsize=12)
    ax.set_title(f"DFT vs {calc_labels_str} Adsorption Energies", fontsize=13)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot DFT vs ML adsorption energies (one or more calculators).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Calculator names (as stored in workflow/summary.csv):\n"
            "  sevennet_omni  -- SevenNet-OMNI\n"
            "  5m             -- MatterSim 5M\n"
            "  5m_d3          -- MatterSim 5M + D3\n"
            "  1m             -- MatterSim 1M\n"
        ),
    )
    parser.add_argument("--dft", default="dft_binding_energies.csv",
                        help="DFT reference CSV (default: dft_binding_energies.csv)")
    parser.add_argument("--ml",  default="workflow/summary.csv",
                        help="GOAD summary CSV (default: workflow/summary.csv)")
    parser.add_argument("--calculators", nargs="+",
                        default=["sevennet_omni"],
                        metavar="CALC",
                        help=(
                            "Calculator name(s) to plot. Default: sevennet_omni. "
                            "Examples:\n"
                            "  --calculators 5m\n"
                            "  --calculators sevennet_omni 5m"
                        ))
    parser.add_argument("--output", default="dft_vs_sevennet.png",
                        help="Output PNG path (default: dft_vs_sevennet.png)")
    parser.add_argument("--csv-out", default=None,
                        help="Also save matched pairs to this CSV")
    args = parser.parse_args()

    dft_path = Path(args.dft)
    ml_path  = Path(args.ml)

    for p in [dft_path, ml_path]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            raise SystemExit(1)

    print(f"Calculators: {[CALC_LABELS.get(c, c) for c in args.calculators]}")
    print()

    dft_data = load_dft(dft_path)
    ml_data  = load_ml_best(ml_path, args.calculators)

    # Build per-calculator matched pairs
    calc_pairs = {}
    for calc in args.calculators:
        common   = sorted(set(dft_data) & set(ml_data[calc]))
        calc_pairs[calc] = common
        ml_only  = len(ml_data[calc]) - len(common)
        dft_only = len(dft_data) - len(common)
        label    = CALC_LABELS.get(calc, calc)
        print(f"[{label}]  matched={len(common)}  DFT-only={dft_only}  ML-only={ml_only}")

        if not common:
            print(f"  WARNING: No matching pairs — check calculator name in summary.csv.")
            continue

        mae, rmse, r2 = compute_stats(dft_data, ml_data[calc], common)
        print(f"  MAE  = {mae:.4f} eV")
        print(f"  RMSE = {rmse:.4f} eV")
        print(f"  R²   = {r2:.4f}")
        print()
        print(f"  {'System':<30} {'DFT (eV)':>10} {'ML (eV)':>10} {'Diff (eV)':>10}")
        print("  " + "-" * 64)
        for (surf, mol) in common:
            d = dft_data[(surf, mol)]
            m = ml_data[calc][(surf, mol)]
            print(f"  {surf+'_'+mol:<30} {d:>10.4f} {m:>10.4f} {m-d:>10.4f}")
        print()

    if all(len(p) == 0 for p in calc_pairs.values()):
        print("No data to plot for any calculator.")
        raise SystemExit(1)

    make_plot(calc_pairs, dft_data, ml_data, Path(args.output))

    # Optional CSV — all calculators merged
    if args.csv_out:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.csv_out).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["calculator", "surface", "molecule",
                             "E_ads_DFT_eV", "E_ads_ML_eV", "diff_eV"])
            for calc, pairs in calc_pairs.items():
                for (surf, mol) in pairs:
                    d = dft_data[(surf, mol)]
                    m = ml_data[calc][(surf, mol)]
                    writer.writerow([calc, surf, mol,
                                     f"{d:.6f}", f"{m:.6f}", f"{m-d:.6f}"])
        print(f"CSV saved: {args.csv_out}")


if __name__ == "__main__":
    main()
