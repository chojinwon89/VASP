#!/usr/bin/env python
"""
plot_dft_vs_sevennet.py
=======================
Compare DFT adsorption energies with SevenNet-OMNI predictions.

Reads:
  - dft_binding_energies.csv   (from calc_binding_energy.py)
  - workflow/summary.csv       (from GOAD runs, sevennet_omni calculator)

For each surface+molecule pair, takes the best (lowest) SevenNet E_ads
across all seeds and plots it against the DFT reference.

Output: dft_vs_sevennet.png  (and optionally dft_vs_sevennet.csv)

Usage
-----
    python plot_dft_vs_sevennet.py
    python plot_dft_vs_sevennet.py --dft dft_binding_energies.csv \\
                                   --ml  workflow/summary.csv \\
                                   --output dft_vs_sevennet.png
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dft(path: Path) -> dict:
    """
    Returns {(surface, molecule): E_ads_eV} for status=='ok' rows.
    """
    data = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "ok":
                continue
            key = (row["surface"], row["molecule"])
            try:
                data[key] = float(row["E_ads"])
            except ValueError:
                pass
    return data


def load_sevennet_best(path: Path) -> dict:
    """
    Returns {(surface, molecule): best_E_ads_eV} — minimum E_ads over all
    seeds for calculator == sevennet_omni with status == finished.
    """
    best = defaultdict(lambda: float("inf"))
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("calculator", "").strip() != "sevennet_omni":
                continue
            if row.get("state", row.get("status", "")).strip() != "finished":
                continue
            key = (row["surface"], row["adsorbate"])
            try:
                e = float(row["E_ads_eV"])
                if e < best[key]:
                    best[key] = e
            except ValueError:
                pass
    # Remove unset entries
    return {k: v for k, v in best.items() if v != float("inf")}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

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

METAL_COLORS = {
    "Cu": "#D55E00",   # orange-red
    "Pt": "#0072B2",   # blue
    "Pd": "#009E73",   # green
    "Ni": "#CC79A7",   # pink
    "Ag": "#F0E442",   # yellow
    "Au": "#56B4E9",   # light blue
}


def make_plot(pairs, dft_vals, ml_vals, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Collect range for parity line
    all_vals = []

    # Plot each point
    for (surf, mol) in pairs:
        metal = surf[:2]
        color = METAL_COLORS.get(metal, "grey")
        marker = MOLECULE_MARKERS.get(mol, "o")
        x = dft_vals[(surf, mol)]
        y = ml_vals[(surf, mol)]
        all_vals.extend([x, y])
        ax.scatter(x, y, color=color, marker=marker,
                   s=80, alpha=0.85, linewidths=0.5, edgecolors="k",
                   zorder=3)

    if not all_vals:
        print("No data to plot.")
        return

    # Parity line
    lo = min(all_vals) - 0.1
    hi = max(all_vals) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x", zorder=2)

    # MAE / RMSE annotation
    dft_arr = np.array([dft_vals[k] for k in pairs])
    ml_arr  = np.array([ml_vals[k]  for k in pairs])
    mae  = np.mean(np.abs(ml_arr - dft_arr))
    rmse = np.sqrt(np.mean((ml_arr - dft_arr) ** 2))
    r2   = np.corrcoef(dft_arr, ml_arr)[0, 1] ** 2

    ax.text(0.04, 0.96,
            f"MAE  = {mae:.3f} eV\nRMSE = {rmse:.3f} eV\n$R^2$   = {r2:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # ---- Legends ----
    # Metal colour legend
    metal_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markeredgecolor="k",
                   markersize=8, label=m)
        for m, c in METAL_COLORS.items()
        if any(k[0].startswith(m) for k in pairs)
    ]
    leg1 = ax.legend(handles=metal_handles, title="Metal",
                     loc="lower right", fontsize=8, title_fontsize=8)

    # Molecule marker legend
    mol_handles = [
        plt.Line2D([0], [0], marker=mk, color="w",
                   markerfacecolor="grey", markeredgecolor="k",
                   markersize=8, label=mol)
        for mol, mk in MOLECULE_MARKERS.items()
        if any(k[1] == mol for k in pairs)
    ]
    ax.legend(handles=mol_handles, title="Molecule",
              loc="upper left", fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)

    ax.set_xlabel("DFT  $E_{ads}$ (eV)", fontsize=12)
    ax.set_ylabel("SevenNet-OMNI  $E_{ads}$ (eV)", fontsize=12)
    ax.set_title("DFT vs SevenNet-OMNI Adsorption Energies", fontsize=13)
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
        description="Plot DFT vs SevenNet-OMNI adsorption energies."
    )
    parser.add_argument("--dft",    default="dft_binding_energies.csv")
    parser.add_argument("--ml",     default="workflow/summary.csv")
    parser.add_argument("--output", default="dft_vs_sevennet.png")
    parser.add_argument("--csv-out", default=None,
                        help="Also save matched pairs to this CSV")
    args = parser.parse_args()

    dft_path = Path(args.dft)
    ml_path  = Path(args.ml)

    for p in [dft_path, ml_path]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            raise SystemExit(1)

    dft_data = load_dft(dft_path)
    ml_data  = load_sevennet_best(ml_path)

    # Find common pairs
    common = sorted(set(dft_data) & set(ml_data))
    if not common:
        print("No matching surface+molecule pairs found between DFT and ML data.")
        print(f"  DFT keys  (first 5): {list(dft_data)[:5]}")
        print(f"  ML keys   (first 5): {list(ml_data)[:5]}")
        raise SystemExit(1)

    print(f"Matched {len(common)} surface+molecule pairs")
    print(f"  DFT-only  : {len(dft_data) - len(common)}")
    print(f"  ML-only   : {len(ml_data)  - len(common)}")
    print()

    # Print table
    print(f"{'System':<30} {'DFT (eV)':>10} {'SevenNet (eV)':>14} {'Diff (eV)':>10}")
    print("-" * 68)
    for (surf, mol) in common:
        d = dft_data[(surf, mol)]
        m = ml_data[(surf, mol)]
        print(f"{surf+'_'+mol:<30} {d:>10.4f} {m:>14.4f} {m-d:>10.4f}")
    print()

    make_plot(common, dft_data, ml_data, Path(args.output))

    # Optional CSV
    if args.csv_out:
        with Path(args.csv_out).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["surface", "molecule", "E_ads_DFT_eV",
                             "E_ads_SevenNet_eV", "diff_eV"])
            for (surf, mol) in common:
                d = dft_data[(surf, mol)]
                m = ml_data[(surf, mol)]
                writer.writerow([surf, mol, f"{d:.6f}", f"{m:.6f}", f"{m-d:.6f}"])
        print(f"CSV saved: {args.csv_out}")


if __name__ == "__main__":
    main()
