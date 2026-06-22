#!/usr/bin/env python
"""
plot_results.py
===============
Generate publication-quality figures from workflow/summary.csv.

Figures produced
----------------
1. figures/heatmap_mean_Eads.png
   Mean E_ads (eV) heatmap — surface (rows) vs adsorbate (cols)

2. figures/heatmap_std_Eads.png
   Seed-to-seed std dev heatmap — shows reproducibility across seeds

3. figures/barplot_by_surface.png
   Grouped bar chart: mean E_ads per adsorbate, one panel per surface

4. figures/barplot_by_adsorbate.png
   Grouped bar chart: mean E_ads per surface, one panel per adsorbate

Usage
-----
    python plot_results.py
    python plot_results.py --csv workflow/summary.csv --out figures/
    python plot_results.py --finished-only   # skip 'running' rows
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Molecule display order (weakest → strongest binder, roughly)
# ---------------------------------------------------------------------------
MOL_ORDER = [
    "CO2", "ethene", "ethane", "propane", "propene",
    "ethanol", "propanol", "isopropanol", "glycerol",
]

# Surface display order (grouped by metal)
SURF_ORDER = [
    "Cu111", "Cu110", "Cu001",
    "Pt111", "Pt110", "Pt100",
    "Pd111", "Pd110", "Pd100",
    # remaining metals if present
]

# Colour palette for bar charts (one colour per metal)
METAL_COLOURS = {
    "Cu": "#E07B54",
    "Pt": "#5B8DB8",
    "Pd": "#6BAE6E",
    "Ni": "#A97BB5",
    "Ag": "#B5A642",
    "Au": "#D4A843",
}


def surface_colour(surf):
    metal = surf[:2]
    return METAL_COLOURS.get(metal, "#888888")


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------

def load(csv_path, finished_only=False):
    df = pd.read_csv(csv_path)
    df["E_ads_eV"] = pd.to_numeric(df["E_ads_eV"], errors="coerce")
    if finished_only:
        df = df[df["state"] == "finished"]
    df = df.dropna(subset=["E_ads_eV"])

    # Keep only recognised molecules and surfaces
    df = df[df["adsorbate"].isin(MOL_ORDER)]

    # Sort surface / adsorbate into display order
    surf_present  = [s for s in SURF_ORDER if s in df["surface"].unique()]
    extra_surfs   = sorted(set(df["surface"].unique()) - set(SURF_ORDER))
    surf_order    = surf_present + extra_surfs

    mol_present   = [m for m in MOL_ORDER if m in df["adsorbate"].unique()]
    extra_mols    = sorted(set(df["adsorbate"].unique()) - set(MOL_ORDER))
    mol_order     = mol_present + extra_mols

    return df, surf_order, mol_order


# ---------------------------------------------------------------------------
# Figure 1 & 2 — Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmap(pivot, title, cmap, out_path, vcenter=None, fmt=".2f"):
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.9),
                                    max(4, len(pivot.index) * 0.55)))

    vmin, vmax = pivot.values.min(), pivot.values.max()
    if vcenter is not None and vmin < vcenter < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                   norm=norm,
                   vmin=None if norm else vmin,
                   vmax=None if norm else vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                txt_col = "white" if abs(val) > 0.7 * abs(vmin) else "black"
                ax.text(j, i, format(val, fmt), ha="center", va="center",
                        fontsize=7, color=txt_col)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("E$_{ads}$ (eV)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Adsorbate", fontsize=10)
    ax.set_ylabel("Surface", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Bar chart: one panel per surface
# ---------------------------------------------------------------------------

def plot_bars_by_surface(agg, surf_order, mol_order, out_path):
    ncols = 3
    nrows = -(-len(surf_order) // ncols)   # ceiling division
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.2),
                             sharey=False)
    axes = axes.flatten()

    for idx, surf in enumerate(surf_order):
        ax = axes[idx]
        sub = agg[agg["surface"] == surf].set_index("adsorbate")
        mols   = [m for m in mol_order if m in sub.index]
        means  = [sub.loc[m, "mean"] for m in mols]
        stds   = [sub.loc[m, "std"]  for m in mols]

        x = np.arange(len(mols))
        bars = ax.bar(x, means, yerr=stds, capsize=3,
                      color=surface_colour(surf), alpha=0.85,
                      error_kw={"elinewidth": 1.0, "ecolor": "0.3"})

        ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
        ax.set_title(surf, fontweight="bold", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(mols, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("E$_{ads}$ (eV)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", labelsize=7)

    # Hide unused panels
    for idx in range(len(surf_order), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Mean Adsorption Energy by Surface\n(error bars = seed std dev)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Bar chart: one panel per adsorbate
# ---------------------------------------------------------------------------

def plot_bars_by_adsorbate(agg, surf_order, mol_order, out_path):
    ncols = 3
    nrows = -(-len(mol_order) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.2),
                             sharey=False)
    axes = axes.flatten()

    for idx, mol in enumerate(mol_order):
        ax = axes[idx]
        sub = agg[agg["adsorbate"] == mol].set_index("surface")
        surfs  = [s for s in surf_order if s in sub.index]
        means  = [sub.loc[s, "mean"] for s in surfs]
        stds   = [sub.loc[s, "std"]  for s in surfs]
        colors = [surface_colour(s) for s in surfs]

        x = np.arange(len(surfs))
        ax.bar(x, means, yerr=stds, capsize=3,
               color=colors, alpha=0.85,
               error_kw={"elinewidth": 1.0, "ecolor": "0.3"})

        ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
        ax.set_title(mol, fontweight="bold", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(surfs, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("E$_{ads}$ (eV)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", labelsize=7)

    for idx in range(len(mol_order), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Mean Adsorption Energy by Adsorbate\n(error bars = seed std dev)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot adsorption energy results from workflow/summary.csv"
    )
    parser.add_argument("--csv", default="workflow/summary.csv",
                        help="Path to summary CSV (default: workflow/summary.csv)")
    parser.add_argument("--out", default="figures",
                        help="Output directory for figures (default: figures/)")
    parser.add_argument("--finished-only", action="store_true",
                        help="Exclude rows with state != 'finished'")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.csv}")
    df, surf_order, mol_order = load(args.csv, finished_only=args.finished_only)
    print(f"  Surfaces:   {surf_order}")
    print(f"  Adsorbates: {mol_order}")
    print(f"  Rows:       {len(df)}")
    print()

    # Aggregate over seeds
    agg = (df.groupby(["surface", "adsorbate"])["E_ads_eV"]
             .agg(mean="mean", std="std", count="count")
             .reset_index())

    # ---- Figure 1: Mean heatmap ----
    print("Generating figures...")
    pivot_mean = (agg.pivot(index="surface", columns="adsorbate", values="mean")
                     .reindex(index=surf_order, columns=mol_order))
    plot_heatmap(
        pivot_mean,
        title="Mean Adsorption Energy E$_{ads}$ (eV)\nSevenNet-omni | Cu/Pt/Pd surfaces",
        cmap="RdYlBu",
        vcenter=0.0,
        out_path=out_dir / "heatmap_mean_Eads.png",
    )

    # ---- Figure 2: Std heatmap ----
    pivot_std = (agg.pivot(index="surface", columns="adsorbate", values="std")
                    .reindex(index=surf_order, columns=mol_order))
    plot_heatmap(
        pivot_std,
        title="Seed-to-Seed Std Dev of E$_{ads}$ (eV)\n(lower = more reproducible)",
        cmap="YlOrRd",
        vcenter=None,
        out_path=out_dir / "heatmap_std_Eads.png",
        fmt=".3f",
    )

    # ---- Figure 3: Bar per surface ----
    plot_bars_by_surface(agg, surf_order, mol_order,
                         out_dir / "barplot_by_surface.png")

    # ---- Figure 4: Bar per adsorbate ----
    plot_bars_by_adsorbate(agg, surf_order, mol_order,
                           out_dir / "barplot_by_adsorbate.png")

    print()
    print("Done! Figures saved to:", out_dir)


if __name__ == "__main__":
    main()
