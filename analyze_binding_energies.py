#!/usr/bin/env python
"""
analyze_binding_energies.py
============================
Comprehensive comparison of DFT vs SevenNet-OMNI adsorption (binding) energies.

Reads
-----
- dft_binding_energies.csv      DFT reference data
- workflow/summary.csv          SevenNet GOAD run results

For each surface+molecule pair the *best* (lowest) SevenNet E_ads across
all finished seeds is selected and compared against the DFT reference.

Outputs
-------
figures/parity_plot.png       – overall DFT vs SevenNet parity plot
figures/mae_by_surface.png    – horizontal bar chart: MAE per surface
figures/mae_by_molecule.png   – horizontal bar chart: MAE per molecule
figures/parity_by_surface.png – 3×3 grid of per-surface parity plots
results/comparison_table.csv  – merged dataset with errors

Usage
-----
    python analyze_binding_energies.py
    python analyze_binding_energies.py \\
        --dft  dft_binding_energies.csv \\
        --ml   workflow/summary.csv \\
        --figures-dir figures \\
        --results-dir results
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # fall back to default if neither style is available

# Per-surface colours (colour-blind-friendly palette)
SURFACE_COLORS = {
    "Cu001": "#D55E00",
    "Cu110": "#E69F00",
    "Cu111": "#F0E442",
    "Pd100": "#009E73",
    "Pd110": "#56B4E9",
    "Pd111": "#0072B2",
    "Pt100": "#CC79A7",
    "Pt110": "#999999",
    "Pt111": "#000000",
}

# Per-molecule markers
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
# Data loading
# ---------------------------------------------------------------------------

def load_dft(path: Path) -> pd.DataFrame:
    """Load DFT binding energies, keeping only rows with status == 'ok'."""
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df = df.rename(columns={"E_ads": "DFT_E_ads"})
    df["DFT_E_ads"] = pd.to_numeric(df["DFT_E_ads"], errors="coerce")
    return df[["surface", "molecule", "DFT_E_ads"]].dropna()


def load_sevennet_best(path: Path) -> pd.DataFrame:
    """
    Load SevenNet results, filter for finished sevennet_omni runs, and return
    the best seed (minimum E_ads_eV) per surface+adsorbate combination.

    Returns a DataFrame with columns:
        surface, molecule, SevenNet_E_ads, best_seed
    """
    df = pd.read_csv(path)

    # Keep only finished sevennet_omni runs
    mask = (
        (df["calculator"].str.strip() == "sevennet_omni")
        & (df["state"].str.strip() == "finished")
    )
    df = df[mask].copy()
    df["E_ads_eV"] = pd.to_numeric(df["E_ads_eV"], errors="coerce")
    df = df.dropna(subset=["E_ads_eV"])

    # For each surface+adsorbate pick the seed with the lowest E_ads
    idx = df.groupby(["surface", "adsorbate"])["E_ads_eV"].idxmin()
    best = df.loc[idx, ["surface", "adsorbate", "E_ads_eV", "seed"]].copy()
    best = best.rename(columns={
        "adsorbate": "molecule",
        "E_ads_eV": "SevenNet_E_ads",
        "seed": "best_seed",
    })
    best["best_seed"] = best["best_seed"].astype(int)
    return best.reset_index(drop=True)


def merge_data(dft: pd.DataFrame, ml: pd.DataFrame) -> pd.DataFrame:
    """
    Merge DFT and SevenNet DataFrames on (surface, molecule).

    Adds columns:
        difference      SevenNet_E_ads − DFT_E_ads
        absolute_error  |difference|
    """
    merged = pd.merge(dft, ml, on=["surface", "molecule"], how="inner")
    merged["difference"] = merged["SevenNet_E_ads"] - merged["DFT_E_ads"]
    merged["absolute_error"] = merged["difference"].abs()
    return merged.sort_values(["surface", "molecule"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    """Return MAE, RMSE, and R² for the full merged DataFrame."""
    errors = df["difference"].values
    mae   = float(np.mean(np.abs(errors)))
    rmse  = float(np.sqrt(np.mean(errors ** 2)))
    r2    = float(np.corrcoef(df["DFT_E_ads"], df["SevenNet_E_ads"])[0, 1] ** 2)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _parity_line(ax, lo: float, hi: float) -> None:
    """Draw a y = x diagonal reference line."""
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, zorder=2, label="y = x")


def _scatter_parity(ax, df: pd.DataFrame, color_col: str = "surface",
                    marker_col: str = "molecule") -> None:
    """
    Scatter-plot DFT (x) vs SevenNet (y) using surface colours and
    molecule markers.
    """
    for _, row in df.iterrows():
        color  = SURFACE_COLORS.get(row[color_col], "grey")
        marker = MOLECULE_MARKERS.get(row[marker_col], "o")
        ax.scatter(
            row["DFT_E_ads"], row["SevenNet_E_ads"],
            color=color, marker=marker,
            s=70, alpha=0.85, linewidths=0.5, edgecolors="k", zorder=3,
        )


def _annotate_stats(ax, df: pd.DataFrame, fontsize: int = 9) -> None:
    """Add MAE / RMSE / R² text box to an axes."""
    stats = compute_stats(df)
    text  = (
        f"MAE  = {stats['mae']:.3f} eV\n"
        f"RMSE = {stats['rmse']:.3f} eV\n"
        f"$R^2$   = {stats['r2']:.3f}"
    )
    ax.text(
        0.04, 0.96, text,
        transform=ax.transAxes, va="top", ha="left",
        fontsize=fontsize, family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )


def _axis_limits(df: pd.DataFrame, pad: float = 0.1):
    """Return symmetric (lo, hi) limits covering both DFT and SevenNet values."""
    all_vals = pd.concat([df["DFT_E_ads"], df["SevenNet_E_ads"]])
    lo = float(all_vals.min()) - pad
    hi = float(all_vals.max()) + pad
    return lo, hi


# ---------------------------------------------------------------------------
# Plot 1: overall parity plot
# ---------------------------------------------------------------------------

def plot_parity(df: pd.DataFrame, output_path: Path) -> None:
    """
    Scatter plot of DFT E_ads (x) vs SevenNet E_ads (y) for all matched
    surface+molecule pairs.  Points are coloured by surface and shaped by
    molecule type.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    _scatter_parity(ax, df)

    lo, hi = _axis_limits(df)
    _parity_line(ax, lo, hi)
    _annotate_stats(ax, df)

    # ---- Legends ----
    # Surface colour legend
    surfaces_present = df["surface"].unique()
    surf_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=SURFACE_COLORS.get(s, "grey"),
            markeredgecolor="k", markersize=8, label=s,
        )
        for s in sorted(surfaces_present)
    ]
    leg1 = ax.legend(
        handles=surf_handles, title="Surface",
        loc="lower right", fontsize=8, title_fontsize=8,
    )

    # Molecule marker legend
    mols_present = df["molecule"].unique()
    mol_handles = [
        plt.Line2D(
            [0], [0], marker=MOLECULE_MARKERS.get(m, "o"), color="w",
            markerfacecolor="grey", markeredgecolor="k",
            markersize=8, label=m,
        )
        for m in sorted(mols_present)
    ]
    ax.legend(
        handles=mol_handles, title="Molecule",
        loc="upper left", fontsize=8, title_fontsize=8,
    )
    ax.add_artist(leg1)

    ax.set_xlabel("DFT  $E_{ads}$ (eV)", fontsize=12)
    ax.set_ylabel("SevenNet-OMNI  $E_{ads}$ (eV)", fontsize=12)
    ax.set_title("DFT vs SevenNet-OMNI Adsorption Energies", fontsize=13)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 2: MAE by surface (horizontal bar chart)
# ---------------------------------------------------------------------------

def plot_mae_by_surface(df: pd.DataFrame, output_path: Path) -> None:
    """Horizontal bar chart of MAE per surface, sorted ascending."""
    mae_surf = (
        df.groupby("surface")["absolute_error"]
        .mean()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(mae_surf))))
    colors = [SURFACE_COLORS.get(s, "steelblue") for s in mae_surf.index]
    ax.barh(mae_surf.index, mae_surf.values, color=colors, edgecolor="k",
            linewidth=0.6)

    # Value labels
    for i, (val, label) in enumerate(zip(mae_surf.values, mae_surf.index)):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel("MAE (eV)", fontsize=11)
    ax.set_title("Mean Absolute Error by Surface", fontsize=12)
    ax.set_xlim(0, mae_surf.max() * 1.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 3: MAE by molecule (horizontal bar chart)
# ---------------------------------------------------------------------------

def plot_mae_by_molecule(df: pd.DataFrame, output_path: Path) -> None:
    """Horizontal bar chart of MAE per molecule type, sorted ascending."""
    mae_mol = (
        df.groupby("molecule")["absolute_error"]
        .mean()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(mae_mol))))
    mol_colors = plt.cm.tab10(np.linspace(0, 1, len(mae_mol)))
    ax.barh(mae_mol.index, mae_mol.values, color=mol_colors, edgecolor="k",
            linewidth=0.6)

    for i, val in enumerate(mae_mol.values):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel("MAE (eV)", fontsize=11)
    ax.set_title("Mean Absolute Error by Molecule", fontsize=12)
    ax.set_xlim(0, mae_mol.max() * 1.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot 4: 3×3 per-surface parity grid
# ---------------------------------------------------------------------------

def plot_parity_by_surface(df: pd.DataFrame, output_path: Path) -> None:
    """
    A 3×3 grid of parity plots, one per surface.  Each panel shows DFT vs
    SevenNet E_ads for all molecules adsorbed on that surface.
    """
    surfaces = sorted(df["surface"].unique())
    n_cols   = 3
    n_rows   = int(np.ceil(len(surfaces) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    for idx, surface in enumerate(surfaces):
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        ax      = axes[row_idx][col_idx]

        sub = df[df["surface"] == surface]
        color = SURFACE_COLORS.get(surface, "steelblue")

        # Plot each molecule
        for _, r in sub.iterrows():
            marker = MOLECULE_MARKERS.get(r["molecule"], "o")
            ax.scatter(
                r["DFT_E_ads"], r["SevenNet_E_ads"],
                color=color, marker=marker,
                s=60, alpha=0.85, linewidths=0.5, edgecolors="k", zorder=3,
            )

        lo, hi = _axis_limits(sub)
        _parity_line(ax, lo, hi)
        _annotate_stats(ax, sub, fontsize=7)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(surface, fontsize=11)
        ax.set_xlabel("DFT (eV)", fontsize=9)
        ax.set_ylabel("SevenNet (eV)", fontsize=9)

        # Molecule legend inside each panel
        mol_handles = [
            plt.Line2D(
                [0], [0], marker=MOLECULE_MARKERS.get(m, "o"), color="w",
                markerfacecolor=color, markeredgecolor="k",
                markersize=6, label=m,
            )
            for m in sorted(sub["molecule"].unique())
        ]
        ax.legend(handles=mol_handles, fontsize=6, loc="upper left",
                  title="Molecule", title_fontsize=6)

    # Hide unused subplots
    for idx in range(len(surfaces), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Per-Surface Parity Plots: DFT vs SevenNet-OMNI", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    header = (
        f"{'Surface':<10} {'Molecule':<15} {'DFT (eV)':>10} "
        f"{'SevenNet (eV)':>15} {'Diff (eV)':>11} {'Best seed':>10}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for _, row in df.iterrows():
        print(
            f"{row['surface']:<10} {row['molecule']:<15} "
            f"{row['DFT_E_ads']:>10.4f} {row['SevenNet_E_ads']:>15.4f} "
            f"{row['difference']:>11.4f} {int(row['best_seed']):>10d}"
        )
    print(sep)

    stats = compute_stats(df)
    print(
        f"\nSummary over {len(df)} pairs:\n"
        f"  MAE  = {stats['mae']:.4f} eV\n"
        f"  RMSE = {stats['rmse']:.4f} eV\n"
        f"  R²   = {stats['r2']:.4f}\n"
    )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_comparison_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save the merged comparison table to a CSV file."""
    out_cols = [
        "surface", "molecule",
        "DFT_E_ads", "SevenNet_E_ads",
        "best_seed", "difference", "absolute_error",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(output_path, index=False, float_format="%.6f")
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyse DFT vs SevenNet adsorption energies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dft",
        default="dft_binding_energies.csv",
        help="Path to DFT binding energies CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--ml",
        default="workflow/summary.csv",
        help="Path to SevenNet summary CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Output directory for PNG figures (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Output directory for CSV results (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    """Main entry point: load data, generate plots, print and save results."""
    args      = parse_args()
    dft_path  = Path(args.dft)
    ml_path   = Path(args.ml)
    fig_dir   = Path(args.figures_dir)
    res_dir   = Path(args.results_dir)

    # Validate inputs
    for p in [dft_path, ml_path]:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    # Create output directories
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load -----
    print("Loading DFT data …")
    dft_df = load_dft(dft_path)
    print(f"  {len(dft_df)} DFT entries (status=ok)")

    print("Loading SevenNet data …")
    ml_df = load_sevennet_best(ml_path)
    print(f"  {len(ml_df)} SevenNet best-seed entries (finished)")

    # ----- Merge -----
    merged = merge_data(dft_df, ml_df)
    dft_only = len(dft_df) - len(merged)
    ml_only  = len(ml_df)  - len(merged)
    print(
        f"\nMatched {len(merged)} surface+molecule pairs "
        f"(DFT-only: {dft_only}, SevenNet-only: {ml_only})\n"
    )

    if merged.empty:
        print("No overlapping pairs — nothing to plot.")
        return

    # ----- Summary table -----
    print_summary(merged)

    # ----- Plots -----
    plot_parity(merged,           fig_dir / "parity_plot.png")
    plot_mae_by_surface(merged,   fig_dir / "mae_by_surface.png")
    plot_mae_by_molecule(merged,  fig_dir / "mae_by_molecule.png")
    plot_parity_by_surface(merged, fig_dir / "parity_by_surface.png")

    # ----- Results CSV -----
    save_comparison_csv(merged, res_dir / "comparison_table.csv")


if __name__ == "__main__":
    main()
