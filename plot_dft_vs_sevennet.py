#!/usr/bin/env python
"""
plot_dft_vs_sevennet.py
=======================
Compare GOAD+MLIP adsorption energies against DFT calculations across
multiple DFT functionals.

Produces a 2×2 parity-plot grid — one panel per DFT functional — each
showing all ML calculators overlaid, with per-panel MAE / RMSE / R² / bias
stats computed only on the data within the plot window.

Point style encoding:
  - Colour  → metal surface  (Cu/Pt/Pd/Ni/Ag/Au)
  - Marker  → molecule
  - Fill    → calculator  (SevenNet = filled, MatterSim = hollow)

Legends (Calculator, Metal, Molecule) always shown outside panels.

Reads
-----
  dft_binding_energies_all.csv   (from calc_binding_energy.py --all-functionals)
  workflow/summary.csv           (from GOAD runs)

Output
------
  PNG : 2×2 subplot figure
  CSV : optional flat table of matched pairs (after all filters)

Usage
-----
    python plot_dft_vs_sevennet.py \\
        --calculators sevennet_omni 5m \\
        --functionals pbe pbe_d3 beef_vdw r2scan \\
        --output results/dft_vs_mlip_all.png \\
        --csv-out results/dft_vs_mlip_all.csv

    # Exclude pairs where |E_ads_DFT - E_ads_ML| > 5 eV (default)
    python plot_dft_vs_sevennet.py --max-diff 5.0

    # Override axis limits
    python plot_dft_vs_sevennet.py --axis-min -3.0 --axis-max 0.5
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

# filled = solid marker, none = hollow marker
CALC_FILL = {
    "sevennet_omni": "full",
    "5m":            "none",
    "5m_d3":         "none",
    "1m":            "none",
}

CALC_LINEWIDTH = {
    "sevennet_omni": 0.5,
    "5m":            1.4,
    "5m_d3":         1.4,
    "1m":            1.4,
}

CALC_COLORS = {
    "sevennet_omni": "#E05C00",
    "5m":            "#0072B2",
    "5m_d3":         "#009E73",
    "1m":            "#CC79A7",
}

FUNC_ORDER = ["pbe", "pbe_d3", "beef_vdw", "r2scan"]

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
    "ethene":      "*",
    "CO2":         "h",
    "isobutene":   "+",
    "1-butene":    "x",
    "butadiene":   "<",
    "isoprene":    ">",
    "benzene":     "H",
    "toluene":     "8",
}

DEFAULT_AXIS_MIN = -2.5
DEFAULT_AXIS_MAX =  0.3
DEFAULT_MAX_DIFF =  5.0   # eV  — pairs with |DFT - ML| > this are excluded
BAND_WIDTH       =  0.2


# ---------------------------------------------------------------------------
# Functional name normalisation
# ---------------------------------------------------------------------------

def normalise_func(name: str) -> str:
    s = name.strip().lower().replace("+", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    aliases = {
        "beef_dfw": "beef_vdw",
        "beefvdw":  "beef_vdw",
        "pbed3":    "pbe_d3",
        "r_2scan":  "r2scan",
    }
    return aliases.get(s, s)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_dft_all(path: Path) -> dict:
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
# Diff filter
# ---------------------------------------------------------------------------

def apply_max_diff_filter(dft_data: dict, ml_data: dict,
                           max_diff: float) -> tuple:
    """
    Returns filtered copies of dft_data and ml_data.

    A (surface, molecule) key is removed from dft_data[func] if, for ANY
    calculator, |E_ads_DFT - E_ads_ML| > max_diff.  This keeps the DFT
    dataset consistent across calculators within each functional.

    Prints a summary of how many pairs were removed per functional.
    """
    if max_diff is None or max_diff <= 0:
        return dft_data, ml_data

    filtered_dft = {}
    total_removed = 0

    for func, dft_vals in dft_data.items():
        keep = {}
        removed = []
        for key, e_dft in dft_vals.items():
            drop = False
            for calc_vals in ml_data.values():
                e_ml = calc_vals.get(key)
                if e_ml is not None and abs(e_dft - e_ml) > max_diff:
                    drop = True
                    break
            if drop:
                removed.append((key, e_dft))
            else:
                keep[key] = e_dft

        if removed:
            fl = FUNC_LABELS.get(func, func)
            print(f"  [{fl}] removed {len(removed)} pair(s) with |diff| > {max_diff} eV:")
            for (surf, mol), e in removed:
                print(f"    {surf}_{mol}  E_DFT={e:.3f} eV")
            total_removed += len(removed)

        filtered_dft[func] = keep

    if total_removed:
        print(f"  Total removed: {total_removed} pair(s)\n")
    else:
        print(f"  No pairs exceeded max-diff={max_diff} eV\n")

    return filtered_dft, ml_data


# ---------------------------------------------------------------------------
# Stats (in-window only)
# ---------------------------------------------------------------------------

def compute_stats(dft_vals, ml_vals, pairs, axis_min, axis_max):
    dft_arr = np.array([dft_vals[k] for k in pairs])
    ml_arr  = np.array([ml_vals[k]  for k in pairs])
    in_w = (
        (dft_arr >= axis_min) & (dft_arr <= axis_max) &
        (ml_arr  >= axis_min) & (ml_arr  <= axis_max)
    )
    n_clip = int((~in_w).sum())
    dft_w, ml_w = dft_arr[in_w], ml_arr[in_w]
    if len(dft_w) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), len(dft_w), n_clip
    mae  = float(np.mean(np.abs(ml_w - dft_w)))
    rmse = float(np.sqrt(np.mean((ml_w - dft_w) ** 2)))
    bias = float(np.mean(ml_w - dft_w))
    ss_res = np.sum((ml_w - dft_w) ** 2)
    ss_tot = np.sum((dft_w - dft_w.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return mae, rmse, bias, r2, int(in_w.sum()), n_clip


# ---------------------------------------------------------------------------
# Single panel
# ---------------------------------------------------------------------------

def _plot_panel(ax, func, func_label, calc_pairs, dft_vals, ml_data,
                axis_min, axis_max):
    n_plotted = 0

    for calc, pairs in calc_pairs.items():
        fill = CALC_FILL.get(calc, "full")
        lw   = CALC_LINEWIDTH.get(calc, 0.5)

        for (surf, mol) in pairs:
            x = dft_vals[func].get((surf, mol))
            y = ml_data[calc].get((surf, mol))
            if x is None or y is None:
                continue
            if x < axis_min or x > axis_max or y < axis_min or y > axis_max:
                continue

            metal  = surf[:2] if len(surf) >= 2 else surf
            mcolor = METAL_COLORS.get(metal, "grey")
            marker = MOLECULE_MARKERS.get(mol, "o")

            if fill == "none":
                ax.scatter(x, y,
                           facecolors="none",
                           edgecolors=mcolor,
                           marker=marker, s=70,
                           linewidths=lw,
                           alpha=0.9, zorder=3)
            else:
                ax.scatter(x, y,
                           facecolors=mcolor,
                           edgecolors="k",
                           marker=marker, s=70,
                           linewidths=0.4,
                           alpha=0.85, zorder=3)
            n_plotted += 1

    if n_plotted == 0:
        ax.set_title(f"DFT: {func_label}  (no data)", fontsize=11)
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal")
        return

    ax.plot([axis_min, axis_max], [axis_min, axis_max], "k--", lw=1.2, zorder=2)
    ax.fill_between(
        [axis_min, axis_max],
        [axis_min - BAND_WIDTH, axis_max - BAND_WIDTH],
        [axis_min + BAND_WIDTH, axis_max + BAND_WIDTH],
        color="grey", alpha=0.12, zorder=1
    )

    stat_lines = []
    for calc, pairs in calc_pairs.items():
        matched = [(s, m) for (s, m) in pairs
                   if (s, m) in dft_vals.get(func, {})
                   and (s, m) in ml_data.get(calc, {})]
        if not matched:
            continue
        mae, rmse, bias, r2, n_in, n_clip = compute_stats(
            dft_vals[func], ml_data[calc], matched, axis_min, axis_max)
        label = CALC_LABELS.get(calc, calc)
        clip_note = f" ({n_clip} clipped)" if n_clip else ""
        stat_lines.append(
            f"[{label}]  n={n_in}{clip_note}\n"
            f"  MAE={mae:.3f}  RMSE={rmse:.3f}\n"
            f"  R\u00b2={r2:.3f}  bias={bias:+.3f}"
        )

    ax.text(0.97, 0.03, "\n".join(stat_lines),
            transform=ax.transAxes, va="bottom", ha="right",
            fontsize=6.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_title(f"DFT: {func_label}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"DFT ({func_label})  $E_{{ads}}$ (eV)", fontsize=9)


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def make_figure(functionals, calc_pairs_per_func, dft_data, ml_data,
                calculators, output_path: Path, axis_min, axis_max,
                max_diff):

    n = len(functionals)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7.5 * ncols, 7 * nrows),
                             squeeze=False)

    ml_label = " / ".join(CALC_LABELS.get(c, c) for c in calculators)

    for idx, func in enumerate(functionals):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        func_label = FUNC_LABELS.get(func, func.upper())
        _plot_panel(ax, func, func_label,
                    calc_pairs_per_func.get(func, {}),
                    dft_data, ml_data, axis_min, axis_max)
        ax.set_ylabel(f"{ml_label}  $E_{{ads}}$ (eV)", fontsize=9)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    all_pairs = [p for fp in calc_pairs_per_func.values()
                 for pairs in fp.values() for p in pairs]

    # Calculator legend (upper left)
    calc_handles = []
    for calc in calculators:
        fill  = CALC_FILL.get(calc, "full")
        label = CALC_LABELS.get(calc, calc)
        if fill == "none":
            h = plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="none",
                           markeredgecolor="#555555",
                           markeredgewidth=1.4, markersize=9, label=label)
        else:
            h = plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor="#888888",
                           markeredgecolor="k",
                           markersize=9, label=label)
        calc_handles.append(h)

    fig.legend(handles=calc_handles, title="Calculator",
               loc="upper left", bbox_to_anchor=(0.01, 1.02),
               ncol=len(calc_handles), fontsize=9, title_fontsize=9,
               frameon=True)

    # Metal legend (upper right)
    metal_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markeredgecolor="k",
                   markersize=9, label=m)
        for m, c in METAL_COLORS.items()
        if any(k[0].startswith(m) for k in all_pairs)
    ]
    fig.legend(handles=metal_handles, title="Metal",
               loc="upper right", bbox_to_anchor=(0.99, 1.02),
               ncol=len(metal_handles), fontsize=9, title_fontsize=9,
               frameon=True)

    # Molecule legend (bottom centre)
    mol_handles = [
        plt.Line2D([0], [0], marker=mk, color="w",
                   markerfacecolor="grey", markeredgecolor="k",
                   markersize=9, label=mol)
        for mol, mk in MOLECULE_MARKERS.items()
        if any(k[1] == mol for k in all_pairs)
    ]
    if mol_handles:
        fig.legend(handles=mol_handles, title="Molecule",
                   loc="lower center", bbox_to_anchor=(0.5, -0.04),
                   ncol=min(len(mol_handles), 6),
                   fontsize=8, title_fontsize=8, frameon=True)

    diff_str = f"|diff| ≤ {max_diff} eV" if max_diff else "no diff filter"
    window_str = f"axis window: [{axis_min:.1f}, {axis_max:.1f}] eV"
    fig.suptitle(
        f"DFT vs {ml_label}  —  Adsorption Energies (all functionals)\n"
        f"{window_str}   |   {diff_str}",
        fontsize=12, fontweight="bold", y=1.04
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
            "Colour = metal, marker = molecule, fill = calculator.\n"
            "SevenNet = filled; MatterSim = hollow.\n"
            "Use --max-diff to exclude pairs where |E_DFT - E_ML| exceeds a threshold."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dft", default="dft_binding_energies_all.csv")
    parser.add_argument("--ml",  default="workflow/summary.csv")
    parser.add_argument("--calculators", nargs="+", default=["sevennet_omni"],
                        metavar="CALC")
    parser.add_argument(
        "--functionals", nargs="+", default=None, metavar="FUNC",
        help="DFT functionals to plot (default: pbe pbe_d3 beef_vdw r2scan)"
    )
    parser.add_argument(
        "--axis-min", type=float, default=DEFAULT_AXIS_MIN,
        help=f"Shared axis minimum in eV (default: {DEFAULT_AXIS_MIN})"
    )
    parser.add_argument(
        "--axis-max", type=float, default=DEFAULT_AXIS_MAX,
        help=f"Shared axis maximum in eV (default: {DEFAULT_AXIS_MAX})"
    )
    parser.add_argument(
        "--max-diff", type=float, default=DEFAULT_MAX_DIFF,
        help=(
            f"Exclude pairs where |E_ads_DFT - E_ads_ML| > this value (eV). "
            f"Default: {DEFAULT_MAX_DIFF}. Set to 0 to disable."
        )
    )
    parser.add_argument(
        "--output", default="results/dft_vs_mlip_all_functionals.png"
    )
    parser.add_argument("--csv-out", default=None)
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
    print(f"Axis window: [{args.axis_min:.1f}, {args.axis_max:.1f}] eV")
    print(f"Max diff   : {args.max_diff} eV  ({'disabled' if not args.max_diff else 'active'})")
    print()

    dft_data = load_dft_all(dft_path)
    ml_data  = load_ml_best(ml_path, args.calculators)

    # Apply |diff| filter before anything else
    print("Applying max-diff filter...")
    dft_data, ml_data = apply_max_diff_filter(dft_data, ml_data, args.max_diff)

    if args.functionals:
        functionals = [normalise_func(f) for f in args.functionals]
    else:
        functionals = [f for f in FUNC_ORDER if f in dft_data]
        extras = [f for f in dft_data if f not in functionals]
        functionals += sorted(extras)

    if not functionals:
        print("ERROR: No functionals found in DFT CSV.")
        raise SystemExit(1)

    print(f"Functionals (panel order): {[FUNC_LABELS.get(f, f) for f in functionals]}")
    print()

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
            label = CALC_LABELS.get(calc, calc)
            fl    = FUNC_LABELS.get(func, func)
            print(f"[{fl} | {label}]  matched={len(common)}"
                  f"  DFT-only={dft_only}  ML-only={ml_only}")

            if common:
                mae, rmse, bias, r2, n_in, n_clip = compute_stats(
                    func_dft, ml_data[calc], common,
                    args.axis_min, args.axis_max)
                print(f"  In window: {n_in}  Clipped: {n_clip}")
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

    if not any(p for cp in calc_pairs_per_func.values() for p in cp.values()):
        print("No matched pairs found.")
        raise SystemExit(1)

    make_figure(functionals, calc_pairs_per_func, dft_data, ml_data,
                args.calculators, Path(args.output),
                args.axis_min, args.axis_max, args.max_diff)

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
