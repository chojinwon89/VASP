#!/usr/bin/env python
"""
validate_experiment.py
======================
Compare GOAD+ML and DFT adsorption energies against experimental
reference values (TPD / SCAC).

Reads
-----
  experimental_references.csv  — curated experimental E_ads values
  dft_binding_energies_all.csv — DFT results (multi-functional)
  workflow/summary.csv         — GOAD+ML results

Outputs
-------
  Figure 1 (--output-parity):  parity plot  — predicted vs experiment
                                one panel per method (GOAD+SevenNet,
                                GOAD+5m, DFT functionals)
  Figure 2 (--output-bar):     bar chart    — per-system comparison of
                                all methods against experiment
  CSV      (--csv-out):        full numerical comparison table

Usage
-----
    python validate_experiment.py

    python validate_experiment.py \\
        --exp   experimental_references.csv \\
        --dft   dft_binding_energies_all.csv \\
        --ml    workflow/summary.csv \\
        --dft-functionals pbe_d3 r2scan beef_vdw \\
        --ml-calcs sevennet_omni 5m \\
        --output-parity results/exp_validation_parity.png \\
        --output-bar    results/exp_validation_bar.png \\
        --csv-out       results/exp_validation.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    # ML calculators
    "sevennet_omni": dict(label="GOAD+SevenNet",  color="#E05C00", marker="o",
                          ls="-",  lw=1.4, ms=8,  fill="full"),
    "5m":            dict(label="GOAD+MatterSim", color="#0072B2", marker="s",
                          ls="-",  lw=1.4, ms=8,  fill="none"),
    # DFT functionals
    "pbe":           dict(label="DFT (PBE)",       color="#888888", marker="^",
                          ls="--", lw=1.0, ms=7,  fill="full"),
    "pbe_d3":        dict(label="DFT (PBE+D3)",    color="#009E73", marker="^",
                          ls="--", lw=1.0, ms=7,  fill="none"),
    "r2scan":        dict(label="DFT (r\u00b2SCAN)",    color="#CC79A7", marker="D",
                          ls="--", lw=1.0, ms=7,  fill="full"),
    "beef_vdw":      dict(label="DFT (BEEF-vdW)",  color="#F0A500", marker="D",
                          ls="--", lw=1.0, ms=7,  fill="none"),
}

FUNCTIONAL_ALIASES = {"beef_dfw": "beef_vdw", "pbed3": "pbe_d3"}

EADS_MIN = -5.0
EADS_MAX =  2.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_experimental(path: Path) -> dict:
    """Returns {(surface, molecule): {E_ads, technique, reference}}"""
    data = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row["surface"].strip(), row["molecule"].strip())
            try:
                data[key] = {
                    "E_ads":     float(row["E_ads_eV"]),
                    "technique": row.get("technique", "").strip(),
                    "reference": row.get("reference", "").strip(),
                }
            except ValueError:
                pass
    return data


def load_dft(path: Path, functionals: list) -> dict:
    """Returns {functional: {(surface, molecule): E_ads_eV}}"""
    data = defaultdict(dict)
    with path.open() as f:
        reader = csv.DictReader(f)
        has_func = "functional" in (reader.fieldnames or [])
        for row in reader:
            if row.get("status", "").strip() != "ok":
                continue
            func = row["functional"].strip().lower() if has_func else "pbe"
            func = FUNCTIONAL_ALIASES.get(func, func)
            if functionals and func not in functionals:
                continue
            key = (row["surface"].strip(), row["molecule"].strip())
            try:
                e = float(row["E_ads"])
                if EADS_MIN <= e <= EADS_MAX:
                    data[func][key] = e
            except ValueError:
                pass
    return dict(data)


def load_ml(path: Path, calculators: list) -> dict:
    """Returns {calc: {(surface, adsorbate): best_E_ads_eV}}"""
    best = {c: defaultdict(lambda: float("inf")) for c in calculators}
    with path.open() as f:
        for row in csv.DictReader(f):
            calc = row.get("calculator", "").strip()
            if calc not in calculators:
                continue
            if row.get("state", "").strip() != "finished":
                continue
            key = (row["surface"].strip(), row["adsorbate"].strip())
            try:
                e = float(row["E_ads_eV"])
                if e < best[calc][key]:
                    best[calc][key] = e
            except ValueError:
                pass
    return {c: {k: v for k, v in d.items() if v != float("inf")}
            for c, d in best.items()}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats(exp_data, pred_vals, keys):
    e = np.array([exp_data[k]["E_ads"] for k in keys])
    p = np.array([pred_vals[k]         for k in keys])
    mae  = float(np.mean(np.abs(p - e)))
    rmse = float(np.sqrt(np.mean((p - e) ** 2)))
    bias = float(np.mean(p - e))
    r2   = float(np.corrcoef(e, p)[0, 1] ** 2) if len(keys) > 1 else float("nan")
    return mae, rmse, bias, r2


# ---------------------------------------------------------------------------
# Figure 1 — Parity plot (one panel per method)
# ---------------------------------------------------------------------------

def make_parity(exp_data, dft_data, ml_data,
                dft_functionals, ml_calcs, output_path: Path):
    """One panel per method. X = experiment, Y = predicted."""
    methods = ml_calcs + dft_functionals
    n = len(methods)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5 * nrows))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten()

    all_comparison = []

    for i, method in enumerate(methods):
        ax = axes_flat[i]
        style  = METHOD_STYLES.get(method, {})
        label  = style.get("label", method)
        color  = style.get("color", "grey")
        marker = style.get("marker", "o")
        fill   = style.get("fill", "full")
        ms     = style.get("ms", 8)

        pred = ml_data.get(method, {}) if method in ml_calcs else dft_data.get(method, {})
        matched = sorted(set(exp_data) & set(pred))

        if not matched:
            ax.set_title(f"{label}\n(no matched data)", fontsize=9)
            continue

        exp_vals  = np.array([exp_data[k]["E_ads"] for k in matched])
        pred_vals = np.array([pred[k]               for k in matched])

        for k, ex, pr in zip(matched, exp_vals, pred_vals):
            fc = color if fill == "full" else "none"
            ax.scatter(ex, pr, facecolors=fc, edgecolors=color,
                       marker=marker, s=ms ** 2 * 0.8,
                       linewidths=1.3, zorder=3, alpha=0.9)
            ax.annotate(f"{k[0]}\n{k[1]}",
                        xy=(ex, pr), fontsize=5.5, color="#333333",
                        xytext=(3, 3), textcoords="offset points")

        lo = min(exp_vals.min(), pred_vals.min()) - 0.1
        hi = max(exp_vals.max(), pred_vals.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=2)
        ax.fill_between([lo, hi],
                        [lo - 0.15, hi - 0.15],
                        [lo + 0.15, hi + 0.15],
                        color="grey", alpha=0.08, zorder=1)

        mae, rmse, bias, r2 = stats(exp_data, pred, matched)
        stat_txt = (f"N={len(matched)}\n"
                    f"MAE  = {mae:.3f} eV\n"
                    f"RMSE = {rmse:.3f} eV\n"
                    f"bias = {bias:+.3f} eV\n"
                    f"R\u00b2   = {r2:.3f}")
        ax.text(0.97, 0.03, stat_txt,
                transform=ax.transAxes, va="bottom", ha="right",
                fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="lightgrey", alpha=0.92))

        ax.set_title(label, fontsize=10, fontweight="bold", pad=5)
        ax.set_xlabel("Experiment  $E_{ads}$ (eV)", fontsize=9)
        ax.set_ylabel("Predicted  $E_{ads}$ (eV)",  fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.tick_params(labelsize=8)

        for k, ex, pr in zip(matched, exp_vals, pred_vals):
            all_comparison.append({
                "method":    method,
                "surface":   k[0],
                "molecule":  k[1],
                "E_exp_eV":  f"{ex:.4f}",
                "E_pred_eV": f"{pr:.4f}",
                "diff_eV":   f"{pr - ex:+.4f}",
                "technique": exp_data[k]["technique"],
                "reference": exp_data[k]["reference"],
            })

    for j in range(len(methods), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Validation Against Experiment (TPD / SCAC)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Parity figure saved: {output_path}")
    plt.close(fig)

    return all_comparison


# ---------------------------------------------------------------------------
# Figure 2 — Bar chart (per system, all methods side by side)
# ---------------------------------------------------------------------------

def make_bar(exp_data, dft_data, ml_data,
             dft_functionals, ml_calcs, output_path: Path):
    """One group of bars per system. Experiment shown as diamond marker."""
    methods = ml_calcs + dft_functionals

    all_pred = {}
    for m in methods:
        all_pred[m] = ml_data.get(m, {}) if m in ml_calcs else dft_data.get(m, {})

    systems = sorted(
        k for k in exp_data
        if any(k in all_pred[m] for m in methods)
    )

    if not systems:
        print("Bar chart: no matched systems found.")
        return

    n_sys  = len(systems)
    n_meth = len(methods)
    bar_w  = 0.8 / n_meth
    x      = np.arange(n_sys)

    fig_w = max(12, n_sys * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    fig.patch.set_facecolor("white")

    for mi, method in enumerate(methods):
        style = METHOD_STYLES.get(method, {})
        label = style.get("label", method)
        color = style.get("color", "grey")
        fill  = style.get("fill", "full")
        hatch = "" if fill == "full" else "///"
        src   = all_pred[method]

        vals = [src[k] if k in src else np.nan for k in systems]
        offset = (mi - n_meth / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, width=bar_w * 0.92,
               label=label, color=color, alpha=0.82,
               hatch=hatch, edgecolor="white", linewidth=0.5)

    # Experiment markers
    exp_vals = [exp_data[k]["E_ads"] for k in systems]
    ax.scatter(x, exp_vals, color="black", marker="D",
               s=55, zorder=5, label="Experiment", clip_on=False)
    ax.errorbar(x, exp_vals, yerr=0.10,
                fmt="none", ecolor="black", elinewidth=1.2,
                capsize=4, zorder=4, alpha=0.7)

    xlabels = [f"{k[0]}\n{k[1]}" for k in systems]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7.5, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="-", alpha=0.4)
    ax.set_ylabel("$E_{ads}$ (eV)", fontsize=11)
    ax.set_title("Adsorption Energy: Predicted Methods vs Experiment",
                 fontsize=12, fontweight="bold", pad=8)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9,
              edgecolor="lightgrey", ncol=2)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(7))
    ax.tick_params(axis="y", labelsize=9)

    # MAE summary box
    mae_lines = []
    for method in methods:
        src   = all_pred[method]
        match = [k for k in systems if k in src]
        if match:
            mae, _, bias, r2 = stats(exp_data, src, match)
            lbl = METHOD_STYLES.get(method, {}).get("label", method)
            mae_lines.append(f"{lbl}: MAE={mae:.3f}  bias={bias:+.3f}  R\u00b2={r2:.3f}")

    ax.text(0.01, 0.99, "\n".join(mae_lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="lightgrey", alpha=0.92))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Bar chart saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate GOAD+ML and DFT against experimental adsorption energies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--exp",   default="experimental_references.csv")
    parser.add_argument("--dft",   default="dft_binding_energies_all.csv")
    parser.add_argument("--ml",    default="workflow/summary.csv")
    parser.add_argument("--dft-functionals", nargs="+",
                        default=["pbe", "pbe_d3", "r2scan", "beef_vdw"],
                        metavar="FUNC")
    parser.add_argument("--ml-calcs", nargs="+",
                        default=["sevennet_omni", "5m"],
                        metavar="CALC")
    parser.add_argument("--output-parity",
                        default="results/exp_validation_parity.png")
    parser.add_argument("--output-bar",
                        default="results/exp_validation_bar.png")
    parser.add_argument("--csv-out",
                        default="results/exp_validation.csv")
    args = parser.parse_args()

    for p in [Path(args.exp), Path(args.dft), Path(args.ml)]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            raise SystemExit(1)

    print(f"Experimental refs : {args.exp}")
    print(f"DFT CSV           : {args.dft}")
    print(f"ML CSV            : {args.ml}")
    print(f"DFT functionals   : {args.dft_functionals}")
    print(f"ML calculators    : {args.ml_calcs}")
    print()

    exp_data = load_experimental(Path(args.exp))
    dft_data = load_dft(Path(args.dft), args.dft_functionals)
    ml_data  = load_ml(Path(args.ml),   args.ml_calcs)

    print(f"Experimental systems loaded : {len(exp_data)}")
    for func in args.dft_functionals:
        n     = len(dft_data.get(func, {}))
        match = len(set(exp_data) & set(dft_data.get(func, {})))
        print(f"  DFT {func:<12}: {n} systems, {match} matched to experiment")
    for calc in args.ml_calcs:
        n     = len(ml_data.get(calc, {}))
        match = len(set(exp_data) & set(ml_data.get(calc, {})))
        print(f"  ML  {calc:<14}: {n} systems, {match} matched to experiment")
    print()

    Path(args.output_parity).parent.mkdir(parents=True, exist_ok=True)
    comparison = make_parity(
        exp_data, dft_data, ml_data,
        args.dft_functionals, args.ml_calcs,
        Path(args.output_parity)
    )

    Path(args.output_bar).parent.mkdir(parents=True, exist_ok=True)
    make_bar(
        exp_data, dft_data, ml_data,
        args.dft_functionals, args.ml_calcs,
        Path(args.output_bar)
    )

    if comparison:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.csv_out).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "method", "surface", "molecule",
                "E_exp_eV", "E_pred_eV", "diff_eV",
                "technique", "reference"
            ])
            writer.writeheader()
            writer.writerows(comparison)
        print(f"CSV saved: {args.csv_out}")

    # Summary table
    print()
    print("=" * 65)
    print(f"{'Method':<22} {'N':>3} {'MAE':>8} {'RMSE':>8} "
          f"{'Bias':>8} {'R\u00b2':>7}")
    print("-" * 65)
    for method in args.ml_calcs + args.dft_functionals:
        src = (ml_data.get(method, {}) if method in args.ml_calcs
               else dft_data.get(method, {}))
        matched = sorted(set(exp_data) & set(src))
        if not matched:
            lbl = METHOD_STYLES.get(method, {}).get("label", method)
            print(f"  {lbl:<20} {'---':>3}")
            continue
        mae, rmse, bias, r2 = stats(exp_data, src, matched)
        lbl = METHOD_STYLES.get(method, {}).get("label", method)
        print(f"  {lbl:<20} {len(matched):>3} {mae:>8.3f} {rmse:>8.3f} "
              f"{bias:>+8.3f} {r2:>7.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
