#!/usr/bin/env python
"""
validate_experiment.py
======================
Compare GOAD+ML and DFT adsorption energies against DFT literature
reference values (PBE+D3 from Catalysis-Hub / published papers).

Reads
-----
  dft_literature_references.csv  -- curated DFT literature E_ads values
  dft_binding_energies_all.csv   -- your own DFT results (multi-functional)
  workflow/summary.csv           -- GOAD+ML results

Outputs
-------
  Figure 1 (--output-parity):  parity plot  -- predicted vs DFT literature
                                one panel per method (GOAD+SevenNet,
                                GOAD+5m, your DFT functionals)
  Figure 2 (--output-bar):     bar chart    -- per-system comparison of
                                all methods against DFT literature
  CSV      (--csv-out):        full numerical comparison table

Usage
-----
    python validate_experiment.py

    python validate_experiment.py \\
        --ref  dft_literature_references.csv \\
        --dft  dft_binding_energies_all.csv \\
        --ml   workflow/summary.csv \\
        --ref-functional pbe_d3 \\
        --dft-functionals pbe_d3 r2scan beef_vdw \\
        --ml-calcs sevennet_omni 5m \\
        --output-parity results/lit_validation_parity.png \\
        --output-bar    results/lit_validation_bar.png \\
        --csv-out       results/lit_validation.csv
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

# Pre-define unicode characters so they can be used safely inside f-strings
# on Python < 3.12 (backslash escapes are not allowed inside f-string braces).
R2_SYM  = "R\u00b2"   # R²
SUP2    = "\u00b2"     # superscript 2 (used in r²SCAN label)


# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    # ML calculators
    "sevennet_omni": dict(label="GOAD+SevenNet",  color="#E05C00", marker="o",
                          ls="-",  lw=1.4, ms=8,  fill="full"),
    "5m":            dict(label="GOAD+MatterSim", color="#0072B2", marker="s",
                          ls="-",  lw=1.4, ms=8,  fill="none"),
    # DFT functionals (your own runs)
    "pbe":           dict(label="DFT (PBE)",            color="#888888", marker="^",
                          ls="--", lw=1.0, ms=7,  fill="full"),
    "pbe_d3":        dict(label="DFT (PBE+D3)",         color="#009E73", marker="^",
                          ls="--", lw=1.0, ms=7,  fill="none"),
    "r2scan":        dict(label="DFT (r" + SUP2 + "SCAN)", color="#CC79A7", marker="D",
                          ls="--", lw=1.0, ms=7,  fill="full"),
    "beef_vdw":      dict(label="DFT (BEEF-vdW)",       color="#F0A500", marker="D",
                          ls="--", lw=1.0, ms=7,  fill="none"),
}

FUNCTIONAL_ALIASES = {"beef_dfw": "beef_vdw", "pbed3": "pbe_d3"}

EADS_MIN = -5.0
EADS_MAX =  2.0


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_references(path: Path, functional_filter=None) -> dict:
    """
    Load DFT literature reference values.
    Returns {(surface, molecule): {E_ads, functional, reference}}
    If functional_filter is set, only rows matching that functional are loaded.
    """
    data = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            func = row.get("functional", "").strip().lower()
            func = FUNCTIONAL_ALIASES.get(func, func)
            if functional_filter and func != functional_filter:
                continue
            key = (row["surface"].strip(), row["molecule"].strip())
            try:
                data[key] = {
                    "E_ads":      float(row["E_ads_eV"]),
                    "functional": func,
                    "reference":  row.get("reference", "").strip(),
                    "note":       row.get("note", "").strip(),
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

def stats(ref_data, pred_vals, keys):
    e = np.array([ref_data[k]["E_ads"] for k in keys])
    p = np.array([pred_vals[k]          for k in keys])
    mae  = float(np.mean(np.abs(p - e)))
    rmse = float(np.sqrt(np.mean((p - e) ** 2)))
    bias = float(np.mean(p - e))
    r2   = float(np.corrcoef(e, p)[0, 1] ** 2) if len(keys) > 1 else float("nan")
    return mae, rmse, bias, r2


# ---------------------------------------------------------------------------
# Figure 1 -- Parity plot (one panel per method)
# ---------------------------------------------------------------------------

def make_parity(ref_data, dft_data, ml_data,
                dft_functionals, ml_calcs, ref_functional, output_path: Path):
    """One panel per method. X = DFT literature, Y = predicted."""
    methods = ml_calcs + dft_functionals
    n     = len(methods)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5 * nrows))
    fig.patch.set_facecolor("white")
    axes_flat = np.array(axes).flatten()

    all_comparison = []
    ref_label = ref_functional.upper().replace("_", "+")

    for i, method in enumerate(methods):
        ax     = axes_flat[i]
        style  = METHOD_STYLES.get(method, {})
        label  = style.get("label", method)
        color  = style.get("color", "grey")
        marker = style.get("marker", "o")
        fill   = style.get("fill", "full")
        ms     = style.get("ms", 8)

        pred    = ml_data.get(method, {}) if method in ml_calcs else dft_data.get(method, {})
        matched = sorted(set(ref_data) & set(pred))

        if not matched:
            ax.set_title(label + "\n(no matched data)", fontsize=9)
            continue

        ref_vals  = np.array([ref_data[k]["E_ads"] for k in matched])
        pred_vals = np.array([pred[k]               for k in matched])

        for k, rv, pv in zip(matched, ref_vals, pred_vals):
            fc = color if fill == "full" else "none"
            ax.scatter(rv, pv, facecolors=fc, edgecolors=color,
                       marker=marker, s=ms ** 2 * 0.8,
                       linewidths=1.3, zorder=3, alpha=0.9)
            ax.annotate(k[0] + "\n" + k[1],
                        xy=(rv, pv), fontsize=5.5, color="#333333",
                        xytext=(3, 3), textcoords="offset points")

        lo = min(ref_vals.min(), pred_vals.min()) - 0.1
        hi = max(ref_vals.max(), pred_vals.max()) + 0.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, zorder=2)
        ax.fill_between([lo, hi],
                        [lo - 0.15, hi - 0.15],
                        [lo + 0.15, hi + 0.15],
                        color="grey", alpha=0.08, zorder=1)

        mae, rmse, bias, r2 = stats(ref_data, pred, matched)
        stat_txt = (
            "N={}\n"
            "MAE  = {:.3f} eV\n"
            "RMSE = {:.3f} eV\n"
            "bias = {:+.3f} eV\n"
            "{} = {:.3f}"
        ).format(len(matched), mae, rmse, bias, R2_SYM, r2)
        ax.text(0.97, 0.03, stat_txt,
                transform=ax.transAxes, va="bottom", ha="right",
                fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="lightgrey", alpha=0.92))

        ax.set_title(label, fontsize=10, fontweight="bold", pad=5)
        ax.set_xlabel("DFT Literature ({})  $E_{{ads}}$ (eV)".format(ref_label),
                      fontsize=9)
        ax.set_ylabel("Predicted  $E_{ads}$ (eV)", fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.tick_params(labelsize=8)

        for k, rv, pv in zip(matched, ref_vals, pred_vals):
            all_comparison.append({
                "method":         method,
                "surface":        k[0],
                "molecule":       k[1],
                "E_ref_eV":       "{:.4f}".format(rv),
                "E_pred_eV":      "{:.4f}".format(pv),
                "diff_eV":        "{:+.4f}".format(pv - rv),
                "ref_functional": ref_data[k]["functional"],
                "reference":      ref_data[k]["reference"],
            })

    for j in range(len(methods), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Validation Against DFT Literature ({})".format(ref_label),
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print("Parity figure saved: {}".format(output_path))
    plt.close(fig)

    return all_comparison


# ---------------------------------------------------------------------------
# Figure 2 -- Bar chart (per system, all methods side by side)
# ---------------------------------------------------------------------------

def make_bar(ref_data, dft_data, ml_data,
             dft_functionals, ml_calcs, ref_functional, output_path: Path):
    """One group of bars per system. DFT literature shown as diamond marker."""
    methods  = ml_calcs + dft_functionals
    all_pred = {
        m: (ml_data.get(m, {}) if m in ml_calcs else dft_data.get(m, {}))
        for m in methods
    }

    systems = sorted(
        k for k in ref_data
        if any(k in all_pred[m] for m in methods)
    )

    if not systems:
        print("Bar chart: no matched systems found.")
        return

    n_sys  = len(systems)
    n_meth = len(methods)
    bar_w  = 0.8 / n_meth
    x      = np.arange(n_sys)

    fig_w = max(14, n_sys * 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    fig.patch.set_facecolor("white")

    for mi, method in enumerate(methods):
        style  = METHOD_STYLES.get(method, {})
        label  = style.get("label", method)
        color  = style.get("color", "grey")
        fill   = style.get("fill", "full")
        hatch  = "" if fill == "full" else "///"
        src    = all_pred[method]
        vals   = [src[k] if k in src else np.nan for k in systems]
        offset = (mi - n_meth / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, width=bar_w * 0.92,
               label=label, color=color, alpha=0.82,
               hatch=hatch, edgecolor="white", linewidth=0.5)

    # DFT literature reference markers
    ref_label = ref_functional.upper().replace("_", "+")
    ref_vals  = [ref_data[k]["E_ads"] for k in systems]
    ax.scatter(x, ref_vals, color="black", marker="D",
               s=55, zorder=5, label="DFT Lit. ({})".format(ref_label),
               clip_on=False)
    # +/-0.05 eV bar representing typical DFT convergence uncertainty
    ax.errorbar(x, ref_vals, yerr=0.05,
                fmt="none", ecolor="black", elinewidth=1.2,
                capsize=4, zorder=4, alpha=0.7)

    xlabels = [k[0] + "\n" + k[1] for k in systems]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7.5, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="-", alpha=0.4)
    ax.set_ylabel("$E_{ads}$ (eV)", fontsize=11)
    ax.set_title(
        "Adsorption Energy: All Methods vs DFT Literature ({})".format(ref_label),
        fontsize=12, fontweight="bold", pad=8
    )
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
            mae, _, bias, r2 = stats(ref_data, src, match)
            lbl = METHOD_STYLES.get(method, {}).get("label", method)
            mae_lines.append(
                "{}: MAE={:.3f}  bias={:+.3f}  {}={:.3f}".format(
                    lbl, mae, bias, R2_SYM, r2)
            )

    ax.text(0.01, 0.99, "\n".join(mae_lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="lightgrey", alpha=0.92))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    print("Bar chart saved: {}".format(output_path))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate GOAD+ML and DFT against DFT literature reference values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ref",  default="dft_literature_references.csv",
                        help="DFT literature reference CSV "
                             "(default: dft_literature_references.csv)")
    parser.add_argument("--dft",  default="dft_binding_energies_all.csv",
                        help="Your own DFT results CSV with functional column")
    parser.add_argument("--ml",   default="workflow/summary.csv",
                        help="GOAD summary CSV")
    parser.add_argument("--ref-functional", default="pbe_d3",
                        help="Functional to use from the reference CSV "
                             "(default: pbe_d3)")
    parser.add_argument("--dft-functionals", nargs="+",
                        default=["pbe", "pbe_d3", "r2scan", "beef_vdw"],
                        metavar="FUNC")
    parser.add_argument("--ml-calcs", nargs="+",
                        default=["sevennet_omni", "5m"],
                        metavar="CALC")
    parser.add_argument("--output-parity",
                        default="results/lit_validation_parity.png")
    parser.add_argument("--output-bar",
                        default="results/lit_validation_bar.png")
    parser.add_argument("--csv-out",
                        default="results/lit_validation.csv")
    args = parser.parse_args()

    for p in [Path(args.ref), Path(args.dft), Path(args.ml)]:
        if not p.exists():
            print("ERROR: {} not found.".format(p))
            raise SystemExit(1)

    print("DFT literature ref : {}  (functional: {})".format(
          args.ref, args.ref_functional))
    print("Your DFT CSV       : {}".format(args.dft))
    print("ML CSV             : {}".format(args.ml))
    print("DFT functionals    : {}".format(args.dft_functionals))
    print("ML calculators     : {}".format(args.ml_calcs))
    print()

    ref_data = load_references(Path(args.ref), args.ref_functional)
    dft_data = load_dft(Path(args.dft), args.dft_functionals)
    ml_data  = load_ml(Path(args.ml),   args.ml_calcs)

    print("DFT literature systems loaded : {}".format(len(ref_data)))
    for func in args.dft_functionals:
        n     = len(dft_data.get(func, {}))
        match = len(set(ref_data) & set(dft_data.get(func, {})))
        print("  Your DFT {:<10}: {} systems, {} matched to literature".format(
              func, n, match))
    for calc in args.ml_calcs:
        n     = len(ml_data.get(calc, {}))
        match = len(set(ref_data) & set(ml_data.get(calc, {})))
        print("  ML  {:<14}: {} systems, {} matched to literature".format(
              calc, n, match))
    print()

    Path(args.output_parity).parent.mkdir(parents=True, exist_ok=True)
    comparison = make_parity(
        ref_data, dft_data, ml_data,
        args.dft_functionals, args.ml_calcs,
        args.ref_functional, Path(args.output_parity)
    )

    Path(args.output_bar).parent.mkdir(parents=True, exist_ok=True)
    make_bar(
        ref_data, dft_data, ml_data,
        args.dft_functionals, args.ml_calcs,
        args.ref_functional, Path(args.output_bar)
    )

    if comparison:
        Path(args.csv_out).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.csv_out).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "method", "surface", "molecule",
                "E_ref_eV", "E_pred_eV", "diff_eV",
                "ref_functional", "reference"
            ])
            writer.writeheader()
            writer.writerows(comparison)
        print("CSV saved: {}".format(args.csv_out))

    # Summary table
    hdr_r2 = R2_SYM
    print()
    print("=" * 65)
    print("{:<22} {:>3} {:>8} {:>8} {:>8} {:>7}".format(
          "Method", "N", "MAE", "RMSE", "Bias", hdr_r2))
    print("-" * 65)
    for method in args.ml_calcs + args.dft_functionals:
        src     = (ml_data.get(method, {}) if method in args.ml_calcs
                   else dft_data.get(method, {}))
        matched = sorted(set(ref_data) & set(src))
        lbl     = METHOD_STYLES.get(method, {}).get("label", method)
        if not matched:
            print("  {:<20} {:>3}".format(lbl, "---"))
            continue
        mae, rmse, bias, r2 = stats(ref_data, src, matched)
        print("  {:<20} {:>3} {:>8.3f} {:>8.3f} {:>+8.3f} {:>7.3f}".format(
              lbl, len(matched), mae, rmse, bias, r2))
    print("=" * 65)


if __name__ == "__main__":
    main()
