#!/usr/bin/env python3
"""
Energy/force parity of a (fine-tuned) MLIP vs held-out DFT frames.

Predicts on every frame of a reference extxyz and reports energy-per-atom and
force MAE/RMSE, plus an optional parity PNG. Run it twice -- once with the
zero-shot base model, once with your fine-tuned checkpoint -- to prove the
fine-tune actually improved accuracy on systems it never trained on.

Examples
--------
    # zero-shot baseline
    python validate.py --ref ml/data/valid.extxyz --model 7net-mf-ompa \
        --csv-out ml/base.csv
    # fine-tuned
    python validate.py --ref ml/data/valid.extxyz --model ml/models/ft_pbe_d3.pt \
        --csv-out ml/ft.csv --plot ml/parity_ft.png
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("validate")


def compute_errors(frames, calc):
    """Return (stats dict, arrays) comparing calc predictions to stored labels."""
    import numpy as np

    eref, epred, fref, fpred = [], [], [], []
    for at in frames:
        n = len(at)
        e_ref = at.get_potential_energy()
        f_ref = at.get_forces()
        probe = at.copy()
        probe.calc = calc
        e_p = probe.get_potential_energy()
        f_p = probe.get_forces()
        eref.append(e_ref / n)
        epred.append(e_p / n)
        fref.append(f_ref.ravel())
        fpred.append(f_p.ravel())

    eref = np.array(eref)
    epred = np.array(epred)
    fref = np.concatenate(fref)
    fpred = np.concatenate(fpred)
    de = epred - eref
    df = fpred - fref
    stats = {
        "n_frames": len(frames),
        "E_MAE_meV_atom": float(1000 * np.mean(np.abs(de))),
        "E_RMSE_meV_atom": float(1000 * np.sqrt(np.mean(de ** 2))),
        "F_MAE_eV_A": float(np.mean(np.abs(df))),
        "F_RMSE_eV_A": float(np.sqrt(np.mean(df ** 2))),
    }
    return stats, (eref, epred, fref, fpred)


def load_calc(args):
    if args.calc == "sevennet":
        from sevenn.sevennet_calculator import SevenNetCalculator
        # A bare name (e.g. 7net-mf-ompa) loads a pretrained model; a path loads
        # your fine-tuned checkpoint. modal only applies to multi-fidelity models.
        kw = {"device": args.device}
        if args.modal:
            kw["modal"] = args.modal
        return SevenNetCalculator(args.model, **kw)
    if args.calc == "emt":                 # local smoke-test backend only
        from ase.calculators.emt import EMT
        return EMT()
    raise SystemExit("unknown --calc %s" % args.calc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="held-out extxyz")
    ap.add_argument("--calc", default="sevennet", choices=["sevennet", "emt"])
    ap.add_argument("--model", default="7net-mf-ompa", help="checkpoint path or model name")
    ap.add_argument("--modal", default=None, help="multi-fidelity modal, e.g. omat24")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)

    from ase.io import read

    frames = read(args.ref, index=":")
    calc = load_calc(args)
    stats, arrays = compute_errors(frames, calc)
    for k, v in stats.items():
        log.info("%-18s: %s", k, v)

    if args.csv_out:
        import csv
        with open(args.csv_out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(stats.keys())
            w.writerow(stats.values())
        log.info("stats -> %s", args.csv_out)

    if args.plot:
        import numpy as np  # noqa: F401
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        eref, epred, fref, fpred = arrays
        fig, ax = plt.subplots(1, 2, figsize=(9, 4))
        ax[0].scatter(eref, epred, s=6)
        ax[0].set(title="energy / atom (eV)", xlabel="DFT", ylabel="MLIP")
        ax[1].scatter(fref, fpred, s=2, alpha=0.3)
        ax[1].set(title="force components (eV/A)", xlabel="DFT", ylabel="MLIP")
        for a in ax:
            lo = min(a.get_xlim()[0], a.get_ylim()[0])
            hi = max(a.get_xlim()[1], a.get_ylim()[1])
            a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        log.info("parity -> %s", args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
