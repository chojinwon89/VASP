#!/usr/bin/env python3
"""
Active-learning query selection by ensemble disagreement.

Given several fine-tuned SevenNet checkpoints trained with different seeds and a
pool of candidate structures (extxyz), rank the pool by ensemble spread and
write the top-K to ``to_label.extxyz`` for the next DFT batch. This is the
"which structures are worth a DFT calculation?" step of the loop:

    GOAD(ensemble)  ->  rank by uncertainty  ->  DFT single-point top-K
        ->  add to training set  ->  retrain  ->  repeat

The pool is anything ASE can read; in practice you feed it the GOAD-proposed
structures (e.g. exported CIFs converted to extxyz) whose energies you do NOT
yet trust.
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("active_learning")


def ensemble_uncertainty(frames, calcs):
    """Per-structure std of energy/atom and max-force across the ensemble."""
    import numpy as np

    rows = []
    for i, at in enumerate(frames):
        energies, fmaxes = [], []
        for c in calcs:
            probe = at.copy()
            probe.calc = c
            energies.append(probe.get_potential_energy() / len(at))
            fmaxes.append(float(np.max(np.abs(probe.get_forces()))))
        rows.append({
            "idx": i,
            "system": at.info.get("system", "?"),
            "E_std_meV_atom": float(1000 * np.std(energies)),
            "Fmax_std_eV_A": float(np.std(fmaxes)),
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True, help="candidate extxyz")
    ap.add_argument("--models", nargs="+", required=True,
                    help="2+ fine-tuned checkpoint paths (different seeds)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", default="to_label.extxyz")
    ap.add_argument("--rank-by", default="E_std_meV_atom",
                    choices=["E_std_meV_atom", "Fmax_std_eV_A"])
    args = ap.parse_args(argv)

    if len(args.models) < 2:
        ap.error("need at least 2 checkpoints to measure disagreement")

    from ase.io import read, write
    from sevenn.sevennet_calculator import SevenNetCalculator

    frames = read(args.pool, index=":")
    calcs = [SevenNetCalculator(m, device=args.device) for m in args.models]
    rows = ensemble_uncertainty(frames, calcs)
    rows.sort(key=lambda r: r[args.rank_by], reverse=True)

    pick = [frames[r["idx"]] for r in rows[:args.top_k]]
    write(args.out, pick, format="extxyz")
    log.info("selected %d / %d highest-uncertainty frames -> %s",
             len(pick), len(frames), args.out)
    log.info("top uncertainty: %s", rows[0] if rows else "n/a")
    return 0


if __name__ == "__main__":
    sys.exit(main())
