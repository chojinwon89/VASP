#!/usr/bin/env python3
"""
Split an extxyz into train/valid by SYSTEM (never by frame).

Frames from the same relaxation are highly correlated: splitting them randomly
leaks near-duplicates into the validation set and reports a falsely optimistic
error. This groups frames by their ``system`` tag (written by extract_vasp.py)
and assigns whole systems to train or valid.

Example
-------
    python split_by_system.py --in ml/data/all.extxyz --train 0.9
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("split")


def group_by_system(frames):
    groups = {}
    for at in frames:
        groups.setdefault(at.info.get("system", "unknown"), []).append(at)
    return groups


def split_systems(systems, train_frac, seed):
    """Deterministically partition system ids into (train, valid) sets."""
    rng = random.Random(seed)
    systems = sorted(systems)
    rng.shuffle(systems)
    k = max(1, round(len(systems) * train_frac))
    return set(systems[:k]), set(systems[k:])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="input extxyz")
    ap.add_argument("--train", type=float, default=0.9, help="train fraction (of systems)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-train", default=None)
    ap.add_argument("--out-valid", default=None)
    args = ap.parse_args(argv)

    from ase.io import read, write

    frames = read(args.inp, index=":")
    groups = group_by_system(frames)
    train_sys, valid_sys = split_systems(groups.keys(), args.train, args.seed)

    base = Path(args.inp)
    out_tr = Path(args.out_train or base.with_name("train.extxyz"))
    out_va = Path(args.out_valid or base.with_name("valid.extxyz"))

    tr_frames = [a for s in sorted(train_sys) for a in groups[s]]
    va_frames = [a for s in sorted(valid_sys) for a in groups[s]]

    write(str(out_tr), tr_frames, format="extxyz")
    write(str(out_va), va_frames, format="extxyz")

    log.info("systems: %d train / %d valid", len(train_sys), len(valid_sys))
    log.info("frames : %d train / %d valid", len(tr_frames), len(va_frames))
    log.info("-> %s\n-> %s", out_tr, out_va)
    return 0


if __name__ == "__main__":
    sys.exit(main())
