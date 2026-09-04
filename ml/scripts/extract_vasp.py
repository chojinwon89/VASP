#!/usr/bin/env python3
"""
Extract VASP relaxation trajectories -> one training-ready extxyz.

Every ionic step of every ``vasprun.xml`` becomes a labelled training frame
carrying total energy, per-atom forces and stress (read via ASE). Each frame is
tagged with the originating ``system`` (its job path) and the DFT ``functional``,
so that:
  * downstream splitting can be done per-system (no frame leakage), and
  * a dataset can be kept to a SINGLE functional (mixing PBE / r2SCAN / BEEF in
    one training set teaches the model contradictory labels).

One relaxation of N ionic steps yields up to N labelled frames for free.

Examples
--------
    python extract_vasp.py --jobs dft_jobs --functional pbe_d3 \
        --out ml/data/all.extxyz

    python extract_vasp.py --jobs dft_jobs --jobs poscar/best \
        --functional pbe_d3 --stride 2 --max-frames-per-system 60 \
        --out ml/data/all.extxyz
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("extract_vasp")


def prepare_frames(frames, system, functional, stride=1, max_frames=None,
                   fmax_cap=100.0):
    """Tag, subsample and sanity-filter a list of ASE Atoms.

    Pure function (no I/O) so it is unit-testable without VASP files.
    """
    import numpy as np

    picked = list(frames)[::stride] if stride > 1 else list(frames)
    if max_frames and len(picked) > max_frames:
        # Keep endpoints + evenly spaced interior; the relaxed end is the most
        # valuable single frame, early SCF steps are largely redundant.
        idx = np.linspace(0, len(picked) - 1, max_frames).round().astype(int)
        picked = [picked[i] for i in sorted(set(int(i) for i in idx))]

    out = []
    for at in picked:
        try:
            e = at.get_potential_energy()
            f = at.get_forces()
        except Exception:
            continue
        if not np.isfinite(e):
            continue
        if np.max(np.abs(f)) > fmax_cap:      # drop pathological/unconverged frames
            continue
        at.info["system"] = system
        at.info["functional"] = functional
        at.info["config_type"] = "%s:%s" % (functional, system)
        out.append(at)
    return out


def iter_vasprun(job_dirs):
    """Yield every vasprun.xml under the given directories."""
    for root in job_dirs:
        root = Path(root)
        if not root.exists():
            log.warning("--jobs not found: %s", root)
            continue
        for xml in sorted(root.glob("**/vasprun.xml")):
            yield xml


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", action="append", required=True,
                    help="Directory searched recursively for vasprun.xml (repeatable)")
    ap.add_argument("--functional", required=True,
                    help="Label for the DFT functional, e.g. pbe_d3. Keep ONE per dataset.")
    ap.add_argument("--out", required=True, help="Output extxyz path")
    ap.add_argument("--stride", type=int, default=1,
                    help="Keep every Nth ionic step (default 1 = all)")
    ap.add_argument("--max-frames-per-system", type=int, default=None,
                    help="Cap frames per relaxation (evenly spaced, keeps endpoints)")
    ap.add_argument("--fmax-cap", type=float, default=100.0,
                    help="Drop frames whose max |force| exceeds this eV/Angstrom")
    args = ap.parse_args(argv)

    from ase.io import read, write

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_frames = []
    n_sys = 0
    for xml in iter_vasprun(args.jobs):
        system = str(xml.parent)
        try:
            frames = read(str(xml), index=":")
        except Exception as e:                                   # noqa: BLE001
            log.warning("skip %s: %s", xml, e)
            continue
        kept = prepare_frames(frames, system, args.functional,
                              stride=args.stride,
                              max_frames=args.max_frames_per_system,
                              fmax_cap=args.fmax_cap)
        if kept:
            all_frames.extend(kept)
            n_sys += 1
            log.info("%s: %d frames", system, len(kept))

    if not all_frames:
        log.error("No frames extracted. Check --jobs paths / vasprun.xml presence.")
        return 1

    write(str(out_path), all_frames, format="extxyz")
    log.info("\nWrote %d frames from %d systems -> %s",
             len(all_frames), n_sys, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
