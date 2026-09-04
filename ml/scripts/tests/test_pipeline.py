#!/usr/bin/env python3
"""
Local smoke test of the ml/ pipeline logic — no DFT, sevenn, or GPU required.

Builds a couple of synthetic EMT "relaxation" trajectories, then exercises the
pure functions from each script: extract -> split -> validate -> ensemble ->
descriptors. Run:

    python ml/scripts/tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # ml/scripts on path

import numpy as np
from ase.build import fcc111, molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

import extract_vasp
import split_by_system
import validate
import active_learning
import descriptors


def make_traj(system, n_keep=6):
    """A tiny real EMT relaxation -> list of Atoms carrying E and forces."""
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=6.0)
    ads = molecule("CO")
    ads.translate(slab.positions[-1] + [0, 0, 2.0] - ads.positions[0])
    atoms = slab + ads
    atoms.calc = EMT()
    frames = []

    def grab(a=atoms):
        s = a.copy()
        s.calc = EMT()
        s.get_potential_energy()
        s.get_forces()
        frames.append(s)

    dyn = BFGS(atoms, logfile=None)
    dyn.attach(grab, interval=1)
    dyn.run(fmax=0.2, steps=n_keep - 1)
    grab()
    return frames


def test_prepare_frames():
    frames = make_traj("Cu111_CO_top")
    kept = extract_vasp.prepare_frames(frames, "Cu111_CO_top", "pbe_d3",
                                       stride=1, max_frames=3, fmax_cap=100.0)
    assert len(kept) <= 3
    assert all(a.info["system"] == "Cu111_CO_top" for a in kept)
    assert all(a.info["functional"] == "pbe_d3" for a in kept)
    print("prepare_frames OK: %d frames, tags set" % len(kept))
    return kept


def test_split_systems():
    systems = ["Cu111_CO", "Pt111_O", "Pd111_H", "Cu100_N2", "Pt100_CH4"]
    tr, va = split_by_system.split_systems(systems, 0.8, seed=1)
    assert tr and va
    assert not (tr & va)                      # disjoint: no system in both
    assert tr | va == set(systems)            # partition covers all
    # determinism
    tr2, _ = split_by_system.split_systems(systems, 0.8, seed=1)
    assert tr == tr2
    print("split_systems OK: %d train / %d valid, disjoint+deterministic" % (len(tr), len(va)))


def test_compute_errors(frames):
    # Same calculator (EMT) as labels => errors must be ~0.
    stats, _ = validate.compute_errors(frames, EMT())
    assert stats["E_MAE_meV_atom"] < 1e-3
    assert stats["F_MAE_eV_A"] < 1e-6
    print("compute_errors OK: self-consistency MAE ~0 (E=%.2e, F=%.2e)"
          % (stats["E_MAE_meV_atom"], stats["F_MAE_eV_A"]))


def test_ensemble(frames):
    # Two identical calcs => zero disagreement; sanity that the shape is right.
    rows = active_learning.ensemble_uncertainty(frames[:3], [EMT(), EMT()])
    assert len(rows) == 3
    assert all(abs(r["E_std_meV_atom"]) < 1e-6 for r in rows)
    print("ensemble_uncertainty OK: identical models => ~0 spread")


def test_descriptors():
    f = descriptors.system_features("Pt111_C2H5OH_top")
    assert f["metal"] == "Pt" and f["facet"] == "111"
    assert f["n_C"] == 2 and f["n_H"] == 6 and f["n_O"] == 1
    print("descriptors OK: parsed Pt/111, C2H6O counts")


if __name__ == "__main__":
    frames = test_prepare_frames()
    test_split_systems()
    test_compute_errors(frames)
    test_ensemble(frames)
    test_descriptors()
    print("\nALL TESTS PASSED")
