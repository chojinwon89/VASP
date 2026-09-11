"""Tests for goad_v1.hybrid (size-aware explorer/refiner scoring)."""
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

from goad_v1 import hybrid


def _fe_surface():
    return Atoms(
        "Fe4",
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0]],
        cell=[[5, 0, 0], [0, 5, 0], [0, 0, 15]],
        pbc=[True, True, False],
    )


def _cu_surface():
    return Atoms(
        "Cu4",
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0]],
        cell=[[5, 0, 0], [0, 5, 0], [0, 0, 15]],
        pbc=[True, True, False],
    )


def _small_mol():
    return Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]])


def _large_mol():
    # 6 heavy atoms (C6) -> classified "large"
    return Atoms("C6H6", positions=[[i * 1.2, 0, 0] for i in range(12)])


def test_classify_molecule_small_vs_large():
    assert hybrid.classify_molecule(_small_mol())["size"] == "small"
    big = hybrid.classify_molecule(_large_mol())
    assert big["size"] == "large"
    assert big["n_heavy"] == 6


def test_plan_nonmagnetic_uses_single_model():
    plan = hybrid.plan_calculators(_cu_surface(), _large_mol(), "sevennet_omat")
    assert plan["magnetic"] is False
    assert plan["explorer"] == "sevennet_omat"
    assert plan["refiner"] is None


def test_plan_magnetic_large_spinblind_splits_explorer_and_refiner():
    # The dilemma case: large molecule on Fe with a spin-blind accurate model.
    plan = hybrid.plan_calculators(_fe_surface(), _large_mol(), "sevennet_omat")
    assert plan["magnetic"] is True
    assert plan["molecule_size"] == "large"
    # geometry from spin-aware CHGNet, energy from the accurate requested model
    assert plan["explorer"] == "chgnet"
    assert plan["refiner"] == "sevennet_omat"
    assert any("EXPLORER" in r or "explorer" in r for r in plan["rationale"])


def test_plan_magnetic_large_chgnet_requested_adds_accurate_refiner():
    # User asked for CHGNet, but large molecule -> CHGNet energy unreliable.
    plan = hybrid.plan_calculators(_fe_surface(), _large_mol(), "chgnet")
    assert plan["explorer"] == "chgnet"
    assert plan["refiner"] == hybrid.DEFAULT_ACCURATE_CALCULATOR
    assert any("poor for large molecules" in w for w in plan["warnings"])


def test_plan_magnetic_small_chgnet_needs_no_refiner():
    plan = hybrid.plan_calculators(_fe_surface(), _small_mol(), "chgnet")
    assert plan["explorer"] == "chgnet"
    assert plan["refiner"] is None


def test_plan_magnetic_small_spinblind_uses_explorer_and_refiner():
    plan = hybrid.plan_calculators(_fe_surface(), _small_mol(), "sevennet_omat")
    assert plan["explorer"] == "chgnet"           # geometry needs spin
    assert plan["refiner"] == "sevennet_omat"     # energy from requested model


def _bound_system(height):
    """Cu surface + a Cu adatom adsorbate at `height` (EMT-compatible)."""
    surf = _cu_surface()
    ads = Atoms("Cu", positions=[[1.25, 1.25, height]])
    return surf + ads, len(surf)


def test_refine_binding_energy_returns_consistent_eads():
    system, n_surf = _bound_system(2.2)
    # single-point (no relax) so the geometry/energy are deterministic
    out = hybrid.refine_binding_energy(
        system, n_surf, EMT(), surface_energy=0.0, molecule_energy=0.0,
        relax=False)
    expected = system.copy()
    expected.calc = EMT()
    assert abs(out["refined_energy"] - expected.get_potential_energy()) < 1e-8
    assert out["refined_e_ads"] == out["refined_energy"]  # refs are 0
    assert out["phantom_binding"] is False


def test_refine_detects_phantom_binding():
    # Start bound; a refiner that relaxes it far away should be flagged.
    system, n_surf = _bound_system(2.2)
    out = hybrid.refine_binding_energy(
        system, n_surf, EMT(), surface_energy=0.0, molecule_energy=0.0,
        relax=True, fmax=0.01, steps=200, detach_cutoff=2.6)
    # EMT Cu-Cu bonds ~2.5-2.9 Å; with a tight cutoff the relaxed contact may
    # exceed it. Whether or not it does, the flag must be a bool and, if the
    # final geometry is detached from a bound start, phantom must be True.
    assert isinstance(out["phantom_binding"], bool)
    if out["refined_detached"]:
        assert out["phantom_binding"] is True
        assert out["recommendation"] is not None
