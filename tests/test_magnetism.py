"""Tests for goad_v1.magnetism (magnetic-surface awareness)."""
import numpy as np
from ase import Atoms

from goad_v1 import magnetism


def _fe_surface():
    # 4 Fe atoms in a plane + vacuum along z
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


def test_is_magnetic_surface_detects_fe_not_cu():
    assert magnetism.is_magnetic_surface(_fe_surface()) is True
    assert magnetism.is_magnetic_surface(_cu_surface()) is False
    assert magnetism.magnetic_elements_in(_fe_surface()) == {"Fe"}


def test_seed_initial_magmoms_ferromagnetic_values():
    surf = _fe_surface()
    magnetism.seed_initial_magmoms(surf, afm=True)
    m = surf.get_initial_magnetic_moments()
    # every Fe seeded at the table value (4.0), ferromagnetic (all same sign)
    assert np.allclose(m, 4.0)


def test_seed_initial_magmoms_antiferromagnetic_cr():
    cr = Atoms(
        "Cr4",
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0]],
        cell=[[5, 0, 0], [0, 5, 0], [0, 0, 15]],
        pbc=[True, True, False],
    )
    magnetism.seed_initial_magmoms(cr, afm=True)
    m = cr.get_initial_magnetic_moments()
    # AFM: alternating +/- 5.0, net moment ~0
    assert sorted(np.abs(m)) == [5.0, 5.0, 5.0, 5.0]
    assert abs(m.sum()) < 1e-9
    assert (m > 0).sum() == 2 and (m < 0).sum() == 2


def test_seed_is_noop_for_nonmagnetic():
    cu = _cu_surface()
    magnetism.seed_initial_magmoms(cu)
    assert np.allclose(cu.get_initial_magnetic_moments(), 0.0)


def test_capture_magmoms_from_seeded_initial():
    surf = _fe_surface()
    magnetism.seed_initial_magmoms(surf)
    cap = magnetism.capture_magmoms(surf, metal_only=True)
    assert cap is not None
    assert cap["total"] == 16.0        # 4 Fe * 4.0
    assert cap["abs_total"] == 16.0
    assert cap["max_abs"] == 4.0


def test_capture_returns_none_without_moments():
    assert magnetism.capture_magmoms(_cu_surface()) is None


def test_is_spin_aware():
    assert magnetism.is_spin_aware("chgnet") is True
    assert magnetism.is_spin_aware("CHGNet") is True
    assert magnetism.is_spin_aware("sevennet_omat") is False
    assert magnetism.is_spin_aware(None) is False


def test_recommend_calculator_routes_magnetic_to_chgnet():
    rec, note = magnetism.recommend_calculator(_fe_surface(), "sevennet_omat")
    assert rec == "chgnet"
    assert note and "Magnetic surface" in note

    # already spin-aware -> unchanged, no note
    rec2, note2 = magnetism.recommend_calculator(_fe_surface(), "chgnet")
    assert rec2 == "chgnet" and note2 is None

    # non-magnetic surface -> unchanged, no note
    rec3, note3 = magnetism.recommend_calculator(_cu_surface(), "sevennet_omat")
    assert rec3 == "sevennet_omat" and note3 is None


def _system(contact_z):
    """Fe surface (4 atoms at z=0) + a single O adsorbate at height contact_z."""
    surf = _fe_surface()
    ads = Atoms("O", positions=[[1.25, 1.25, contact_z]])
    return surf + ads, len(surf)


def test_min_adsorbate_surface_distance():
    system, n_surf = _system(2.0)
    d = magnetism.min_adsorbate_surface_distance(system, n_surf)
    # nearest Fe is at (0,0,0) or (2.5,2.5,0); O at (1.25,1.25,2.0)
    expected = np.sqrt(1.25**2 + 1.25**2 + 2.0**2)
    assert abs(d - expected) < 1e-6


def test_binding_status_bound_vs_detached():
    bound, n = _system(2.0)
    st = magnetism.binding_status(bound, n, detach_cutoff=3.0)
    assert st["detached"] is False

    detached, n = _system(6.0)
    st2 = magnetism.binding_status(detached, n, detach_cutoff=3.0)
    assert st2["detached"] is True


def test_min_distance_handles_degenerate_fragments():
    surf = _fe_surface()
    assert magnetism.min_adsorbate_surface_distance(surf, len(surf)) is None
    assert magnetism.min_adsorbate_surface_distance(surf, 0) is None
