"""Magnetic-surface awareness for GOAD v1.0.

Why this module exists
----------------------
Adsorbates were being left *detached* (failed binding) far more often on the
reactive, magnetic transition metals Cr, Mn, Fe, Co and Ni than on the noble
metals. The root cause is that the GOAD relaxation pipeline was completely
spin-blind: the surface + adsorbate systems carried no magnetic information,
and the universal MLIPs used for the search (MatterSim, SevenNet, MACE-MP)
neither read nor produce magnetic moments, so no magnetic ground state — and
therefore no magnetically-driven chemisorption — could ever form.

This module makes GOAD *magnetism-aware*:

1. Detect magnetic-metal surfaces (``is_magnetic_surface``).
2. Seed physically-motivated initial magnetic moments on the atoms
   (``seed_initial_magmoms``) with antiferromagnetic ordering for Cr/Mn.
3. Capture the magnetic moments back out of a relaxed structure
   (``capture_magmoms``) — works with the spin-aware calculator (CHGNet).
4. Recommend / flag the right calculator for magnetic surfaces
   (``recommend_calculator``, ``is_spin_aware``).
5. Detect a *failed* (detached) binding geometry
   (``min_adsorbate_surface_distance``, ``binding_status``) so a failed
   structure is never silently accepted as the global minimum.

IMPORTANT physics caveat
------------------------
Seeding magnetic moments does **not** change the potential-energy surface of a
spin-agnostic MLIP (MatterSim / SevenNet / MACE-MP take atomic numbers and
positions only). To actually recover binding on a magnetic surface you must
relax it with a magnetism-aware model. Among GOAD's calculators, **CHGNet** is
the one whose learned energetics encode magnetism, so magnetic surfaces are
routed to it. The seeded moments are still valuable: they travel with the
structure into CHGNet's output bookkeeping and into any downstream
spin-polarised DFT (VASP ``MAGMOM``/``ISPIN=2``).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from ase import Atoms

# Reactive/magnetic 3d transition metals that drive strong chemisorption.
MAGNETIC_ELEMENTS = {"Cr", "Mn", "Fe", "Co", "Ni"}

# Elements that order antiferromagnetically in the bulk -> seed alternating
# +/- moments instead of a ferromagnetic block.
AFM_ELEMENTS = {"Cr", "Mn"}

# Calculator types (CalculatorManager keys) whose energetics encode magnetism.
SPIN_AWARE_CALCULATORS = {"chgnet"}

# Physically-motivated *initial* magnetic-moment guesses (mu_B / atom). These
# are generous starting spins; a magnetism-aware relaxation refines them. Zero
# seeds on Fe/Co/Ni/Cr are a common reason the magnetic (binding) state never
# forms.
DEFAULT_MAGMOM: Dict[str, float] = {
    "Sc": 1.0, "Ti": 2.0, "V": 3.0, "Cr": 5.0, "Mn": 5.0,
    "Fe": 4.0, "Co": 3.0, "Ni": 2.0,
    # open-shell adsorbate atoms (helps O2, NO, NO2, HCO, ...)
    "O": 0.6, "N": 0.6,
}


def magnetic_elements_in(atoms: Atoms) -> set:
    """Return the magnetic metal elements present in ``atoms``."""
    return set(atoms.get_chemical_symbols()) & MAGNETIC_ELEMENTS


def is_magnetic_surface(atoms: Atoms) -> bool:
    """True if the structure contains any magnetic transition-metal atom."""
    return bool(magnetic_elements_in(atoms))


def initial_magmoms(atoms: Atoms,
                    afm: bool = True,
                    overrides: Optional[Dict[str, float]] = None) -> List[float]:
    """Build the per-atom initial magnetic-moment list for ``atoms``.

    ``afm`` seeds AFM_ELEMENTS with alternating +/- signs (ordered by the atom
    index within each element) so an antiferromagnetic guess can form.
    ``overrides`` lets a caller tweak per-element values, e.g. ``{"Fe": 3.0}``.
    """
    table = dict(DEFAULT_MAGMOM)
    if overrides:
        table.update({k.capitalize(): float(v) for k, v in overrides.items()})

    symbols = atoms.get_chemical_symbols()
    moments = [table.get(s, 0.0) for s in symbols]

    if afm:
        seen: Dict[str, int] = {}
        for i, s in enumerate(symbols):
            if s in AFM_ELEMENTS and moments[i]:
                k = seen.get(s, 0)
                if k % 2:
                    moments[i] = -moments[i]
                seen[s] = k + 1
    return moments


def seed_initial_magmoms(atoms: Atoms,
                         afm: bool = True,
                         overrides: Optional[Dict[str, float]] = None) -> Atoms:
    """Set physically-motivated initial magnetic moments on ``atoms`` in place.

    Returns the same ``atoms`` for chaining. Safe to call on any structure; it
    is a no-op for purely non-magnetic systems (all moments 0).
    """
    atoms.set_initial_magnetic_moments(initial_magmoms(atoms, afm=afm,
                                                        overrides=overrides))
    return atoms


def capture_magmoms(atoms: Atoms,
                    metal_only: bool = False) -> Optional[Dict[str, float]]:
    """Read magnetic moments out of a (relaxed) structure.

    Tries the *computed* moments first (available when a spin-aware calculator
    such as CHGNet produced them); falls back to the *initial* seeded moments.
    Returns ``None`` if no moment information is available at all.

    The returned dict has ``total`` (sum, mu_B), ``abs_total`` (sum of |m|) and
    ``max_abs`` (largest single-atom |m|). With ``metal_only`` the sums are
    restricted to magnetic-metal atoms (a proxy for surface magnetisation).
    """
    moments = None
    try:
        m = atoms.get_magnetic_moments()
        if m is not None and len(m):
            moments = np.asarray(m, dtype=float)
    except Exception:
        moments = None

    if moments is None:
        init = atoms.get_initial_magnetic_moments()
        if init is not None and np.any(init):
            moments = np.asarray(init, dtype=float)

    if moments is None:
        return None

    if metal_only:
        mask = np.array([s in MAGNETIC_ELEMENTS
                         for s in atoms.get_chemical_symbols()])
        sel = moments[mask] if mask.any() else moments[:0]
    else:
        sel = moments

    if not len(sel):
        return {"total": 0.0, "abs_total": 0.0, "max_abs": 0.0}
    return {
        "total": float(sel.sum()),
        "abs_total": float(np.abs(sel).sum()),
        "max_abs": float(np.abs(sel).max()),
    }


def is_spin_aware(calculator_type: Optional[str]) -> bool:
    """True if the given CalculatorManager type encodes magnetism."""
    if not calculator_type:
        return False
    t = calculator_type.lower().strip().replace("+", "_").replace("-", "_")
    return t in SPIN_AWARE_CALCULATORS


def recommend_calculator(atoms: Atoms,
                         requested: Optional[str],
                         spin_aware_default: str = "chgnet"
                         ) -> Tuple[Optional[str], Optional[str]]:
    """Recommend a calculator for ``atoms`` given the ``requested`` one.

    Returns ``(recommended_type, note)``. When the surface is magnetic and the
    requested calculator is spin-blind, this recommends the spin-aware default
    (CHGNet) and explains why. Otherwise it returns ``(requested, None)``.
    """
    if not is_magnetic_surface(atoms):
        return requested, None
    if is_spin_aware(requested):
        return requested, None

    els = ", ".join(sorted(magnetic_elements_in(atoms)))
    note = (
        f"Magnetic surface detected ({els}). The requested calculator "
        f"'{requested}' is spin-agnostic and cannot form the magnetic ground "
        f"state, which frequently leaves adsorbates unbound (detached). "
        f"Recommend the spin-aware calculator '{spin_aware_default}' "
        f"(or an adsorption/OC20-fine-tuned checkpoint) for this surface."
    )
    return spin_aware_default, note


def min_adsorbate_surface_distance(system: Atoms, n_surface_atoms: int) -> Optional[float]:
    """Nearest surface-atom -> adsorbate-atom distance (minimum-image).

    ``system`` is assumed to be ``surface + adsorbate`` with the first
    ``n_surface_atoms`` entries being the surface (as GOAD builds it). Returns
    ``None`` if either fragment is empty.
    """
    n_total = len(system)
    if n_surface_atoms <= 0 or n_surface_atoms >= n_total:
        return None
    surf_idx = list(range(n_surface_atoms))
    best = None
    for a in range(n_surface_atoms, n_total):
        d = float(system.get_distances(a, surf_idx, mic=True).min())
        if best is None or d < best:
            best = d
    return best


def binding_status(system: Atoms,
                   n_surface_atoms: int,
                   detach_cutoff: float = 3.0) -> Dict[str, object]:
    """Classify whether the adsorbate is bound or detached.

    ``detach_cutoff`` (Å) is the nearest surface--adsorbate contact beyond
    which the adsorbate is considered not bound. 3.0 Å sits comfortably above
    typical metal--O/C/N bond lengths (~1.8-2.4 Å) and below physisorption/
    vacuum separations.
    """
    d = min_adsorbate_surface_distance(system, n_surface_atoms)
    detached = (d is None) or (d > detach_cutoff)
    return {
        "min_contact": d,
        "detached": bool(detached),
        "detach_cutoff": float(detach_cutoff),
    }
