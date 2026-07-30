"""Size-aware hybrid (explorer/refiner) scoring for GOAD v1.0.

The magnetic-surface dilemma
----------------------------
Two *different* model weaknesses collide on the reactive magnetic metals
(Cr/Mn/Fe/Co/Ni):

* Spin-agnostic MLIPs (MatterSim / SevenNet / MACE-MP) cannot form the magnetic
  ground state, so adsorbates are frequently left **detached** — a *geometry*
  failure.
* CHGNet is magnetism-aware (it fixes the geometry failure) but is trained on
  Materials-Project bulk/formation energies and gives **poor binding-energy
  accuracy for large molecules** — an *energetics* failure.

The intersection — large oxygenates on Fe/Cr — is exactly the flagged failure
set, and neither single model handles it well.

Resolution: decouple geometry from energetics
----------------------------------------------
The two failures are physically separable:

* Detachment is about the *forces / PES shape* near the surface (does the
  adsorbate get pulled in?). On magnetic metals this needs spin -> use a
  spin-aware **explorer** (CHGNet) to drive the GA geometry so it binds.
* Binding-energy accuracy is about the *depth of the well*. For large molecules
  this needs an OC20-grade **refiner** (SevenNet-OMat / MACE-MP-D3) -> compute
  the reported binding energy with the accurate model via a short,
  surface-fixed relaxation starting from the (already bound) explorer geometry.

Honest guard: "phantom binding"
-------------------------------
If the accurate refiner has *no* bound minimum for the system (it re-detaches
the adsorbate at its own optimum), then its binding energy is physically
meaningless. ``refine_binding_energy`` detects this ("phantom binding") and
flags it so the number is not trusted — for those systems only spin-polarised
DFT or an adsorption-fine-tuned checkpoint will do.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from . import magnetism

# A molecule is "large" (CHGNet binding energy considered unreliable) when it
# has at least this many heavy (non-H) atoms. This default is a physically
# motivated guess; it can be replaced by a value fitted from measured
# (n_heavy, |eps_v|) data via ``goad_v1.calibration.fit_threshold_for_target``.
DEFAULT_HEAVY_ATOM_THRESHOLD = 6

# Accurate, dispersion-aware default refiner for large organics.
DEFAULT_ACCURATE_CALCULATOR = "sevennet_omat"
# Spin-aware default explorer for magnetic surfaces.
DEFAULT_SPIN_AWARE_CALCULATOR = "chgnet"


def classify_molecule(molecule: Atoms,
                      heavy_atom_threshold: int = DEFAULT_HEAVY_ATOM_THRESHOLD
                      ) -> Dict[str, object]:
    """Classify a molecule by size for accuracy-aware routing."""
    symbols = molecule.get_chemical_symbols()
    n_heavy = sum(1 for s in symbols if s != "H")
    size = "large" if n_heavy >= heavy_atom_threshold else "small"
    return {"n_atoms": len(symbols), "n_heavy": n_heavy, "size": size}


def plan_calculators(surface: Atoms,
                     molecule: Atoms,
                     requested: Optional[str],
                     spin_aware_default: str = DEFAULT_SPIN_AWARE_CALCULATOR,
                     accurate_default: str = DEFAULT_ACCURATE_CALCULATOR,
                     heavy_atom_threshold: int = DEFAULT_HEAVY_ATOM_THRESHOLD
                     ) -> Dict[str, object]:
    """Pick explorer + refiner calculators for a surface/molecule pair.

    Returns a dict with:
        magnetic        : bool, magnetic surface?
        molecule_size   : 'small' | 'large'
        explorer        : calculator type to DRIVE THE GA GEOMETRY
        refiner         : calculator type to SCORE THE BINDING ENERGY
                          (None -> explorer energy is trustworthy, no 2nd pass)
        rationale       : list[str] explaining the choice
        warnings        : list[str] of accuracy caveats

    Routing summary
    ---------------
    * Non-magnetic surface        -> explorer = refiner = requested (one model).
    * Magnetic + small molecule   -> spin-aware explorer; CHGNet energy is
                                      acceptable, so no separate refiner unless
                                      the user's model is itself accurate.
    * Magnetic + large molecule   -> spin-aware explorer for GEOMETRY ONLY, and
                                      an accurate refiner for the binding ENERGY
                                      (this is the dilemma-resolving path).
    """
    magnetic = magnetism.is_magnetic_surface(surface)
    info = classify_molecule(molecule, heavy_atom_threshold)
    size = info["size"]
    rationale: List[str] = []
    warnings: List[str] = []

    if not magnetic:
        rationale.append(
            "Non-magnetic surface: spin is not required; a single accurate "
            f"model ('{requested}') is used for both geometry and energy.")
        return {"magnetic": False, "molecule_size": size,
                "explorer": requested, "refiner": None,
                "rationale": rationale, "warnings": warnings,
                "molecule_info": info}

    els = ", ".join(sorted(magnetism.magnetic_elements_in(surface)))

    # --- explorer: must be spin-aware so the geometry actually binds ---
    if magnetism.is_spin_aware(requested):
        explorer = requested
        rationale.append(
            f"Magnetic surface ({els}): requested model '{requested}' is "
            f"spin-aware and drives the GA geometry.")
    else:
        explorer = spin_aware_default
        rationale.append(
            f"Magnetic surface ({els}): requested model '{requested}' is "
            f"spin-agnostic and leaves adsorbates detached. Using spin-aware "
            f"'{explorer}' as the EXPLORER to drive the GA geometry.")

    # --- refiner: accurate binding energy, size-dependent ---
    if size == "large":
        if magnetism.is_spin_aware(requested):
            # user picked CHGNet, but its energy is unreliable for big molecules
            refiner = accurate_default
            warnings.append(
                f"CHGNet binding-energy accuracy is poor for large molecules "
                f"(n_heavy={info['n_heavy']}). Refining the ENERGY with the "
                f"accurate model '{refiner}'; CHGNet is used only for geometry.")
        else:
            refiner = requested
            rationale.append(
                f"Large molecule (n_heavy={info['n_heavy']}): binding ENERGY "
                f"refined with the accurate model '{refiner}'; the spin-aware "
                f"explorer is used only to obtain a bound geometry.")
        warnings.append(
            "Report the refiner's binding energy, not the explorer's. Check "
            "the phantom_binding flag: if the accurate refiner re-detaches the "
            "adsorbate, its energy is meaningless and DFT / an adsorption-"
            "fine-tuned checkpoint is required.")
    else:
        # small molecule: CHGNet energetics are acceptable
        if magnetism.is_spin_aware(requested):
            refiner = None
            rationale.append(
                "Small molecule: CHGNet energy is acceptable; no separate "
                "refiner needed.")
        else:
            refiner = requested
            rationale.append(
                f"Small molecule: geometry from spin-aware explorer, energy "
                f"refined with the requested accurate model '{refiner}'.")

    return {"magnetic": True, "molecule_size": size,
            "explorer": explorer, "refiner": refiner,
            "rationale": rationale, "warnings": warnings,
            "molecule_info": info}


def refine_binding_energy(system: Atoms,
                          n_surface_atoms: int,
                          refiner_calc,
                          surface_energy: float,
                          molecule_energy: float,
                          detach_cutoff: float = 3.0,
                          fmax: float = 0.05,
                          steps: int = 60,
                          relax: bool = True) -> Dict[str, object]:
    """Score a bound geometry with an accurate refiner calculator.

    Starting from ``system`` (surface + adsorbate, surface first), fix the
    surface, relax the adsorbate under ``refiner_calc``, and return the accurate
    binding energy plus a phantom-binding check.

    ``surface_energy`` and ``molecule_energy`` MUST be computed with the *same*
    refiner calculator, otherwise ``refined_e_ads`` is not a consistent binding
    energy.

    Returned keys:
        refined_energy     : total energy from the refiner (eV)
        refined_e_ads      : refined_energy - (surface_energy + molecule_energy)
        refined_structure  : the refiner-relaxed Atoms
        refined_min_contact: nearest surface-adsorbate distance after refine (Å)
        refined_detached   : bool, adsorbate not bound after refine
        phantom_binding    : bool, refiner re-detached an explorer-bound geometry
        recommendation     : str | None, guidance when phantom binding occurs
    """
    s = system.copy()
    s.set_constraint(FixAtoms(indices=list(range(n_surface_atoms))))
    s.calc = refiner_calc

    before = magnetism.binding_status(s, n_surface_atoms, detach_cutoff)

    if relax:
        opt = BFGS(s, logfile=None)
        opt.run(fmax=fmax, steps=steps)

    energy = float(s.get_potential_energy())
    e_ads = energy - (surface_energy + molecule_energy)
    after = magnetism.binding_status(s, n_surface_atoms, detach_cutoff)

    # phantom binding: was bound before the refiner touched it, detached after
    phantom = (not before["detached"]) and after["detached"]
    recommendation = None
    if phantom:
        recommendation = (
            "Phantom binding: the accurate refiner has no bound minimum for "
            "this system and re-detached the adsorbate, so its binding energy "
            "is not physical. Use spin-polarised DFT (ISPIN=2 + MAGMOM) or an "
            "adsorption/OC20-fine-tuned checkpoint for this system.")

    return {
        "refined_energy": energy,
        "refined_e_ads": e_ads,
        "refined_structure": s,
        "refined_min_contact": after["min_contact"],
        "refined_detached": bool(after["detached"]),
        "phantom_binding": bool(phantom),
        "recommendation": recommendation,
    }


def error_decomposition(eads_model_at_model_min: float,
                        eads_model_at_ref_min: float,
                        eads_ref_truth: float) -> Dict[str, float]:
    r"""Split a model's binding-energy error into geometry vs value parts.

    Given a reference (e.g. DFT) minimiser ``R*_ref`` and a model ``theta`` with
    its own minimiser ``R*_theta``, the total binding-energy error decomposes
    exactly (telescoping identity) as

        E_ads[theta](R*_theta) - E*_ads(R*_ref)  =  eps_g  +  eps_v

    where

        eps_g = E_ads[theta](R*_theta) - E_ads[theta](R*_ref)   (GEOMETRY error)
        eps_v = E_ads[theta](R*_ref)   - E*_ads(R*_ref)          (VALUE error)

    * ``eps_g`` depends only on *where the model's minimiser lands* (its forces /
      PES shape). It is large exactly when the model relaxes into the wrong
      basin -- e.g. a spin-blind MLIP that lets the adsorbate detach on a
      magnetic metal. |eps_g| approaches |E*_ads| for a fully detached result.
    * ``eps_v`` depends only on the *value at the correct geometry* (well depth)
      -- e.g. CHGNet's poor binding energy for large molecules.

    The hybrid explorer/refiner design targets each term with the model that
    minimises it: a spin-aware explorer to make ``eps_g`` small (correct
    geometry), an accurate refiner to make ``eps_v`` small (correct depth).

    Parameters
    ----------
    eads_model_at_model_min:
        E_ads[theta](R*_theta) -- the model's own reported binding energy.
    eads_model_at_ref_min:
        E_ads[theta](R*_ref) -- the model's energy evaluated at the reference
        geometry (single point, no relaxation).
    eads_ref_truth:
        E*_ads(R*_ref) -- the reference (DFT) binding energy.

    Returns
    -------
    dict with ``eps_g``, ``eps_v``, ``total`` (== eps_g + eps_v, and equal to
    the raw ``eads_model_at_model_min - eads_ref_truth`` up to float error).
    """
    eps_g = eads_model_at_model_min - eads_model_at_ref_min
    eps_v = eads_model_at_ref_min - eads_ref_truth
    return {"eps_g": eps_g, "eps_v": eps_v, "total": eps_g + eps_v}
