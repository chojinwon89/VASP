"""Shared molecule-name canonicalisation for the DFT-vs-MLIP benchmark tools.

Single source of truth so that formula-style tokens (``C2H6``, ``H2O``, ``SO2``)
and common-name tokens (``ethane``, ``water``/``H2O``, ...) are treated as the
same species EVERYWHERE:

  * calc_binding_energy.py          - DFT binding energies (+ gas-ref lookup)
  * plot_dft_vs_mlip.py             - DFT<->MLIP energy pairing (parity plot)
  * compare_dft_mlip_structures.py  - DFT<->MLIP geometry table
  * render_dft_structures.py        - DFT final-structure PNGs
  * mlip_contact_geometry.py        - MLIP bond-distance / site
  * build_dft_pages.py              - website generator

Canonical style follows the MLIP gallery filenames and the live energy page:
organics use the common name (``ethane``, ``ethanol``, ``acetic_acid``) while
small inorganics / diatomics use the UPPER-CASE formula (``CO``, ``CO2``,
``H2O``, ``SO2``, ``NH3``).  These are exactly the tokens the gallery ``.cif`` /
``.png`` files are named with, so a canonicalised (surface, molecule) key looks
the image up directly.

Add new species in ONE place here; every tool picks it up on import.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# alias (lower-cased)  ->  canonical token
# ---------------------------------------------------------------------------
MOLECULE_CANON: dict[str, str] = {
    # --- alkanes ---
    "c2h6": "ethane", "ethane": "ethane",
    "c3h8": "propane", "propane": "propane",
    "ch4": "methane", "methane": "methane",
    # --- alkenes ---
    "c2h4": "ethene", "ethene": "ethene", "ethylene": "ethene",
    "c3h6": "propene", "propene": "propene", "propylene": "propene",
    # --- alkyne ---
    "c2h2": "acetylene", "acetylene": "acetylene",
    # --- alcohols / polyols ---
    "ch3oh": "methanol", "methanol": "methanol",
    "c2h5oh": "ethanol", "ch3ch2oh": "ethanol", "ethanol": "ethanol",
    "propanol": "propanol", "1-propanol": "propanol", "n-propanol": "propanol",
    "isopropanol": "isopropanol", "2-propanol": "isopropanol", "ipa": "isopropanol",
    "glycerol": "glycerol", "c3h8o3": "glycerol",
    # --- aldehydes ---
    "ch3cho": "acetaldehyde", "acetaldehyde": "acetaldehyde",
    "h2co": "formaldehyde", "ch2o": "formaldehyde", "formaldehyde": "formaldehyde",
    # --- acids ---
    "hcooh": "formic_acid", "formic_acid": "formic_acid",
    "ch3cooh": "acetic_acid", "acetic_acid": "acetic_acid",
    # --- ethers ---
    "ch3och3": "DME", "dme": "DME",
    # --- radicals / fragments (common names, match gallery + cluster .cif) ---
    "ch3o": "methoxy", "methoxy": "methoxy",
    "oh": "hydroxyl", "hydroxyl": "hydroxyl",
    "ch3": "CH3", "ch2": "CH2",
    "hco": "HCO", "hcn": "HCN",
    # --- small inorganics / diatomics (UPPER-CASE formula = gallery token) ---
    "co": "CO", "co2": "CO2",
    "h2": "H2", "h2o": "H2O", "water": "H2O",
    "h2s": "H2S", "so2": "SO2",
    "n2": "N2", "nh3": "NH3", "no": "NO", "no2": "NO2", "o2": "O2",
    # --- atomic adsorbates (match cluster '<surf>_atomicX' naming) ---
    "h": "atomicH", "atomich": "atomicH",
    "o": "atomicO", "atomico": "atomicO",
    "n": "atomicN", "atomicn": "atomicN",
    "c": "atomicC", "atomicc": "atomicC",
    "s": "atomicS", "atomics": "atomicS",
}

# Canonical token -> formula spellings that a gas-phase reference directory
# (vasp_mol/<name>/) might use.  Lets the binding-energy tool find the gas
# reference whether the job dir and the reference dir use formula or common
# names.  Only closed-shell species with a meaningful gas reference are listed.
FORMULA_OF: dict[str, list[str]] = {
    "ethane": ["C2H6"],
    "ethene": ["C2H4"],
    "acetylene": ["C2H2"],
    "methane": ["CH4"],
    "methanol": ["CH3OH"],
    "ethanol": ["CH3CH2OH", "C2H5OH"],
    "propane": ["C3H8"],
    "propene": ["C3H6"],
    "glycerol": ["C3H8O3"],
    "acetaldehyde": ["CH3CHO", "C2H4O"],
    "formaldehyde": ["H2CO", "CH2O"],
    "formic_acid": ["HCOOH", "CH2O2"],
    "acetic_acid": ["CH3COOH", "C2H4O2"],
    "DME": ["CH3OCH3", "C2H6O"],
}


def canon_molecule(name: str) -> str:
    """Map a molecule token (formula or common name) to its canonical form.

    Unknown tokens pass through unchanged (original case preserved) so exotic
    gallery names such as ``1-butanol`` still round-trip.
    """
    if name is None:
        return ""
    n = name.strip()
    return MOLECULE_CANON.get(n.lower(), n)


def match_keys(name: str) -> set[str]:
    """Candidate lookup keys for a molecule token: {canonical, raw-lower}.

    Use when joining data sources that may spell a molecule differently
    (formula vs common name).  The canonical map is a superset, so this pair is
    enough to bridge any two spellings of the same species.
    """
    if name is None:
        return {""}
    raw = name.strip().lower()
    return {canon_molecule(name), raw}


def vasp_mol_candidates(name: str) -> list[str]:
    """Ordered, de-duplicated gas-reference directory names to try for a token.

    E.g. ``C2H6`` -> ['C2H6', 'ethane']; ``ethanol`` -> ['ethanol',
    'CH3CH2OH', 'C2H5OH'].  The raw token is tried first so an exact
    directory match always wins.
    """
    raw = (name or "").strip()
    canon = canon_molecule(raw)
    out: list[str] = []
    for cand in [raw, canon, *FORMULA_OF.get(canon, [])]:
        if cand and cand not in out:
            out.append(cand)
    return out
