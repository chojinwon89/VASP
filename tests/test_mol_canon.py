import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mol_canon import FORMULA_OF, MOLECULE_CANON, canon_molecule, match_keys, vasp_mol_candidates


def test_formula_and_common_name_share_canonical_token():
    assert canon_molecule("C2H6") == canon_molecule("ethane") == "ethane"
    assert canon_molecule("CH3CH2OH") == canon_molecule("C2H5OH") == "ethanol"
    assert canon_molecule("water") == canon_molecule("h2o") == "H2O"


def test_dft_mlip_name_bridges():
    # Spellings that differ between DFT job dirs and MLIP gallery files.
    assert canon_molecule("C2H2") == "acetylene"
    assert canon_molecule("H") == "atomicH"
    assert canon_molecule("O") == "atomicO"
    assert canon_molecule("OH") == "hydroxyl"
    assert canon_molecule("CH3O") == "methoxy"
    assert canon_molecule("hcn") == "HCN"


def test_unknown_token_passes_through_unchanged():
    assert canon_molecule("  1-Butanol ") == "1-Butanol"
    assert canon_molecule(None) == ""


def test_match_keys_bridges_common_name_to_formula():
    keys = match_keys("methane")
    assert {"methane", "CH4", "ch4"} <= keys
    assert {"ethane", "c2h6", "C2H6"} <= match_keys("C2H6")


def test_vasp_mol_candidates_tries_raw_token_first_without_duplicates():
    assert vasp_mol_candidates("C2H6") == ["C2H6", "ethane"]
    assert vasp_mol_candidates("ethanol") == ["ethanol", "CH3CH2OH", "C2H5OH"]
    assert vasp_mol_candidates("") == []


def test_tables_are_self_consistent():
    assert all(alias == alias.lower() for alias in MOLECULE_CANON)
    canon_tokens = set(MOLECULE_CANON.values())
    assert set(FORMULA_OF) <= canon_tokens
