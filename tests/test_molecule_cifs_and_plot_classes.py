from pathlib import Path

import numpy as np
from ase.io import read

from plot_dft_vs_sevennet import CLASS_MARKERS, CLASS_ORDER, MOL_CLASS

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = REPO_ROOT / 'inputs'

EXPECTED_CLASS_ORDER = [
    'Alkane',
    'Alkene',
    'Aromatic',
    'Alcohol',
    'Aldehyde',
    'Ketone',
    'Carboxylic acid',
    'Hydroxy/keto acid',
    'Ester/ether',
    'C1 ref',
    'Inorganic',
]

REQUIRED_NEW_MOLECULES = {
    'methane': 'C1 ref',
    'butane': 'Alkane',
    'isobutane': 'Alkane',
    'pentane': 'Alkane',
    'isopentane': 'Alkane',
    'hexane': 'Alkane',
    'heptane': 'Alkane',
    'octane': 'Alkane',
    '2-butene': 'Alkene',
    'furan': 'Aromatic',
    'pyrrole': 'Aromatic',
    'thiophene': 'Aromatic',
    'phenol': 'Aromatic',
    'aniline': 'Aromatic',
    'styrene': 'Aromatic',
    'xylene': 'Aromatic',
    'naphthalene': 'Aromatic',
    '1-butanol': 'Alcohol',
    '2-butanol': 'Alcohol',
    'pentanol': 'Alcohol',
    'xylitol': 'Alcohol',
    'sorbitol': 'Alcohol',
    'formaldehyde': 'C1 ref',
    'propanal': 'Aldehyde',
    'butanal': 'Aldehyde',
    'valeraldehyde': 'Aldehyde',
    'hexanal': 'Aldehyde',
    'benzaldehyde': 'Aldehyde',
    '5-methylfurfural': 'Aldehyde',
    'acetone': 'Ketone',
    'methylethylketone': 'Ketone',
    'cyclobutanone': 'Ketone',
    '2-hexanone': 'Ketone',
    'cyclohexanone': 'Ketone',
    'acetophenone': 'Ketone',
    '2-heptanone': 'Ketone',
    'oxalic_acid': 'Carboxylic acid',
    'malonic_acid': 'Carboxylic acid',
    'succinic_acid': 'Carboxylic acid',
    'glutaric_acid': 'Carboxylic acid',
    'glycolic_acid': 'Hydroxy/keto acid',
    'malic_acid': 'Hydroxy/keto acid',
    'tartaric_acid': 'Hydroxy/keto acid',
    'levulinic_acid': 'Hydroxy/keto acid',
    'citric_acid': 'Hydroxy/keto acid',
    'gluconic_acid': 'Hydroxy/keto acid',
    'diethyl_ether': 'Ester/ether',
    'THF': 'Ester/ether',
    'ethyl_acetate': 'Ester/ether',
    'furfuryl_alcohol': 'Ester/ether',
    'gamma_valerolactone': 'Ester/ether',
    'dimethyl_succinate': 'Ester/ether',
    'DMSO': 'Ester/ether',
    'formate': 'C1 ref',
    'CH4': 'C1 ref',
    'NH3': 'C1 ref',
    'NO': 'C1 ref',
    'N2': 'Inorganic',
    'O2': 'Inorganic',
    'NO2': 'Inorganic',
    'SO2': 'Inorganic',
    'H2S': 'Inorganic',
}


def test_plot_class_mappings_cover_all_new_molecules():
    assert CLASS_ORDER == EXPECTED_CLASS_ORDER

    for molecule, expected_class in REQUIRED_NEW_MOLECULES.items():
        assert molecule in MOL_CLASS
        class_label, marker = MOL_CLASS[molecule]
        assert class_label == expected_class
        assert marker == CLASS_MARKERS[expected_class]


def test_new_cifs_exist_and_have_isolated_molecule_padding():
    for molecule in REQUIRED_NEW_MOLECULES:
        cif_path = INPUTS_DIR / f'{molecule}.cif'
        assert cif_path.exists(), f'missing {cif_path.name}'

        atoms = read(cif_path)
        assert atoms.pbc.all()

        cell_lengths = np.array(atoms.cell.lengths())
        span = np.ptp(atoms.positions, axis=0)
        assert np.all(cell_lengths - span >= 30.0 - 1e-6), molecule
