"""
generate_molecule_cifs.py
=========================
Generate CIF files for all gas-phase molecules needed for the CCB project
adsorption workflow.

Uses ASE `ase.build.molecule()` for named molecules where available, and
RDKit SMILES + MMFF force-field optimisation for the rest.

Output: one .cif file per molecule in inputs/

Usage
-----
    python generate_molecule_cifs.py
"""

import os
from ase.build import molecule as ase_molecule
from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit.Chem import AllChem

os.makedirs("inputs", exist_ok=True)

# ---------------------------------------------------------------------------
# Molecules available via ase.build.molecule()
# ---------------------------------------------------------------------------
ASE_NAMED = {
    "CO":       "CO",
    "CO2":      "CO2",
    "H2":       "H2",
    "H2O":      "H2O",
    "methanol": "CH3OH",
    "ethanol":  "CH3CH2OH",
    "benzene":  "C6H6",
}

# ---------------------------------------------------------------------------
# Molecules built from SMILES via RDKit
# ---------------------------------------------------------------------------
SMILES_MOLECULES = {
    # Oxygenates
    "acetaldehyde":            "CC=O",
    "acetic_acid":             "CC(=O)O",
    "propionic_acid":          "CCC(=O)O",
    "butyric_acid":            "CCCC(=O)O",
    "valeric_acid":            "CCCCC(=O)O",
    "caproic_acid":            "CCCCCC(=O)O",
    "lactic_acid":             "CC(O)C(=O)O",
    "pyruvic_acid":            "CC(=O)C(=O)O",
    "3-hydroxypropionic_acid": "OCCC(=O)O",
    "furfural":                "O=Cc1ccco1",
    "5-HMF":                   "OCc1ccc(C=O)o1",
    "cyclopentanone":          "O=C1CCCC1",
    "2-pentanone":             "CCCC(=O)C",
    "5-heptanone":             "CCCCC(=O)CC",
    "itaconic_acid":           "OC(=O)CC(=C)C(=O)O",
    "3-MTHF":                  "CC1CCCO1",
    "DME":                     "COC",
    "methylmethacrylate":      "COC(=O)C(=C)C",
    "formic_acid":             "OC=O",
    # Olefins / hydrocarbons
    "ethylene":    "C=C",
    "propene":     "CC=C",
    "1-butene":    "CCC=C",
    "isobutene":   "CC(=C)C",
    "1-pentene":   "CCCC=C",
    "butadiene":   "C=CC=C",
    "isoprene":    "CC(=C)C=C",
    "toluene":     "Cc1ccccc1",
}


def smiles_to_atoms(smiles: str, cell_size: float = 20.0) -> Atoms:
    """Convert a SMILES string to an ASE Atoms object using RDKit MMFF."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = [
        (conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z)
        for i in range(mol.GetNumAtoms())
    ]

    atoms = Atoms(symbols=symbols, positions=positions,
                  cell=[cell_size, cell_size, cell_size], pbc=True)
    atoms.center()
    return atoms


written = []
skipped = []

# --- ASE named molecules ---
for name, ase_name in ASE_NAMED.items():
    out_path = f"inputs/{name}.cif"
    if os.path.exists(out_path):
        skipped.append(name)
        continue
    try:
        atoms = ase_molecule(ase_name)
        atoms.set_cell([20.0, 20.0, 20.0])
        atoms.set_pbc(True)
        atoms.center()
        write(out_path, atoms)
        written.append(f"{name} ({len(atoms)} atoms)")
    except Exception as exc:
        print(f"  WARNING: could not build {name} via ASE: {exc}")

# --- SMILES-based molecules ---
for name, smiles in SMILES_MOLECULES.items():
    out_path = f"inputs/{name}.cif"
    if os.path.exists(out_path):
        skipped.append(name)
        continue
    try:
        atoms = smiles_to_atoms(smiles)
        write(out_path, atoms)
        written.append(f"{name} ({len(atoms)} atoms)")
    except Exception as exc:
        print(f"  WARNING: could not build {name} from SMILES '{smiles}': {exc}")

print(f"\n{'='*60}")
print(f"Molecule CIF generation complete")
print(f"  Written : {len(written)}")
print(f"  Skipped : {len(skipped)}  (already existed)")
print(f"{'='*60}")
for entry in written:
    print(f"  + {entry}")
if skipped:
    print(f"\nSkipped (not overwritten):")
    for s in skipped:
        print(f"  ~ {s}")
