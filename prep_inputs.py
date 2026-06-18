"""
Generate input CIFs for the GOAD workflow:

Slabs (4x4x4, 15 Å vacuum) for all FCC metals and facets:
  Cu, Pt, Pd, Ni, Ag, Au  ×  (111), (110), (100)
  Plus Cu001 alias (same as Cu100).

Gas-phase molecules in a 20×20×20 Å box built from SMILES via RDKit:
  isopropanol, CO2, ethanol, ethene, ethane,
  propane, propene, propanol, glycerol
"""
import os
from ase.build import fcc111, fcc100, fcc110
from ase.io import write
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

os.makedirs("inputs", exist_ok=True)

# ---------------------------------------------------------------------------
# Slabs
# ---------------------------------------------------------------------------
# (metal, facet_func, extra_kwargs, output_name)
SLAB_SPECS = [
    # Cu
    ("Cu", fcc111, {"orthogonal": True}, "Cu111"),
    ("Cu", fcc110, {},                  "Cu110"),
    ("Cu", fcc100, {},                  "Cu100"),
    ("Cu", fcc100, {},                  "Cu001"),   # alias
    # Pt
    ("Pt", fcc111, {"orthogonal": True}, "Pt111"),
    ("Pt", fcc110, {},                  "Pt110"),
    ("Pt", fcc100, {},                  "Pt100"),
    # Pd
    ("Pd", fcc111, {"orthogonal": True}, "Pd111"),
    ("Pd", fcc110, {},                  "Pd110"),
    ("Pd", fcc100, {},                  "Pd100"),
    # Ni
    ("Ni", fcc111, {"orthogonal": True}, "Ni111"),
    ("Ni", fcc110, {},                  "Ni110"),
    ("Ni", fcc100, {},                  "Ni100"),
    # Ag
    ("Ag", fcc111, {"orthogonal": True}, "Ag111"),
    ("Ag", fcc110, {},                  "Ag110"),
    ("Ag", fcc100, {},                  "Ag100"),
    # Au
    ("Au", fcc111, {"orthogonal": True}, "Au111"),
    ("Au", fcc110, {},                  "Au110"),
    ("Au", fcc100, {},                  "Au100"),
]

for symbol, builder, extra, name in SLAB_SPECS:
    slab = builder(symbol, size=(4, 4, 4), vacuum=15.0, **extra)
    write(f"inputs/{name}.cif", slab)
    print(f"{name}: {len(slab)} atoms, cell = {slab.get_cell().lengths()}")

# ---------------------------------------------------------------------------
# Gas-phase molecules
# ---------------------------------------------------------------------------
MOLECULES = {
    "isopropanol": "CC(C)O",
    "CO2":         "O=C=O",
    "ethanol":     "CCO",
    "ethene":      "C=C",
    "ethane":      "CC",
    "propane":     "CCC",
    "propene":     "CC=C",
    "propanol":    "CCCO",
    "glycerol":    "OCC(O)CO",
}

for mol_name, smiles in MOLECULES.items():
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(m)

    conf = m.GetConformer()
    symbols = [a.GetSymbol() for a in m.GetAtoms()]
    positions = [
        (conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z)
        for i in range(m.GetNumAtoms())
    ]

    mol = Atoms(symbols=symbols, positions=positions,
                cell=[20.0, 20.0, 20.0], pbc=True)
    mol.center()
    write(f"inputs/{mol_name}.cif", mol)
    print(f"{mol_name}: {len(mol)} atoms, cell = {mol.get_cell().lengths()}")