"""
Generate input CIFs for the GOAD test runs:
  - Cu(111) 4x4x4 slab, 15 Å vacuum
  - Cu(110) 4x4x4 slab, 15 Å vacuum
  - Cu(100) 4x4x4 slab, 15 Å vacuum
  - Cu(001) alias written from the Cu(100)/fcc100 slab
  - Isopropanol (CC(C)O) molecule in a 20 Å cubic box
"""
import os
from ase.build import fcc111, fcc100, fcc110
from ase.io import write
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

os.makedirs("inputs", exist_ok=True)

# ---- Cu slabs ----
# Lattice constant a = 3.615 Å is fine for Cu (default in ase.build).
# 4 layers, 4x4 in-plane, ~15 Å vacuum, orthogonal cell -> easy for periodic GA.
slab_111 = fcc111("Cu", size=(4, 4, 4), vacuum=15.0, orthogonal=True)
slab_110 = fcc110("Cu", size=(4, 4, 4), vacuum=15.0)
slab_100 = fcc100("Cu", size=(4, 4, 4), vacuum=15.0)

write("inputs/Cu111.cif", slab_111)
write("inputs/Cu110.cif", slab_110)
write("inputs/Cu100.cif", slab_100)
write("inputs/Cu001.cif", slab_100)
print(f"Cu(111): {len(slab_111)} atoms, cell = {slab_111.get_cell().lengths()}")
print(f"Cu(110): {len(slab_110)} atoms, cell = {slab_110.get_cell().lengths()}")
print(f"Cu(100)/Cu(001): {len(slab_100)} atoms, cell = {slab_100.get_cell().lengths()}")

# ---- Isopropanol molecule (CH3-CHOH-CH3) ----
smiles = "CC(C)O"
m = Chem.AddHs(Chem.MolFromSmiles(smiles))
AllChem.EmbedMolecule(m, randomSeed=42)
AllChem.MMFFOptimizeMolecule(m)

conf = m.GetConformer()
symbols = [a.GetSymbol() for a in m.GetAtoms()]
positions = [(conf.GetAtomPosition(i).x,
              conf.GetAtomPosition(i).y,
              conf.GetAtomPosition(i).z) for i in range(m.GetNumAtoms())]

mol = Atoms(symbols=symbols, positions=positions,
            cell=[20.0, 20.0, 20.0], pbc=True)
mol.center()
write("inputs/isopropanol.cif", mol)
print(f"Isopropanol: {len(mol)} atoms (formula expected C3H8O = 12 atoms)")