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

# Cell size for all newly generated molecules (Angstrom).
# 40 Å ensures at least 30 Å of vacuum clearance around even large molecules,
# satisfying the isolation requirement for gas-phase reference calculations.
_CELL_SIZE = 40.0

# ---------------------------------------------------------------------------
# Molecules available via ase.build.molecule()
# ---------------------------------------------------------------------------
ASE_NAMED = {
    # Already existing
    "CO":       "CO",
    "CO2":      "CO2",
    "H2":       "H2",
    "H2O":      "H2O",
    "methanol": "CH3OH",
    "ethanol":  "CH3CH2OH",
    "benzene":  "C6H6",
    # C1 references
    "CH4":          "CH4",
    "methane":      "CH4",
    "formaldehyde": "H2CO",
    # Inorganics available in ASE
    "N2":  "N2",
    "O2":  "O2",
    "NH3": "NH3",
    "NO":  "NO",
    "NO2": "NO2",
    "SO2": "SO2",
}

# ---------------------------------------------------------------------------
# Molecules built from SMILES via RDKit
# ---------------------------------------------------------------------------
SMILES_MOLECULES = {
    # Oxygenates (existing)
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
    # Olefins / hydrocarbons (existing)
    "ethylene":    "C=C",
    "propene":     "CC=C",
    "1-butene":    "CCC=C",
    "isobutene":   "CC(=C)C",
    "1-pentene":   "CCCC=C",
    "butadiene":   "C=CC=C",
    "isoprene":    "CC(=C)C=C",
    "toluene":     "Cc1ccccc1",

    # --- Alkanes ---
    "ethane":     "CC",
    "propane":    "CCC",
    "butane":     "CCCC",
    "isobutane":  "CC(C)C",
    "pentane":    "CCCCC",
    "isopentane": "CC(C)CC",
    "hexane":     "CCCCCC",
    "heptane":    "CCCCCCC",
    "octane":     "CCCCCCCC",

    # --- Alkenes ---
    "2-butene": "CC=CC",
    "hexene":   "CCCCC=C",
    "heptene":  "CCCCCC=C",
    "octene":   "CCCCCCC=C",

    # --- Aromatics ---
    "furan":       "c1ccoc1",
    "pyrrole":     "c1cc[nH]c1",
    "thiophene":   "c1ccsc1",
    "styrene":     "C=Cc1ccccc1",
    "xylene":      "Cc1ccc(C)cc1",
    "phenol":      "Oc1ccccc1",
    "aniline":     "Nc1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",

    # --- Alcohols ---
    "isopropanol": "CC(C)O",
    "propanol":    "CCCO",
    "glycerol":    "OCC(O)CO",
    "1-butanol":   "CCCCO",
    "2-butanol":   "CCC(O)C",
    "pentanol":    "CCCCCO",
    "sorbitol":    "OCC(O)C(O)C(O)C(O)CO",
    "xylitol":     "OCC(O)C(O)C(O)CO",

    # --- Aldehydes ---
    "propanal":       "CCC=O",
    "butanal":        "CCCC=O",
    "valeraldehyde":  "CCCCC=O",
    "hexanal":        "CCCCCC=O",
    "benzaldehyde":   "O=Cc1ccccc1",
    "5-methylfurfural": "Cc1ccc(C=O)o1",

    # --- Ketones ---
    "acetone":           "CC(=O)C",
    "methylethylketone": "CCC(=O)C",
    "cyclobutanone":     "O=C1CCC1",
    "2-hexanone":        "CCCCC(=O)C",
    "cyclohexanone":     "O=C1CCCCC1",
    "acetophenone":      "CC(=O)c1ccccc1",
    "2-heptanone":       "CCCCCC(=O)C",

    # --- Carboxylic acids ---
    "oxalic_acid":  "OC(=O)C(=O)O",
    "malonic_acid": "OC(=O)CC(=O)O",
    "succinic_acid": "OC(=O)CCC(=O)O",
    "glutaric_acid": "OC(=O)CCCC(=O)O",

    # --- Hydroxy/keto acids ---
    "glycolic_acid":  "OCC(=O)O",
    "malic_acid":     "OC(CC(=O)O)C(=O)O",
    "tartaric_acid":  "OC(C(O)C(=O)O)C(=O)O",
    "citric_acid":    "OC(=O)CC(O)(C(=O)O)CC(=O)O",
    "levulinic_acid": "CC(=O)CCC(=O)O",
    "gluconic_acid":  "OCC(O)C(O)C(O)C(O)C(=O)O",
    "muconic_acid":   "OC(=O)C=CC=CC(=O)O",

    # --- Esters/ethers ---
    "diethyl_ether":      "CCOCC",
    "THF":                "C1CCOC1",
    "ethyl_acetate":      "CC(=O)OCC",
    "gamma_valerolactone": "CC1CCC(=O)O1",
    "dimethyl_succinate":  "COC(=O)CCC(=O)OC",
    "furfuryl_alcohol":    "OCc1ccco1",
    "DMSO":               "CS(=O)C",

    # --- Esters ---
    "methyl_formate":      "COC=O",
    "angelica_lactone":    "CC1=CCC(=O)O1",
    "gamma_butyrolactone": "O=C1CCCO1",

    # --- Alcohols ---
    "ethylene_glycol": "OCCO",

    # --- Carbonyls ---
    "glyoxal": "O=CC=O",

    # --- Phenols ---
    "2-ethylphenol": "CCc1ccccc1O",
    "hydroquinone":  "Oc1ccc(O)cc1",

    # --- Guaiacols ---
    "guaiacol":         "COc1ccccc1O",
    "4-methylguaiacol": "Cc1ccc(O)c(OC)c1",
    "eugenol":          "C=CCc1ccc(O)c(OC)c1",
    "isoeugenol":       "C/C=C/c1ccc(O)c(OC)c1",

    # --- Syringols ---
    "syringol":         "COc1cccc(OC)c1O",
    "propyl_syringol":  "CCCc1cc(OC)c(O)c(OC)c1",
    "syringaldehyde":   "COc1cc(C=O)cc(OC)c1O",

    # --- Sugars ---
    # NOTE: these cyclic polyols/anhydrosugars are approximate structures;
    # stereochemistry may not be perfectly resolved by RDKit MMFF and should
    # be manually verified before production DFT/MLIP use.
    "levoglucosan":            "OC1C(O)C(O)C2COC1O2",
    "alpha-D-glucopyranose":   "OCC1OC(O)C(O)C(O)C1O",
    "D-fructofuranose":        "OCC1(O)OCC(O)C1O",
    "D-xylopyranose":          "OC1COC(O)C(O)C1O",
    "1,6-anhydroglucofuranose": "OC1C2COC1OC2O",

    # --- Furan ---
    "2-furanone": "O=C1C=CCO1",

    # --- Oxygenates ---
    "hydroxyacetaldehyde":  "OCC=O",
    "acetal":               "CC(OCC)OCC",
    "methylcyclopentenolone": "CC1=C(O)CCC1=O",
    "vanillin":             "COc1cc(C=O)ccc1O",

    # --- Inorganic (fallback via RDKit for those ASE doesn't have) ---
    "H2S": "S",

    # --- C1 reference fragments ---
    # NOTE: these are open-shell or charged species (formate is a carboxylate
    # anion, carbonate is a dianion, HCO is a formyl radical, CH2/CH3 are
    # radicals).  The SMILES below are approximate closed-shell surrogates
    # for geometry-only CIF generation; they should NOT be used for spin- or
    # charge-sensitive DFT/MLIP calculations without appropriate settings.
    "formate":    "[O-]C=O",   # formate anion (approximation)
    "carbonate":  "[O-]C(=O)[O-]",  # carbonate dianion (approximation)
    "HCO":        "C=O",       # formyl radical approximated as formaldehyde
    "CH2":        "[CH2]",     # singlet/triplet methylene (approximation)
    "CH3":        "[CH3]",     # methyl radical (approximation)
}


def smiles_to_atoms(smiles: str, cell_size: float = _CELL_SIZE) -> Atoms:
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
        atoms.set_cell([_CELL_SIZE, _CELL_SIZE, _CELL_SIZE])
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
