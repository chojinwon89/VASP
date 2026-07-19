"""
Shared molecule metadata used across workflow scripts.
"""

MOLECULE_SMILES = {
    # C0 — no carbon
    "H2":                       "[H][H]",
    "H2O":                      "O",
    "N2":                       "[N]#[N]",
    "O2":                       "[O][O]",
    "NH3":                      "N",
    "NO":                       "[N]=O",
    "NO2":                      "O=[N+][O-]",
    "SO2":                      "O=S=O",
    "H2S":                      "S",
    # C1
    "CO":                       "[C-]#[O+]",
    "CO2":                      "O=C=O",
    "methanol":                 "CO",
    "formic_acid":              "OC=O",
    "methane":                  "C",
    "CH4":                      "C",
    "formaldehyde":             "C=O",
    "formate":                  "[O-]C=O",
    "carbonate":                "[O-]C(=O)[O-]",
    "HCO":                      "C=O",
    "CH2":                      "[CH2]",
    "CH3":                      "[CH3]",
    # C2
    "ethanol":                  "CCO",
    "ethylene":                 "C=C",
    "ethene":                   "C=C",
    "ethane":                   "CC",
    "acetaldehyde":             "CC=O",
    "acetic_acid":              "CC(=O)O",
    "DME":                      "COC",
    "oxalic_acid":              "OC(=O)C(=O)O",
    "glycolic_acid":            "OCC(=O)O",
    "DMSO":                     "CS(=O)C",
    # C2 — esters/alcohols/carbonyls/oxygenates
    "methyl_formate":           "COC=O",
    "ethylene_glycol":          "OCCO",
    "glyoxal":                  "O=CC=O",
    "hydroxyacetaldehyde":      "OCC=O",
    # C3
    "isopropanol":              "CC(C)O",
    "propanol":                 "CCCO",
    "propene":                  "CC=C",
    "propane":                  "CCC",
    "propionic_acid":           "CCC(=O)O",
    "lactic_acid":              "CC(O)C(=O)O",
    "pyruvic_acid":             "CC(=O)C(=O)O",
    "3-hydroxypropionic_acid":  "OCCC(=O)O",
    "3-MTHF":                   "CC1CCCO1",
    "propanal":                 "CCC=O",
    "acetone":                  "CC(=O)C",
    "malonic_acid":             "OC(=O)CC(=O)O",
    "THF":                      "C1CCOC1",
    # C4
    "butyric_acid":             "CCCC(=O)O",
    "1-butene":                 "CCC=C",
    "isobutene":                "CC(=C)C",
    "butadiene":                "C=CC=C",
    "methylmethacrylate":       "COC(=O)C(=C)C",
    "butane":                   "CCCC",
    "isobutane":                "CC(C)C",
    "2-butene":                 "CC=CC",
    "1-butanol":                "CCCCO",
    "2-butanol":                "CCC(O)C",
    "butanal":                  "CCCC=O",
    "methylethylketone":        "CCC(=O)C",
    "cyclobutanone":            "O=C1CCC1",
    "succinic_acid":            "OC(=O)CCC(=O)O",
    "malic_acid":               "OC(CC(=O)O)C(=O)O",
    "tartaric_acid":            "OC(C(O)C(=O)O)C(=O)O",
    "diethyl_ether":            "CCOCC",
    "ethyl_acetate":            "CC(=O)OCC",
    "furan":                    "c1ccoc1",
    "pyrrole":                  "c1cc[nH]c1",
    "thiophene":                "c1ccsc1",
    # C4 — esters/furans
    "gamma_butyrolactone":      "O=C1CCCO1",
    "2-furanone":               "O=C1C=CCO1",
    # C5
    "valeric_acid":             "CCCCC(=O)O",
    "1-pentene":                "CCCC=C",
    "2-pentanone":              "CCCC(=O)C",
    "cyclopentanone":           "O=C1CCCC1",
    "furfural":                 "O=Cc1ccco1",
    "isoprene":                 "CC(=C)C=C",
    "itaconic_acid":            "OC(=O)CC(=C)C(=O)O",
    "pentane":                  "CCCCC",
    "isopentane":               "CC(C)CC",
    "pentanol":                 "CCCCCO",
    "valeraldehyde":            "CCCCC=O",
    "glutaric_acid":            "OC(=O)CCCC(=O)O",
    "levulinic_acid":           "CC(=O)CCC(=O)O",
    "gamma_valerolactone":      "CC1CCC(=O)O1",
    "furfuryl_alcohol":         "OCc1ccco1",
    "xylitol":                  "OCC(O)C(O)C(O)CO",
    # C5 — esters/sugars
    "angelica_lactone":         "CC1=CCC(=O)O1",
    "D-xylopyranose":           "OC1COC(O)C(O)C1O",
    # C6
    "caproic_acid":             "CCCCCC(=O)O",
    "5-HMF":                    "OCc1ccc(C=O)o1",
    "benzene":                  "c1ccccc1",
    "hexane":                   "CCCCCC",
    "hexanal":                  "CCCCCC=O",
    "2-hexanone":               "CCCCC(=O)C",
    "cyclohexanone":            "O=C1CCCCC1",
    "citric_acid":              "OC(=O)CC(O)(C(=O)O)CC(=O)O",
    "gluconic_acid":            "OCC(O)C(O)C(O)C(O)C(=O)O",
    "muconic_acid":             "OC(=O)C=CC=CC(=O)O",
    "dimethyl_succinate":       "COC(=O)CCC(=O)OC",
    "sorbitol":                 "OCC(O)C(O)C(O)C(O)CO",
    "phenol":                   "Oc1ccccc1",
    "aniline":                  "Nc1ccccc1",
    "5-methylfurfural":         "Cc1ccc(C=O)o1",
    # C6 — phenols/sugars/oxygenates
    "hydroquinone":             "Oc1ccc(O)cc1",
    "levoglucosan":             "OC1C(O)C(O)C2COC1O2",
    "alpha-D-glucopyranose":    "OCC1OC(O)C(O)C(O)C1O",
    "D-fructofuranose":         "OCC1(O)OCC(O)C1O",
    "1,6-anhydroglucofuranose": "OC1C2COC1OC2O",
    "acetal":                   "CC(OCC)OCC",
    "methylcyclopentenolone":   "CC1=C(O)CCC1=O",
    "5-heptanone":              "CCCCC(=O)CC",
    "toluene":                  "Cc1ccccc1",
    "heptane":                  "CCCCCCC",
    "benzaldehyde":             "O=Cc1ccccc1",
    "2-heptanone":              "CCCCCC(=O)C",
    "guaiacol":                 "COc1ccccc1O",
    # C8
    "octane":                   "CCCCCCCC",
    "styrene":                  "C=Cc1ccccc1",
    "xylene":                   "Cc1ccc(C)cc1",
    "acetophenone":             "CC(=O)c1ccccc1",
    # C8 — phenols/guaiacols/syringols/oxygenates
    "2-ethylphenol":            "CCc1ccccc1O",
    "4-methylguaiacol":         "Cc1ccc(O)c(OC)c1",
    "syringol":                 "COc1cccc(OC)c1O",
    "vanillin":                 "COc1cc(C=O)ccc1O",
    # C9
    "syringaldehyde":           "COc1cc(C=O)cc(OC)c1O",
    # C10
    "naphthalene":              "c1ccc2ccccc2c1",
    "eugenol":                  "C=CCc1ccc(O)c(OC)c1",
    "isoeugenol":               "C/C=C/c1ccc(O)c(OC)c1",
    # C11
    "propyl_syringol":          "CCCc1cc(OC)c(O)c(OC)c1",
    # C3 multi-OH
    "glycerol":                 "OCC(O)CO",
}


def carbon_count(molecule_name: str) -> int:
    """
    Return the number of carbon atoms for a molecule by counting
    C/c characters in its SMILES string.
    Falls back to counting 'C' in the molecule name if SMILES unknown.
    """
    smiles = MOLECULE_SMILES.get(molecule_name)
    if smiles:
        return sum(1 for ch in smiles if ch in ("C", "c"))
    return molecule_name.upper().count("C")
