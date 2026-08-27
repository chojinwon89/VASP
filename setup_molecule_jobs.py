#!/usr/bin/env python
"""
setup_molecule_jobs.py
======================
Generate VASP input files (POSCAR, INCAR, KPOINTS, POTCAR, slm.vasp.kestrel)
for gas-phase molecule reference energy calculations.

These molecular energies (E_mol) are needed to compute the DFT adsorption energy:
    E_ads = E_total(slab+mol) - E_surf(slab) - E_mol(gas)

Molecules are placed in a 20×20×20 Å cubic box (Gamma-point only).
The INCAR uses ISMEAR=0, SIGMA=0.01, and LREAL=.FALSE. — correct for
isolated molecules in large cells.

Output layout
-------------
    vasp_mol/
        isopropanol/
            PBE/
                POSCAR  INCAR  KPOINTS  POTCAR  slm.vasp.kestrel
            PBE_D3/
                ...
            r2scan/
                ...
            beef_vdw/
                ...
        CO2/
            ...

Usage
-----
    python setup_molecule_jobs.py --functional pbe
    python setup_molecule_jobs.py --functional pbe-d3
    python setup_molecule_jobs.py --functional r2scan
    python setup_molecule_jobs.py --functional beef-vdw
    python setup_molecule_jobs.py --functional pbe --molecules isopropanol CO2 ethanol
    python setup_molecule_jobs.py --functional r2scan --out-dir /scratch/jcho5/mol_jobs
    python setup_molecule_jobs.py --functional pbe --single-point
    python setup_molecule_jobs.py --functional pbe --dry-run
    python setup_molecule_jobs.py --functional pbe --force

Prerequisites
-------------
    conda activate goad
    export VASP_PP_PATH=/home/jcho5/project/paw64/potpaw_PBE_64

For beef-vdw jobs, this script also tries to copy:
    /projects/2dmgcat/vdw_kernel.bindat
"""

import argparse
import os
import re
import shutil
from pathlib import Path

from ase.io import read
from ase import Atoms

# Default POTCAR / vdW-kernel paths, per cluster (mirror setup_vasp_jobs.py).
# Resolution order for each: --pp-path/--vdw-kernel-path flag
#   > VASP_PP_PATH / VASP_VDW_KERNEL_PATH env var
#   > per-cluster default below (keyed by --cluster)
#   > generic fallback (DEFAULT_PP_PATH / DEFAULT_VDW_KERNEL_PATH)
DEFAULT_PP_PATH = "/projects/2dmgcat/paw64/potpaw_PBE_64"
DEFAULT_VDW_KERNEL_PATH = "/projects/2dmgcat/vdw_kernel.bindat"

CLUSTER_PP_PATH = {
    "kestrel":        "/projects/2dmgcat/paw64/potpaw_PBE_64",
    "perlmutter-cpu": "/pscratch/sd/j/jcho5/paw64/potpaw_PBE_64",
}
CLUSTER_VDW_KERNEL_PATH = {
    "kestrel":        "/projects/2dmgcat/vdw_kernel.bindat",
    "perlmutter-cpu": "/pscratch/sd/j/jcho5/vdw_kernel.bindat",
}


# ---------------------------------------------------------------------------
# Supported functionals  ->  (subfolder_name, INCAR_xc_block)
# (Matches setup_vasp_jobs.py conventions)
# ---------------------------------------------------------------------------
FUNCTIONAL_CONFIGS = {
    "pbe": {
        "subfolder": "PBE",
        "xc_block": """\
! Exchange-correlation
GGA = PE
""",
    },
    "pbe-d3": {
        "subfolder": "PBE_D3",
        "xc_block": """\
! Exchange-correlation
GGA    = PE
IVDW   = 11
VDW_S6 = 1.0
VDW_SR = 1.217
""",
    },
    "r2scan": {
        "subfolder": "r2scan",
        "xc_block": """\
! Exchange-correlation
METAGGA = R2SCAN
LASPH   = .TRUE.
""",
    },
    "beef-vdw": {
        "subfolder": "beef_vdw",
        "xc_block": """\
! Exchange-correlation
GGA  = BF
LUSE_VDW  = .TRUE.
AGGAC     = 0.0000
""",
    },
}

# ---------------------------------------------------------------------------
# INCAR — molecule-specific settings
# ---------------------------------------------------------------------------
INCAR_TEMPLATE = """\
SYSTEM = {system}

! Startparameter
NWRITE = 2
ISTART = 0
ISPIN  = {ispin}

! Electronic Relaxation
ENCUT  = 450
NELM   = 150
NELMIN = 4
EDIFF  = 1E-05
EDIFFG = -5E-02

{xc_block}
! Ionic Relaxation
NSW    = {nsw}
IBRION = {ibrion}
POTIM  = 0.3

! DOS — Gaussian smearing, small sigma for molecules
ISMEAR = 0
SIGMA  = 0.01

! Algorithmic — LREAL must be .FALSE. for small cells
IALGO  = 48
LDIAG  = .TRUE.
LREAL  = .FALSE.
LWAVE  = .FALSE.
"""

# ---------------------------------------------------------------------------
# KPOINTS — Gamma point only (1×1×1) for isolated molecules
# ---------------------------------------------------------------------------
KPOINTS_TEMPLATE = """\
Gamma-point only
 0
Gamma
  1  1  1
  0  0  0
"""

# ---------------------------------------------------------------------------
# Slurm — Kestrel template
# ---------------------------------------------------------------------------
SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=17
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --account=ccpc
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.out
#SBATCH --error={job_name}.err

export OMP_NUM_THREADS=1
export OMP_STACKSIZE=3G
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
ulimit -s unlimited

module load vasp/6.3.2_openMP+tpc

srun vasp_std
"""

# ---------------------------------------------------------------------------
# Slurm — Perlmutter (NERSC, CPU nodes). Molecules are tiny, so shared QOS on a
# fraction of a node keeps the charge small.
# ---------------------------------------------------------------------------
SLURM_TEMPLATE_PERLMUTTER_CPU = """\
#!/bin/bash
#SBATCH -J {job_name}
#SBATCH -A m5281
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -n 32
#SBATCH --mem=60G
#SBATCH -t 04:00:00
#SBATCH -o {job_name}.out
#SBATCH -e {job_name}.err

module load vasp-tpc/6.4.2-cpu

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

srun --cpu-bind=cores vasp_std
"""

# cluster name -> (slurm template, filename written into each job dir)
SLURM_TEMPLATES = {
    "kestrel":        (SLURM_TEMPLATE, "slm.vasp.kestrel"),
    "perlmutter-cpu": (SLURM_TEMPLATE_PERLMUTTER_CPU, "slm.vasp.perlmutter"),
}

# ---------------------------------------------------------------------------
# Formula alias map: dft_jobs/<FORMULA>_<surface> uses formula names, but the
# molecule CIFs use common names. Maps the formula (also the vasp_mol/ dir name
# calc_binding_energy.py expects) -> registry key with the CIF geometry.
# ---------------------------------------------------------------------------
FORMULA_TO_REGISTRY = {
    "C2H4":     "ethylene",
    "C2H6":     "ethane",
    "CH3CH2OH": "ethanol",
    "CH3CHO":   "acetaldehyde",
    "CH3COOH":  "acetic_acid",
    "CH3OCH3":  "DME",
    "CH3OH":    "methanol",
    "H2CO":     "formaldehyde",
    "HCOOH":    "formic_acid",
    # direct (formula == registry key): CH3, CH4, CO, CO2, H2O, H2S, N2, NH3, NO, SO2
}

# ---------------------------------------------------------------------------
# POTCAR element map
# ---------------------------------------------------------------------------
POTCAR_MAP = {
    # Non-metals / common adsorbate elements (bare-element first, matching the
    # potpaw_PBE_64 library layout used by setup_vasp_jobs.py).
    "H":  ["H"],
    "C":  ["C"],
    "N":  ["N"],
    "O":  ["O"],
    "S":  ["S"],
    "Al": ["Al"],
    # Metals
    "Cu": ["Cu", "Cu_pv"],
    "Pt": ["Pt", "Pt_pv"],
    "Pd": ["Pd", "Pd_pv"],
    "Ni": ["Ni", "Ni_pv"],
    "Ag": ["Ag", "Ag_pv"],
    "Au": ["Au"],
    "Ir": ["Ir", "Ir_pv"],
    "Rh": ["Rh", "Rh_pv"],
    "Fe": ["Fe", "Fe_pv"],
    "Co": ["Co", "Co_pv"],
    "Zn": ["Zn"],
}

# ---------------------------------------------------------------------------
# Molecule registry — maps name -> CIF path
# ---------------------------------------------------------------------------
MOLECULE_REGISTRY = {
    # Original molecules
    "isopropanol": "inputs/isopropanol.cif",
    "CO2":         "inputs/CO2.cif",
    "ethanol":     "inputs/ethanol.cif",
    "ethene":      "inputs/ethene.cif",
    "ethane":      "inputs/ethane.cif",
    "propane":     "inputs/propane.cif",
    "propene":     "inputs/propene.cif",
    "propanol":    "inputs/propanol.cif",
    "glycerol":    "inputs/glycerol.cif",
    # Inorganics / simple gases
    "H2":    "inputs/H2.cif",
    "H2O":   "inputs/H2O.cif",
    "N2":    "inputs/N2.cif",
    "O2":    "inputs/O2.cif",
    "CO":    "inputs/CO.cif",
    "NO":    "inputs/NO.cif",
    "NO2":   "inputs/NO2.cif",
    "SO2":   "inputs/SO2.cif",
    "H2S":   "inputs/H2S.cif",
    "NH3":   "inputs/NH3.cif",
    # Radicals (open-shell → need ISPIN=2)
    "CH3":   "inputs/CH3.cif",
    # C1 references
    "CH4":         "inputs/CH4.cif",
    "methane":     "inputs/methane.cif",
    "methanol":    "inputs/methanol.cif",
    "formaldehyde": "inputs/formaldehyde.cif",
    "formate":     "inputs/formate.cif",
    # Alkanes
    "butane":     "inputs/butane.cif",
    "isobutane":  "inputs/isobutane.cif",
    "pentane":    "inputs/pentane.cif",
    "isopentane": "inputs/isopentane.cif",
    "hexane":     "inputs/hexane.cif",
    "heptane":    "inputs/heptane.cif",
    "octane":     "inputs/octane.cif",
    # Alkenes
    "ethylene":  "inputs/ethylene.cif",
    "1-butene":  "inputs/1-butene.cif",
    "2-butene":  "inputs/2-butene.cif",
    "isobutene": "inputs/isobutene.cif",
    "1-pentene": "inputs/1-pentene.cif",
    "butadiene": "inputs/butadiene.cif",
    "isoprene":  "inputs/isoprene.cif",
    # Aromatics
    "benzene":     "inputs/benzene.cif",
    "toluene":     "inputs/toluene.cif",
    "furan":       "inputs/furan.cif",
    "pyrrole":     "inputs/pyrrole.cif",
    "thiophene":   "inputs/thiophene.cif",
    "styrene":     "inputs/styrene.cif",
    "xylene":      "inputs/xylene.cif",
    "phenol":      "inputs/phenol.cif",
    "2-ethylphenol": "inputs/2-ethylphenol.cif",
    "hydroquinone":  "inputs/hydroquinone.cif",
    "aniline":     "inputs/aniline.cif",
    "naphthalene": "inputs/naphthalene.cif",
    # Guaiacols
    "guaiacol":         "inputs/guaiacol.cif",
    "4-methylguaiacol": "inputs/4-methylguaiacol.cif",
    "eugenol":          "inputs/eugenol.cif",
    "isoeugenol":       "inputs/isoeugenol.cif",
    # Syringols
    "syringol":        "inputs/syringol.cif",
    "propyl_syringol": "inputs/propyl_syringol.cif",
    "syringaldehyde":  "inputs/syringaldehyde.cif",
    # Alcohols
    "1-butanol": "inputs/1-butanol.cif",
    "2-butanol": "inputs/2-butanol.cif",
    "ethylene_glycol": "inputs/ethylene_glycol.cif",
    "pentanol":  "inputs/pentanol.cif",
    "sorbitol":  "inputs/sorbitol.cif",
    "xylitol":   "inputs/xylitol.cif",
    # Sugars (approximate cyclic forms)
    "levoglucosan":             "inputs/levoglucosan.cif",
    "alpha-D-glucopyranose":    "inputs/alpha-D-glucopyranose.cif",
    "D-fructofuranose":         "inputs/D-fructofuranose.cif",
    "D-xylopyranose":           "inputs/D-xylopyranose.cif",
    "1,6-anhydroglucofuranose": "inputs/1,6-anhydroglucofuranose.cif",
    # Aldehydes
    "acetaldehyde":    "inputs/acetaldehyde.cif",
    "furfural":        "inputs/furfural.cif",
    "5-HMF":           "inputs/5-HMF.cif",
    "glyoxal":         "inputs/glyoxal.cif",
    "propanal":        "inputs/propanal.cif",
    "butanal":         "inputs/butanal.cif",
    "valeraldehyde":   "inputs/valeraldehyde.cif",
    "hexanal":         "inputs/hexanal.cif",
    "benzaldehyde":    "inputs/benzaldehyde.cif",
    "5-methylfurfural": "inputs/5-methylfurfural.cif",
    # Ketones
    "acetone":           "inputs/acetone.cif",
    "methylethylketone": "inputs/methylethylketone.cif",
    "cyclobutanone":     "inputs/cyclobutanone.cif",
    "2-pentanone":       "inputs/2-pentanone.cif",
    "cyclopentanone":    "inputs/cyclopentanone.cif",
    "2-hexanone":        "inputs/2-hexanone.cif",
    "cyclohexanone":     "inputs/cyclohexanone.cif",
    "5-heptanone":       "inputs/5-heptanone.cif",
    "2-heptanone":       "inputs/2-heptanone.cif",
    "acetophenone":      "inputs/acetophenone.cif",
    # Carboxylic acids
    "formic_acid":    "inputs/formic_acid.cif",
    "acetic_acid":    "inputs/acetic_acid.cif",
    "propionic_acid": "inputs/propionic_acid.cif",
    "butyric_acid":   "inputs/butyric_acid.cif",
    "valeric_acid":   "inputs/valeric_acid.cif",
    "caproic_acid":   "inputs/caproic_acid.cif",
    "oxalic_acid":    "inputs/oxalic_acid.cif",
    "malonic_acid":   "inputs/malonic_acid.cif",
    "succinic_acid":  "inputs/succinic_acid.cif",
    "glutaric_acid":  "inputs/glutaric_acid.cif",
    # Hydroxy/keto acids
    "lactic_acid":             "inputs/lactic_acid.cif",
    "pyruvic_acid":            "inputs/pyruvic_acid.cif",
    "3-hydroxypropionic_acid": "inputs/3-hydroxypropionic_acid.cif",
    "itaconic_acid":           "inputs/itaconic_acid.cif",
    "glycolic_acid":           "inputs/glycolic_acid.cif",
    "malic_acid":              "inputs/malic_acid.cif",
    "tartaric_acid":           "inputs/tartaric_acid.cif",
    "levulinic_acid":          "inputs/levulinic_acid.cif",
    "citric_acid":             "inputs/citric_acid.cif",
    "gluconic_acid":           "inputs/gluconic_acid.cif",
    "muconic_acid":            "inputs/muconic_acid.cif",
    # Esters/ethers
    "DME":                "inputs/DME.cif",
    "DMSO":               "inputs/DMSO.cif",
    "3-MTHF":             "inputs/3-MTHF.cif",
    "methylmethacrylate": "inputs/methylmethacrylate.cif",
    "diethyl_ether":      "inputs/diethyl_ether.cif",
    "THF":                "inputs/THF.cif",
    "ethyl_acetate":      "inputs/ethyl_acetate.cif",
    "furfuryl_alcohol":   "inputs/furfuryl_alcohol.cif",
    "gamma_valerolactone": "inputs/gamma_valerolactone.cif",
    "dimethyl_succinate": "inputs/dimethyl_succinate.cif",
    "methyl_formate":     "inputs/methyl_formate.cif",
    "angelica_lactone":   "inputs/angelica_lactone.cif",
    "gamma_butyrolactone": "inputs/gamma_butyrolactone.cif",
    # Furan
    "2-furanone":         "inputs/2-furanone.cif",
    # Oxygenates
    "hydroxyacetaldehyde":  "inputs/hydroxyacetaldehyde.cif",
    "acetal":               "inputs/acetal.cif",
    "methylcyclopentenolone": "inputs/methylcyclopentenolone.cif",
    "vanillin":             "inputs/vanillin.cif",
}

# Open-shell species (odd electron count) — must be spin-polarized (ISPIN=2).
OPEN_SHELL = {"CH3", "NO", "O2"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_poscar(atoms, comment=""):
    """Return a VASP5 POSCAR string (Cartesian, no Selective Dynamics)."""
    symbols = atoms.get_chemical_symbols()
    seen = []
    for s in symbols:
        if s not in seen:
            seen.append(s)
    sorted_idx = []
    for el in seen:
        sorted_idx.extend(i for i, s in enumerate(symbols) if s == el)
    atoms = atoms[sorted_idx]

    positions = atoms.get_positions()
    cell = atoms.get_cell()

    lines = []
    lines.append(comment or atoms.get_chemical_formula() + " molecule")
    lines.append("   1.00000000000000")

    for vec in cell:
        lines.append(f"  {vec[0]:20.16f}  {vec[1]:20.16f}  {vec[2]:20.16f}")

    species_list = []
    counts = []
    for el in seen:
        n = sum(1 for s in atoms.get_chemical_symbols() if s == el)
        species_list.append(el)
        counts.append(str(n))
    lines.append("  " + "  ".join(species_list))
    lines.append("  " + "  ".join(counts))

    lines.append("Cartesian")
    for pos in positions:
        lines.append(f"  {pos[0]:20.16f}  {pos[1]:20.16f}  {pos[2]:20.16f}")

    return "\n".join(lines) + "\n"


def find_potcar(element, pp_root):
    candidates = POTCAR_MAP.get(element, [element])
    tried = []
    for folder in candidates:
        p = pp_root / folder / "POTCAR"
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        f"POTCAR for '{element}' not found under {pp_root}.\nTried: {tried}"
    )


def build_potcar(species, pp_root, out_path, dry_run=False):
    parts = []
    for el in species:
        try:
            parts.append(find_potcar(el, pp_root))
        except FileNotFoundError as exc:
            print(f"  WARNING: {exc}")
            return False
    if not dry_run:
        with out_path.open("w") as f:
            for p in parts:
                f.write(p.read_text())
    return True


# ---------------------------------------------------------------------------
# Per-molecule setup
# ---------------------------------------------------------------------------

def setup_mol_dir(mol_name, cif_path, out_dir, pp_root,
                  functional, single_point=False, dry_run=False, force=False,
                  cluster="kestrel", vdw_kernel_path=None):
    """Load molecule CIF and write all VASP input files into out_dir/mol_name/subfolder/."""

    slurm_template, slurm_filename = SLURM_TEMPLATES[cluster]
    ispin = 2 if mol_name in OPEN_SHELL else 1

    func_cfg = FUNCTIONAL_CONFIGS[functional]
    subfolder = func_cfg["subfolder"]
    xc_block = func_cfg["xc_block"]

    job_dir = out_dir / mol_name / subfolder
    outcar_path = job_dir / "OUTCAR"
    if outcar_path.exists() and not force:
        return {
            "molecule": mol_name,
            "status": "skipped",
            "reason": f"OUTCAR already exists in {job_dir}; skipping finished job (use --force to regenerate).",
            "warnings": [],
        }

    if not dry_run:
        job_dir.mkdir(parents=True, exist_ok=True)

    if not Path(cif_path).exists():
        return {"status": "error", "reason": f"CIF not found: {cif_path}"}

    atoms = read(cif_path)

    # Ensure 20×20×20 Å box and center
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.set_pbc(True)
    atoms.center()

    n_atoms = len(atoms)
    species = list(dict.fromkeys(atoms.get_chemical_symbols()))

    status = {
        "molecule": mol_name,
        "n_atoms":  n_atoms,
        "species":  species,
        "status":   "ok",
        "warnings": [],
    }

    nsw    = 0 if single_point else 500
    ibrion = -1 if single_point else 2

    # POSCAR
    comment = f"{mol_name} molecule | {n_atoms} atoms | 20x20x20 A box"
    poscar_text = make_poscar(atoms, comment)
    if not dry_run:
        (job_dir / "POSCAR").write_text(poscar_text)

    if ispin == 2:
        status["warnings"].append("open-shell radical — ISPIN=2")

    # INCAR
    if not dry_run:
        (job_dir / "INCAR").write_text(
            INCAR_TEMPLATE.format(system=mol_name, nsw=nsw, ibrion=ibrion,
                                  xc_block=xc_block, ispin=ispin)
        )

    # KPOINTS
    if not dry_run:
        (job_dir / "KPOINTS").write_text(KPOINTS_TEMPLATE)

    # POTCAR
    potcar_path = job_dir / "POTCAR"
    if pp_root is not None:
        ok = build_potcar(species, pp_root, potcar_path, dry_run=dry_run)
        if not ok:
            status["warnings"].append("POTCAR not written — element POTCARs not found.")
            status["status"] = "partial"
    else:
        cat_cmd = " ".join(
            f"$VASP_PP_PATH/{POTCAR_MAP.get(el, [el])[0]}/POTCAR"
            for el in species
        )
        helper = (
            f"# Build POTCAR for {mol_name}\n"
            f"# Set VASP_PP_PATH first, then run:\n"
            f"cat {cat_cmd} > POTCAR\n"
        )
        if not dry_run:
            (job_dir / "make_potcar.sh").write_text(helper)
        status["warnings"].append(
            "VASP_PP_PATH not set — written make_potcar.sh instead."
        )
        status["status"] = "partial"

    # Slurm
    slurm_path = job_dir / slurm_filename
    if not dry_run:
        slurm_path.write_text(slurm_template.format(job_name=mol_name[:40]))
        slurm_path.chmod(0o755)

    # vdw_kernel.bindat for beef-vdw
    if functional == "beef-vdw":
        kernel_path = Path(vdw_kernel_path or DEFAULT_VDW_KERNEL_PATH)
        if kernel_path.exists():
            if not dry_run:
                shutil.copy2(kernel_path, job_dir / "vdw_kernel.bindat")
            status["vdw_kernel_written"] = True
        else:
            status["warnings"].append(
                f"vdw_kernel.bindat not found at: {kernel_path}. "
                "beef-vdw jobs will be missing this file; copy it manually."
            )
            status["status"] = "partial"

    return status


# ---------------------------------------------------------------------------
# Systems-dir → molecule set
# ---------------------------------------------------------------------------

_SURFACE_SUFFIX = re.compile(
    r"_(Ag|Au|Cu|Ir|Pd|Pt|Rh|Ni|Co|Fe|Cr)(100|110|111)$"
)


def derive_molecules_from_systems(systems_dir: Path) -> list[str]:
    """Return the unique adsorbate formula names under a staged DFT jobs tree.

    Strips the trailing _<metal><facet> from each <FORMULA>_<surface> dir name.
    """
    if not systems_dir.exists():
        raise FileNotFoundError(f"--systems-dir not found: {systems_dir}")
    formulas = set()
    for d in systems_dir.iterdir():
        if not d.is_dir():
            continue
        formulas.add(_SURFACE_SUFFIX.sub("", d.name))
    return sorted(formulas)


def resolve_cif(name: str):
    """Map a target name (registry key OR dft_jobs formula) to (registry_key, cif_path).

    Returns (None, None) if the name cannot be resolved.
    """
    if name in MOLECULE_REGISTRY:
        return name, MOLECULE_REGISTRY[name]
    reg_key = FORMULA_TO_REGISTRY.get(name)
    if reg_key and reg_key in MOLECULE_REGISTRY:
        return reg_key, MOLECULE_REGISTRY[reg_key]
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Generate VASP inputs for gas-phase molecule reference energy calculations."
    )
    parser.add_argument(
        "--functional",
        required=True,
        choices=list(FUNCTIONAL_CONFIGS.keys()),
        metavar="FUNC",
        help=(
            "Exchange-correlation functional to use. "
            "Supported: pbe, pbe-d3, r2scan, beef-vdw. "
            "Creates a functional-specific subfolder under each molecule, e.g. "
            "vasp_mol/CO2/PBE/, vasp_mol/CO2/PBE_D3/, etc."
        ),
    )
    parser.add_argument(
        "--molecules", nargs="+",
        default=None,
        help=(
            "Molecule/adsorbate names to set up. Accepts registry keys "
            "(e.g. ethanol) or dft_jobs formula names (e.g. CH3CH2OH); formula "
            "names are mapped to their CIF and the output dir keeps the formula "
            "so calc_binding_energy.py finds vasp_mol/<FORMULA>/. "
            "Default: all registry molecules (or the set derived from --systems-dir)."
        )
    )
    parser.add_argument(
        "--systems-dir", default=None,
        help=(
            "Derive the exact molecule set from a staged DFT jobs tree "
            "(e.g. dft_jobs). Each <FORMULA>_<surface> dir contributes its "
            "<FORMULA>; references are written to vasp_mol/<FORMULA>/ to match "
            "calc_binding_energy.py."
        )
    )
    parser.add_argument(
        "--out-dir", default="vasp_mol",
        help="Output root directory (default: ./vasp_mol)"
    )
    parser.add_argument(
        "--cluster", choices=list(SLURM_TEMPLATES.keys()), default="kestrel",
        help=(
            "Target cluster: sets the slurm script + default POTCAR/vdW paths. "
            "kestrel (slm.vasp.kestrel, /projects/2dmgcat) or perlmutter-cpu "
            "(slm.vasp.perlmutter, /pscratch). Default: kestrel."
        )
    )
    parser.add_argument(
        "--pp-path", default=None,
        help="Path to VASP PBE PAW library (overrides --cluster default and VASP_PP_PATH)"
    )
    parser.add_argument(
        "--vdw-kernel-path", default=None,
        help="Path to vdw_kernel.bindat for beef-vdw (overrides --cluster default)"
    )
    parser.add_argument(
        "--single-point", action="store_true",
        help="Write NSW=0 INCAR (single-point only, no ionic relaxation)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing any files"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate job files even when OUTCAR already exists (default: skip finished jobs)."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help=(
            "Skip any molecule whose functional job dir is already fully set up "
            "(has BOTH INCAR and POTCAR). A dir with an INCAR but a missing "
            "POTCAR is re-generated so the POTCAR gets written. Use this for "
            "incremental waves (mirror of setup_vasp_jobs.py)."
        ),
    )
    parser.add_argument(
        "--emit-joblist", default=None, metavar="PATH",
        help=(
            "Write the runnable job dirs CREATED this run to PATH (one per "
            "line, POTCAR present), ready to submit. Combine with "
            "--skip-existing to capture only the newly-created wave."
        ),
    )
    args = parser.parse_args()

    functional = args.functional
    func_cfg = FUNCTIONAL_CONFIGS[functional]
    subfolder = func_cfg["subfolder"]

    out_dir = Path(args.out_dir)

    # Determine target molecule/adsorbate names.
    #   priority: --molecules > --systems-dir > all registry keys
    if args.molecules:
        targets = args.molecules
    elif args.systems_dir:
        targets = derive_molecules_from_systems(Path(args.systems_dir))
        print(f"Derived {len(targets)} molecule(s) from {args.systems_dir}: "
              f"{' '.join(targets)}\n")
    else:
        targets = list(MOLECULE_REGISTRY.keys())

    # Resolve POTCAR library: --pp-path > VASP_PP_PATH env > per-cluster default
    # > generic fallback (mirror setup_vasp_jobs.py).
    cluster_pp_default = CLUSTER_PP_PATH.get(args.cluster, DEFAULT_PP_PATH)
    pp_path_str = (args.pp_path or os.environ.get("VASP_PP_PATH", "")
                   or cluster_pp_default)
    pp_root = Path(pp_path_str) if pp_path_str else None

    # vdW kernel: --vdw-kernel-path > VASP_VDW_KERNEL_PATH env > per-cluster
    # default > generic fallback.
    cluster_vdw_default = CLUSTER_VDW_KERNEL_PATH.get(args.cluster,
                                                      DEFAULT_VDW_KERNEL_PATH)
    vdw_kernel_path = (args.vdw_kernel_path
                       or os.environ.get("VASP_VDW_KERNEL_PATH", "")
                       or cluster_vdw_default)

    if pp_root is not None and not pp_root.exists():
        print(f"WARNING: POTCAR library not found at: {pp_root}")
        print("         POTCAR will NOT be built automatically; "
              "make_potcar.sh written instead.")
        print("         Override with --pp-path or VASP_PP_PATH env var.")
        print()
        pp_root = None
    elif pp_root is None:
        print("WARNING: VASP_PP_PATH not set — POTCAR will not be built automatically.")
        print("         make_potcar.sh will be written in each job directory instead.")
        print()
    else:
        source = ("--pp-path" if args.pp_path
                  else "VASP_PP_PATH" if os.environ.get("VASP_PP_PATH")
                  else f"{args.cluster} default")
        print(f"Using POTCAR library ({source}): {pp_root}")
        print()

    _, slurm_filename = SLURM_TEMPLATES[args.cluster]
    mode = "single-point" if args.single_point else "full relaxation"
    print(f"Functional:        {functional} -> subfolder '{subfolder}'")
    print(f"Cluster:           {args.cluster} ({slurm_filename})")
    print(f"Molecules:         {' '.join(targets)}")
    print(f"Output directory:  {out_dir}/")
    print(f"Calculation mode:  {mode}")
    print(f"Force overwrite:   {args.force}")
    print()

    all_ok = True
    n_created = 0
    n_skipped = 0
    created_dirs = []
    for target in targets:
        reg_key, cif_path = resolve_cif(target)
        if cif_path is None:
            print(f"  ERROR: '{target}' has no registry/CIF mapping. "
                  f"Add it to MOLECULE_REGISTRY or FORMULA_TO_REGISTRY. Skipping.")
            all_ok = False
            continue

        job_dir = out_dir / target / subfolder
        # Skip if the job already finished (OUTCAR) or is already fully set up
        # (INCAR+POTCAR). OUTCAR alone protects converged jobs even when POTCAR
        # isn't present locally.
        if args.skip_existing and (
            (job_dir / "OUTCAR").exists()
            or ((job_dir / "INCAR").exists() and (job_dir / "POTCAR").exists())
        ):
            n_skipped += 1
            continue

        alias = f" (CIF: {reg_key})" if reg_key != target else ""
        action = "[DRY-RUN]" if args.dry_run else "writing"
        print(f"  {action}: {job_dir}/{alias}")
        n_created += 1

        # Output dir is named by `target` (the dft_jobs formula) so
        # calc_binding_energy.py finds vasp_mol/<FORMULA>/<FUNC>/OUTCAR.
        result = setup_mol_dir(
            target, cif_path, out_dir,
            pp_root=pp_root,
            functional=functional,
            single_point=args.single_point,
            dry_run=args.dry_run,
            force=args.force,
            cluster=args.cluster,
            vdw_kernel_path=vdw_kernel_path,
        )

        if result["status"] == "skipped":
            print(f"    SKIPPED: {result['reason']}")
            print()
            continue

        if result["status"] == "error":
            print(f"    ERROR: {result['reason']}")
            all_ok = False
            continue

        print(f"    atoms:    {result['n_atoms']}")
        print(f"    species:  {' '.join(result['species'])}")

        if not args.dry_run:
            files = ["POSCAR", "INCAR", "KPOINTS", slurm_filename]
            if result["status"] == "ok":
                files.insert(3, "POTCAR")
            if result.get("vdw_kernel_written"):
                files.append("vdw_kernel.bindat")
            print(f"    written:  {', '.join(files)}")
            # Only list fully-runnable dirs (POTCAR present) in the joblist.
            if (job_dir / "POTCAR").is_file():
                created_dirs.append(job_dir)

        for w in result.get("warnings", []):
            print(f"    WARNING:  {w}")
            all_ok = False

        print()

    # ---- Optionally emit a joblist of exactly what was created this run ------
    if args.emit_joblist and not args.dry_run:
        jl = Path(args.emit_joblist)
        jl.write_text(
            f"# {len(created_dirs)} runnable job dir(s) created by "
            f"setup_molecule_jobs.py --functional {functional} this run\n"
            + "".join(f"{d.as_posix()}\n" for d in created_dirs)
        )
        print("=" * 65)
        print(f"Wrote {len(created_dirs)} job dir(s) -> {jl}")
        print()

    # Summary
    print("=" * 65)
    if args.skip_existing:
        print(f"--skip-existing: created {n_created}, skipped "
              f"{n_skipped} already-set-up molecule(s).")
        if n_created == 0:
            print("Nothing new to set up — all molecules already have inputs.")
    print("NEXT STEPS")
    print("=" * 65)
    print()
    print("Finished job directories containing OUTCAR are skipped by default.")
    print("Use --force to regenerate those directories when needed.")
    print()
    if not pp_root:
        print("1. Build POTCARs:")
        print()
        print("     export VASP_PP_PATH=/home/jcho5/project/paw64/potpaw_PBE_64")
        print(f"     python setup_molecule_jobs.py --functional {functional}    # re-run to build POTCARs")
        print()
        step = 2
    else:
        step = 1

    print(f"{step}. Submit molecule jobs:")
    print()
    for mol in targets:
        print(f"     cd {out_dir}/{mol}/{subfolder} && sbatch {slurm_filename} && cd -")
    print()
    step += 1
    print(f"{step}. After jobs finish, extract E_mol from OUTCAR:")
    print()
    for mol in targets:
        print(f"     grep 'free  energy' {out_dir}/{mol}/{subfolder}/OUTCAR | tail -1")
    print()
    step += 1
    print(f"{step}. Compute DFT adsorption energy:")
    print()
    print("     E_ads = E_total(slab+mol) - E_surf(slab) - E_mol(gas)")
    print()
    if not all_ok:
        print("Some warnings were raised — see above.")


if __name__ == "__main__":
    main()
