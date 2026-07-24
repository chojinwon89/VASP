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
import shutil
from pathlib import Path

from ase.io import read
from ase import Atoms

DEFAULT_VDW_KERNEL_PATH = "/projects/2dmgcat/vdw_kernel.bindat"


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
ISPIN  = 1

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
# POTCAR element map
# ---------------------------------------------------------------------------
POTCAR_MAP = {
    "Cu": ["Cu_pv", "Cu"],
    "C":  ["C"],
    "H":  ["H"],
    "O":  ["O"],
    "N":  ["N"],
    "S":  ["S"],
    "Pt": ["Pt_pv", "Pt"],
    "Pd": ["Pd_pv", "Pd"],
    "Ni": ["Ni_pv", "Ni"],
    "Ag": ["Ag_pv", "Ag"],
    "Au": ["Au_pv", "Au"],
    "Fe": ["Fe_pv", "Fe"],
    "Co": ["Co_pv", "Co"],
    "Zn": ["Zn_pv", "Zn"],
    "Al": ["Al"],
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
                  functional, single_point=False, dry_run=False, force=False):
    """Load molecule CIF and write all VASP input files into out_dir/mol_name/subfolder/."""

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

    # INCAR
    if not dry_run:
        (job_dir / "INCAR").write_text(
            INCAR_TEMPLATE.format(system=mol_name, nsw=nsw, ibrion=ibrion, xc_block=xc_block)
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
    slurm_path = job_dir / "slm.vasp.kestrel"
    if not dry_run:
        slurm_path.write_text(SLURM_TEMPLATE.format(job_name=mol_name[:40]))
        slurm_path.chmod(0o755)

    # vdw_kernel.bindat for beef-vdw
    if functional == "beef-vdw":
        kernel_path = Path(DEFAULT_VDW_KERNEL_PATH)
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
# Main
# ---------------------------------------------------------------------------

def main():
    all_mol_names = list(MOLECULE_REGISTRY.keys())

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
        default=all_mol_names,
        help=f"Molecule names to set up (default: all {len(all_mol_names)} molecules)"
    )
    parser.add_argument(
        "--out-dir", default="vasp_mol",
        help="Output root directory (default: ./vasp_mol)"
    )
    parser.add_argument(
        "--pp-path", default=None,
        help="Path to VASP PBE PAW library (overrides VASP_PP_PATH env var)"
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
    args = parser.parse_args()

    functional = args.functional
    func_cfg = FUNCTIONAL_CONFIGS[functional]
    subfolder = func_cfg["subfolder"]

    out_dir = Path(args.out_dir)

    # Resolve POTCAR library
    pp_path_str = args.pp_path or os.environ.get("VASP_PP_PATH", "")
    pp_root = Path(pp_path_str) if pp_path_str else None

    if pp_root is None:
        print("WARNING: VASP_PP_PATH not set — POTCAR will not be built automatically.")
        print("         make_potcar.sh will be written in each job directory instead.")
        print()
    else:
        print(f"Using POTCAR library: {pp_root}")
        print()

    mode = "single-point" if args.single_point else "full relaxation"
    print(f"Functional:        {functional} -> subfolder '{subfolder}'")
    print(f"Molecules:         {args.molecules}")
    print(f"Output directory:  {out_dir}/")
    print(f"Calculation mode:  {mode}")
    print(f"Force overwrite:   {args.force}")
    print()

    all_ok = True
    for mol_name in args.molecules:
        if mol_name not in MOLECULE_REGISTRY:
            print(f"  ERROR: '{mol_name}' not in MOLECULE_REGISTRY. Skipping.")
            all_ok = False
            continue

        cif_path = MOLECULE_REGISTRY[mol_name]
        action = "[DRY-RUN]" if args.dry_run else "writing"
        print(f"  {action}: {out_dir / mol_name / subfolder}/")

        result = setup_mol_dir(
            mol_name, cif_path, out_dir,
            pp_root=pp_root,
            functional=functional,
            single_point=args.single_point,
            dry_run=args.dry_run,
            force=args.force,
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
            files = ["POSCAR", "INCAR", "KPOINTS", "slm.vasp.kestrel"]
            if result["status"] == "ok":
                files.insert(3, "POTCAR")
            if result.get("vdw_kernel_written"):
                files.append("vdw_kernel.bindat")
            print(f"    written:  {', '.join(files)}")

        for w in result.get("warnings", []):
            print(f"    WARNING:  {w}")
            all_ok = False

        print()

    # Summary
    print("=" * 65)
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
    for mol in args.molecules:
        print(f"     cd {out_dir}/{mol}/{subfolder} && sbatch slm.vasp.kestrel && cd -")
    print()
    step += 1
    print(f"{step}. After jobs finish, extract E_mol from OUTCAR:")
    print()
    for mol in args.molecules:
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
