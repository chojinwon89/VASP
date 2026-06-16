"""
GOAD batch driver — isopropanol on Cu(111) and Cu(100).
Bypasses the Tkinter GUI and calls the GOAD modules directly.
"""
import os, json, datetime, logging, traceback
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms

from goad_v1.analysis.surface_analyzer import SurfaceAnalyzer
from goad_v1.analysis.molecule_analyzer import MoleculeAnalyzer
from goad_v1.calculator_manager import CalculatorManager
from goad_v1.ga.genetic_algorithm import GeneticAlgorithm

# -------------------- config --------------------
CALCULATOR_TYPE = "sevennet_omni"     # "1m" (fast smoke test) | "5m" | "5m_d3" (best for vdW)
N_FIXED_LAYERS  = 2           # bottom layers fixed
RELAX_FMAX      = 0.05        # eV/Å for BFGS
RELAX_STEPS_REF = 200
RELAX_STEPS_FINAL = 300

GA_KW = dict(
    generations     = 80,
    population_size = 40,
    mutation_rate   = 0.3,
    crossover_rate  = 0.7,
    elite_size      = 5,
    verbose         = True,
)
N_SEEDS = 3   # how many independent GA runs per system; keep best

SYSTEMS = [
    {"name": "isopropanol_on_Cu111",
     "surface_cif":  "inputs/Cu111.cif",
     "molecule_cif": "inputs/isopropanol.cif"},
    {"name": "isopropanol_on_Cu100",
     "surface_cif":  "inputs/Cu100.cif",
     "molecule_cif": "inputs/isopropanol.cif"},
]
OUTROOT = f"results/batch_{datetime.datetime.now():%Y%m%d_%H%M%S}"
DB_PATH = "adsorption_db.json"
# ------------------------------------------------

os.makedirs(OUTROOT, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(f"{OUTROOT}/run.log"), logging.StreamHandler()],
)
log = logging.getLogger("batch")


def fix_bottom_layers(atoms, surface_analyzer, n_fixed):
    """Fix the bottom n_fixed atomic layers detected by SurfaceAnalyzer."""
    layers = surface_analyzer._info["layers"]["layers_list"]   # top -> bottom
    bottom = layers[-n_fixed:]                                  # last n are the bottom
    fixed_idx = [i for L in bottom for i in L["atom_indices"]]
    atoms.set_constraint(FixAtoms(indices=fixed_idx))
    return fixed_idx


def relax(atoms, calc, fmax, steps, label, outdir):
    atoms = atoms.copy()
    # re-apply any pre-existing constraints (FixAtoms is preserved by .copy())
    atoms.calc = calc
    traj = f"{outdir}/{label}.traj"
    log.info(f"Relaxing {label} (fmax={fmax}, max steps={steps}) ...")
    opt = BFGS(atoms, trajectory=traj, logfile=f"{outdir}/{label}.bfgs.log")
    opt.run(fmax=fmax, steps=steps)
    e = atoms.get_potential_energy()
    log.info(f"  {label}: E = {e:.6f} eV")
    write(f"{outdir}/{label}.cif", atoms)
    return atoms, e


def run_one_system(system, calc):
    name = system["name"]
    sysdir = f"{OUTROOT}/{name}"
    os.makedirs(sysdir, exist_ok=True)
    log.info("=" * 70)
    log.info(f"SYSTEM: {name}")
    log.info("=" * 70)

    # 1. Load and analyze
    surface  = read(system["surface_cif"])
    molecule = read(system["molecule_cif"])

    sa = SurfaceAnalyzer(surface);   sa.analyze()
    ma = MoleculeAnalyzer(molecule); ma.analyze()
    log.info("\n" + sa.get_info_text())
    log.info("\n" + ma.get_info_text())

    # 2. Reference energies
    fix_bottom_layers(surface, sa, N_FIXED_LAYERS)
    surface_relaxed,  E_surf = relax(surface,  calc, RELAX_FMAX, RELAX_STEPS_REF,
                                     "ref_surface", sysdir)
    molecule_relaxed, E_mol  = relax(molecule, calc, RELAX_FMAX, RELAX_STEPS_REF,
                                     "ref_molecule", sysdir)

    # 3. GA — multiple seeds, keep best
    best_overall = {"E_ads": np.inf, "structure": None, "seed": None,
                    "history": None, "individual": None}

    for seed in range(N_SEEDS):
        np.random.seed(seed)
        log.info(f"\n--- GA run {seed+1}/{N_SEEDS} (seed={seed}) ---")
        ga = GeneticAlgorithm(
            surface=surface_relaxed,
            molecule=molecule_relaxed,
            calculator=calc,
            surface_energy=E_surf,
            molecule_energy=E_mol,
            n_fixed_layers=N_FIXED_LAYERS,
            **GA_KW,
        )
        res = ga.run()
        log.info(f"Seed {seed}: best E_ads (single-point) = {res['best_energy']:.4f} eV")
        if res["best_energy"] < best_overall["E_ads"]:
            best_overall.update(
                E_ads      = res["best_energy"],
                structure  = res["best_structure"],
                seed       = seed,
                history    = res["fitness_history"],
                individual = res["best_individual"],
            )

    log.info(f"\nBest GA seed: {best_overall['seed']}, "
             f"E_ads (pre-relax) = {best_overall['E_ads']:.4f} eV")

    # 4. Final relaxation of the best adsorbed configuration
    best_struct = best_overall["structure"].copy()
    # GA fixes ALL surface atoms; for the final relaxation we only fix the bottom layers
    # so that surface relaxation under the adsorbate is captured.
    fix_bottom_layers(best_struct, sa, N_FIXED_LAYERS)
    final, E_total = relax(best_struct, calc, RELAX_FMAX, RELAX_STEPS_FINAL,
                           "final_adsorbed", sysdir)
    E_ads_final = E_total - (E_surf + E_mol)
    log.info(f"FINAL E_ads = {E_ads_final:.4f} eV  "
             f"(E_total={E_total:.4f}, E_surf={E_surf:.4f}, E_mol={E_mol:.4f})")

    # 5. Save GA history
    np.savetxt(f"{sysdir}/ga_history.txt", best_overall["history"],
               header="evaluation_index  E_ads_eV (single-point)")

    # 6. DB entry
    entry = {
        "system":            name,
        "surface_cif":       system["surface_cif"],
        "molecule_cif":      system["molecule_cif"],
        "calculator":        CALCULATOR_TYPE,
        "n_fixed_layers":    N_FIXED_LAYERS,
        "ga":                {**GA_KW, "n_seeds": N_SEEDS, "best_seed": best_overall["seed"]},
        "E_surface_eV":      float(E_surf),
        "E_molecule_eV":     float(E_mol),
        "E_total_eV":        float(E_total),
        "E_ads_eV":          float(E_ads_final),
        "E_ads_pre_relax":   float(best_overall["E_ads"]),
        "best_individual":   {
            "position":    best_overall["individual"]["position"].tolist(),
            "orientation": best_overall["individual"]["orientation"].tolist(),
            "torsions":    best_overall["individual"]["torsions"].tolist(),
        },
        "outputs_dir":       sysdir,
        "timestamp":         datetime.datetime.now().isoformat(timespec="seconds"),
    }
    return entry


def main():
    log.info(f"Output dir: {OUTROOT}")
    log.info(f"Loading calculator: {CALCULATOR_TYPE}")
    calc = CalculatorManager.get_calculator(CALCULATOR_TYPE)

    db = []
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            db = json.load(f)

    for system in SYSTEMS:
        try:
            entry = run_one_system(system, calc)
            db.append(entry)
            with open(DB_PATH, "w") as f:
                json.dump(db, f, indent=2)
            log.info(f"DB updated -> {DB_PATH}")
        except Exception as e:
            log.error(f"FAILED on {system['name']}: {e}")
            log.error(traceback.format_exc())

    log.info("\n=== SUMMARY ===")
    for e in db[-len(SYSTEMS):]:
        log.info(f"{e['system']}:  E_ads = {e['E_ads_eV']:.4f} eV  "
                 f"({e['calculator']})")


if __name__ == "__main__":
    main()