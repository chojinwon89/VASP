#!/usr/bin/env python

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# SMILES lookup for carbon counting (mirrors batch_isopropanol.py)
# ---------------------------------------------------------------------------
MOLECULE_SMILES = {
    "H2":                       "[H][H]",
    "H2O":                      "O",
    "CO":                       "[C-]#[O+]",
    "CO2":                      "O=C=O",
    "methanol":                 "CO",
    "formic_acid":              "OC=O",
    "ethanol":                  "CCO",
    "ethylene":                 "C=C",
    "ethene":                   "C=C",
    "ethane":                   "CC",
    "acetaldehyde":             "CC=O",
    "acetic_acid":              "CC(=O)O",
    "DME":                      "COC",
    "isopropanol":              "CC(C)O",
    "propanol":                 "CCCO",
    "propene":                  "CC=C",
    "propane":                  "CCC",
    "propionic_acid":           "CCC(=O)O",
    "lactic_acid":              "CC(O)C(=O)O",
    "pyruvic_acid":             "CC(=O)C(=O)O",
    "3-hydroxypropionic_acid":  "OCCC(=O)O",
    "3-MTHF":                   "CC1CCCO1",
    "butyric_acid":             "CCCC(=O)O",
    "1-butene":                 "CCC=C",
    "isobutene":                "CC(=C)C",
    "butadiene":                "C=CC=C",
    "methylmethacrylate":       "COC(=O)C(=C)C",
    "valeric_acid":             "CCCCC(=O)O",
    "1-pentene":                "CCCC=C",
    "2-pentanone":              "CCCC(=O)C",
    "cyclopentanone":           "O=C1CCCC1",
    "furfural":                 "O=Cc1ccco1",
    "isoprene":                 "CC(=C)C=C",
    "itaconic_acid":            "OC(=O)CC(=C)C(=O)O",
    "caproic_acid":             "CCCCCC(=O)O",
    "5-HMF":                    "OCc1ccc(C=O)o1",
    "benzene":                  "c1ccccc1",
    "5-heptanone":              "CCCCC(=O)CC",
    "toluene":                  "Cc1ccccc1",
    "glycerol":                 "OCC(O)CO",
}


def carbon_count(molecule_name: str) -> int:
    """Count carbon atoms from SMILES (C + c). Falls back to name-based count."""
    smiles = MOLECULE_SMILES.get(molecule_name)
    if smiles:
        return sum(1 for ch in smiles if ch in ("C", "c"))
    return molecule_name.upper().count("C")


def load_task(tasks_csv: Path, task_id: int) -> dict:
    with tasks_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["task_id"]) == task_id:
                return row
    raise ValueError(f"Task ID {task_id} not found in {tasks_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--tasks-csv", default="workflow/tasks.csv")
    args = parser.parse_args()

    repo = Path.cwd()
    tasks_csv = repo / args.tasks_csv
    task = load_task(tasks_csv, args.task_id)

    surface    = task["surface"]
    adsorbate  = task["adsorbate"]
    seed       = int(task["seed"])
    calculator = task["calculator"]

    # Organise runs by carbon count: runs/C{n}/<surface>_<adsorbate>_seed<N>_<calc>/
    n_carbon   = carbon_count(adsorbate)
    run_name   = f"{surface}_{adsorbate}_seed{seed}_{calculator}"
    run_dir    = repo / "runs" / f"C{n_carbon}" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    status_file = run_dir / "status.json"
    log_file    = run_dir / "run.log"

    status = {
        "task_id":    args.task_id,
        "surface":    surface,
        "adsorbate":  adsorbate,
        "seed":       seed,
        "calculator": calculator,
        "n_carbon":   n_carbon,
        "run_dir":    str(run_dir),
        "started_at": datetime.now().isoformat(),
        "state":      "running",
    }
    status_file.write_text(json.dumps(status, indent=2))

    env = os.environ.copy()
    env["GOAD_SURFACE"]   = surface
    env["GOAD_ADSORBATE"] = adsorbate
    env["GOAD_SEED"]      = str(seed)
    env["GOAD_CALC"]      = calculator
    env["GOAD_RUN_DIR"]   = str(run_dir)

    # Pass per-task GA overrides (only present in tasks_custom.csv)
    env["GOAD_POPULATION_SIZE"] = task.get("population_size", "")
    env["GOAD_GENERATIONS"]     = task.get("generations", "")

    try:
        with log_file.open("w") as log:
            log.write("=" * 80 + "\n")
            log.write(f"Starting GOAD task {args.task_id}  [C{n_carbon}]\n")
            log.write(json.dumps(task, indent=2) + "\n")
            log.write("=" * 80 + "\n\n")
            log.flush()

            subprocess.run(
                [sys.executable, "prep_inputs.py"],
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

            subprocess.run(
                [sys.executable, "batch_isopropanol.py"],
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

        status["state"]       = "finished"
        status["finished_at"] = datetime.now().isoformat()

    except subprocess.CalledProcessError as e:
        status["state"]       = "failed"
        status["returncode"]  = e.returncode
        status["finished_at"] = datetime.now().isoformat()

    status_file.write_text(json.dumps(status, indent=2))

    if status["state"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
