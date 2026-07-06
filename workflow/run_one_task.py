#!/usr/bin/env python

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


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
    parser.add_argument("--tasks-csv", default="workflow/tasks_custom.csv")
    args = parser.parse_args()

    repo = Path.cwd()
    tasks_csv = repo / args.tasks_csv
    task = load_task(tasks_csv, args.task_id)

    surface    = task["surface"]
    adsorbate  = task["adsorbate"]
    seed       = int(task["seed"])
    calculator = task["calculator"]

    env = os.environ.copy()
    env["GOAD_SURFACE"]   = surface
    env["GOAD_ADSORBATE"] = adsorbate
    env["GOAD_SEED"]      = str(seed)
    env["GOAD_CALC"]      = calculator
    # Do NOT set GOAD_RUN_DIR here.
    # batch_isopropanol.py builds runs/C{n}/... automatically.

    # Pass per-task GA overrides (only present in tasks_custom.csv)
    env["GOAD_POPULATION_SIZE"] = task.get("population_size", "")
    env["GOAD_GENERATIONS"]     = task.get("generations", "")

    # Temporary log before run_dir is known
    tmp_log = repo / "slurm-logs" / f"task_{args.task_id}.log"
    tmp_log.parent.mkdir(parents=True, exist_ok=True)

    state      = "finished"
    returncode = 0

    try:
        with tmp_log.open("w") as log:
            log.write("=" * 80 + "\n")
            log.write(f"Starting GOAD task {args.task_id}\n")
            log.write(json.dumps(task, indent=2) + "\n")
            log.write("=" * 80 + "\n\n")
            log.flush()

            # --- Step 1: generate input CIFs (skips existing files) ---
            # generate_surface_cifs.py and generate_molecule_cifs.py
            # replace the old prep_inputs.py and cover all metals/molecules.
            for script in ["generate_surface_cifs.py", "generate_molecule_cifs.py"]:
                subprocess.run(
                    [sys.executable, script],
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

            # --- Step 2: run GOAD ---
            subprocess.run(
                [sys.executable, "batch_isopropanol.py"],
                cwd=repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    except subprocess.CalledProcessError as e:
        state      = "failed"
        returncode = e.returncode

    # Reconstruct run_dir using same carbon-count logic as batch_isopropanol.py
    # so we can write status.json in the right place.
    sys.path.insert(0, str(repo))
    from batch_isopropanol import carbon_count
    n_carbon = carbon_count(adsorbate)
    run_name = f"{surface}_{adsorbate}_seed{seed}_{calculator}"
    run_dir  = repo / "runs" / f"C{n_carbon}" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "task_id":    args.task_id,
        "surface":    surface,
        "adsorbate":  adsorbate,
        "seed":       seed,
        "calculator": calculator,
        "n_carbon":   n_carbon,
        "run_dir":    str(run_dir),
        "started_at": datetime.now().isoformat(),
        "finished_at": datetime.now().isoformat(),
        "state":      state,
    }
    if state == "failed":
        status["returncode"] = returncode

    (run_dir / "status.json").write_text(json.dumps(status, indent=2))

    if state == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
