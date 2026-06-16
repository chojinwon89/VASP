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
    parser.add_argument("--tasks-csv", default="workflow/tasks.csv")
    args = parser.parse_args()

    repo = Path.cwd()
    tasks_csv = repo / args.tasks_csv
    task = load_task(tasks_csv, args.task_id)

    surface = task["surface"]
    adsorbate = task["adsorbate"]
    seed = int(task["seed"])
    calculator = task["calculator"]

    run_name = f"{surface}_{adsorbate}_seed{seed}_{calculator}"
    run_dir = repo / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    status_file = run_dir / "status.json"

    status = {
        "task_id": args.task_id,
        "surface": surface,
        "adsorbate": adsorbate,
        "seed": seed,
        "calculator": calculator,
        "run_dir": str(run_dir),
        "started_at": datetime.now().isoformat(),
        "state": "running",
    }

    status_file.write_text(json.dumps(status, indent=2))

    env = os.environ.copy()
    env["GOAD_SURFACE"] = surface
    env["GOAD_ADSORBATE"] = adsorbate
    env["GOAD_SEED"] = str(seed)
    env["GOAD_CALC"] = calculator
    env["GOAD_RUN_DIR"] = str(run_dir)

    log_file = run_dir / "run.log"

    try:
        with log_file.open("w") as log:
            log.write(f"Starting task {args.task_id}\n")
            log.write(json.dumps(task, indent=2) + "\n\n")
            log.flush()

            # Optional: only run prep once globally, not per task.
            # If prep_inputs.py is lightweight, this is okay.
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

        status["state"] = "finished"
        status["finished_at"] = datetime.now().isoformat()

    except subprocess.CalledProcessError as e:
        status["state"] = "failed"
        status["returncode"] = e.returncode
        status["finished_at"] = datetime.now().isoformat()

    status_file.write_text(json.dumps(status, indent=2))

    if status["state"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()