#!/usr/bin/env python
"""
run_one_task.py
===============
Execute a single task from the tasks CSV, handling all pipeline steps.

Skip-if-finished behaviour
--------------------------
Before running any computation, this script checks whether the expected
run directory already contains a ``status.json`` with ``state == "finished"``.
If so, the task is skipped and the script exits cleanly with code 0.

This makes bulk re-submission via ``submit_missing.sh`` (produced by
``find_missing_tasks.py``) safe to run repeatedly: already-finished tasks are
silently skipped, so no compute is wasted re-running work that completed in a
previous Slurm array submission.

Use ``--force`` to override this check and re-run a task even if it is already
marked as finished (e.g. after updating a calculator or changing GA parameters).
"""

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
    parser = argparse.ArgumentParser(
        description="Run one GOAD task from the tasks CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--tasks-csv", default="workflow/tasks_custom.csv")
    parser.add_argument(
        "--force", action="store_true", default=False,
        help=(
            "Re-run the task even if its run directory already contains "
            "status.json with state == 'finished'. Without this flag, "
            "already-finished tasks are skipped and exit 0."
        ),
    )
    args = parser.parse_args()

    repo = Path.cwd()
    tasks_csv = repo / args.tasks_csv
    task = load_task(tasks_csv, args.task_id)

    surface    = task["surface"]
    adsorbate  = task["adsorbate"]
    seed       = int(task["seed"])
    calculator = task["calculator"]

    # Reconstruct run_dir using same carbon-count logic as batch_isopropanol.py
    # so we can check / write status.json in the right place.
    sys.path.insert(0, str(repo))
    from batch_isopropanol import carbon_count
    n_carbon = carbon_count(adsorbate)
    run_name = f"{surface}_{adsorbate}_seed{seed}_{calculator}"
    run_dir  = repo / "runs" / f"C{n_carbon}" / run_name

    # ------------------------------------------------------------------
    # Skip-if-finished check (bypassed by --force)
    # ------------------------------------------------------------------
    if not args.force:
        status_file = run_dir / "status.json"
        if status_file.exists():
            try:
                existing = json.loads(status_file.read_text())
                if existing.get("state") == "finished":
                    print(
                        f"[run_one_task] Task {args.task_id} already finished — "
                        f"skipping.\n"
                        f"  run_dir: {run_dir}\n"
                        f"  Use --force to re-run."
                    )
                    raise SystemExit(0)
            except (json.JSONDecodeError, OSError):
                pass  # Corrupt status.json — re-run the task

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
