#!/usr/bin/env python
"""
find_missing_tasks.py
=====================
Compare workflow/tasks_custom.csv against the runs/ directory and
classify every task into one of three buckets:

  finished — run dir exists AND status.json has state == "finished"
  failed   — run dir exists but state != "finished" (failed / crashed / still running)
  missing  — run directory does not exist at all

Automatically writes submit_missing.sh with chunked sbatch commands
(to stay under Slurm's --array argument length limit of ~4096 chars)
covering BOTH failed and missing tasks so all incomplete work is
resubmitted in one go.

Usage
-----
    # Default: checks workflow/tasks_custom.csv vs ./runs/
    python find_missing_tasks.py

    # Custom paths
    python find_missing_tasks.py \\
        --tasks    workflow/tasks_custom.csv \\
        --runs-dir /scratch/jcho5/goad-global-optimization/runs

    # Custom chunk size and throttle (default: 200 IDs per chunk, 20 concurrent)
    python find_missing_tasks.py --chunk-size 150 --throttle 30

    # Only resubmit truly missing (skip failed/crashed)
    python find_missing_tasks.py --missing-only

    # Print every incomplete task in a table
    python find_missing_tasks.py --show-all

Output
------
    Tasks CSV      : workflow/tasks_custom.csv  (1296 total)
    Runs directory : runs

      Finished              :  324
      Failed / not finished :  125   ← dir exists, state != finished
      Missing (no dir)      :  847   ← never started

    Missing by calculator:
      5m             : 648 tasks
      sevennet_omni  : 199 tasks
    ...
    Written: submit_missing.sh  (5 sbatch commands)
    Run with:  bash submit_missing.sh
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_CHUNK    = 200
DEFAULT_THROTTLE = 20


def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def get_run_state(run_dir: Path) -> str:
    """
    Return the state of a run directory:
      'finished' — status.json present with state == finished
      'failed'   — directory exists but state != finished
      'missing'  — directory does not exist
    """
    if not run_dir.exists():
        return "missing"
    status_file = run_dir / "status.json"
    if not status_file.exists():
        # Directory created but job never wrote status (crashed before startup)
        return "failed"
    try:
        data = json.loads(status_file.read_text())
        state = data.get("state", "").strip()
        return "finished" if state == "finished" else "failed"
    except (json.JSONDecodeError, OSError):
        return "failed"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify tasks from tasks_custom.csv as finished / failed / missing "
            "and write a chunked submit_missing.sh to resubmit all incomplete work."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tasks", default="workflow/tasks_custom.csv",
        help="Tasks CSV (default: workflow/tasks_custom.csv)"
    )
    parser.add_argument(
        "--runs-dir", default="runs",
        help="Path to the runs directory (default: ./runs)"
    )
    parser.add_argument(
        "--slurm-script", default="goad_array_kestrel.slurm",
        help="Slurm batch script to use (default: goad_array_kestrel.slurm)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK,
        help=f"Max task IDs per sbatch command (default: {DEFAULT_CHUNK})"
    )
    parser.add_argument(
        "--throttle", type=int, default=DEFAULT_THROTTLE,
        help=f"Max simultaneous jobs per sbatch command (default: {DEFAULT_THROTTLE})"
    )
    parser.add_argument(
        "--out", default="submit_missing.sh",
        help="Output shell script (default: submit_missing.sh)"
    )
    parser.add_argument(
        "--missing-only", action="store_true",
        help="Only resubmit tasks with no run directory (skip failed/crashed)"
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print every incomplete task in a table"
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    runs_dir   = Path(args.runs_dir)
    out_path   = Path(args.out)

    if not tasks_path.exists():
        print(f"ERROR: {tasks_path} not found.")
        print("Run:  python workflow/make_tasks_custom.py  first.")
        raise SystemExit(1)

    if not runs_dir.exists():
        print(f"WARNING: runs directory not found: {runs_dir}")
        print("Treating all tasks as missing.")

    # ---- Read all tasks ----
    tasks = []
    with tasks_path.open() as f:
        for row in csv.DictReader(f):
            tasks.append(row)

    total = len(tasks)

    # ---- Classify ----
    finished_tasks = []
    failed_tasks   = []
    missing_tasks  = []

    for task in tasks:
        run_name = (
            f"{task['surface']}_{task['adsorbate']}"
            f"_seed{task['seed']}_{task['calculator']}"
        )
        state = get_run_state(runs_dir / run_name)
        if state == "finished":
            finished_tasks.append(task)
        elif state == "failed":
            failed_tasks.append(task)
        else:
            missing_tasks.append(task)

    # ---- Summary ----
    print(f"Tasks CSV      : {tasks_path}  ({total} total)")
    print(f"Runs directory : {runs_dir}")
    print()
    print(f"  Finished              : {len(finished_tasks):>6}")
    print(f"  Failed / not finished : {len(failed_tasks):>6}   "
          f"← dir exists, state != finished")
    print(f"  Missing (no dir)      : {len(missing_tasks):>6}   "
          f"← never started")
    print()

    # Tasks to resubmit
    if args.missing_only:
        to_resubmit = missing_tasks
        label = "missing"
    else:
        to_resubmit = failed_tasks + missing_tasks
        label = "failed + missing"

    if not to_resubmit:
        print(f"Nothing to resubmit ({label}) — all tasks finished!")
        return

    # ---- Optional full table ----
    if args.show_all:
        print(f"{'task_id':<10} {'surface':<10} {'adsorbate':<15} "
              f"{'seed':<6} {'calculator':<20} {'state'}")
        print("-" * 72)
        for t in failed_tasks:
            print(f"{t['task_id']:<10} {t['surface']:<10} "
                  f"{t['adsorbate']:<15} {t['seed']:<6} "
                  f"{t['calculator']:<20} FAILED/RUNNING")
        for t in missing_tasks:
            print(f"{t['task_id']:<10} {t['surface']:<10} "
                  f"{t['adsorbate']:<15} {t['seed']:<6} "
                  f"{t['calculator']:<20} MISSING")
        print()

    # ---- Breakdowns for tasks to resubmit ----
    by_calc = Counter(t["calculator"] for t in to_resubmit)
    print(f"To resubmit ({label}) by calculator:")
    for calc, n in sorted(by_calc.items()):
        print(f"  {calc:<20}: {n} tasks")
    print()

    by_surf = Counter(t["surface"] for t in to_resubmit)
    print(f"To resubmit by surface:")
    for surf, n in sorted(by_surf.items()):
        print(f"  {surf:<10}: {n} tasks")
    print()

    by_mol = Counter(t["adsorbate"] for t in to_resubmit)
    print(f"To resubmit by molecule:")
    for mol, n in sorted(by_mol.items()):
        print(f"  {mol:<15}: {n} tasks")
    print()

    # ---- Build chunked sbatch script ----
    ids      = [t["task_id"] for t in to_resubmit]
    chunks   = list(chunk_list(ids, args.chunk_size))
    n_chunks = len(chunks)

    lines = [
        "#!/bin/bash",
        "# Auto-generated by find_missing_tasks.py",
        f"# Resubmitting {len(ids)} tasks ({label}) in {n_chunks} chunk(s)",
        f"# Chunk size  : {args.chunk_size} IDs",
        f"# Throttle    : {args.throttle} concurrent jobs per chunk",
        f"# Tasks CSV   : {tasks_path}",
        f"# Slurm script: {args.slurm_script}",
        "",
        "set -euo pipefail",
        "",
    ]

    for i, chunk in enumerate(chunks):
        ids_str = ",".join(chunk)
        lines.append(f"# Chunk {i+1}/{n_chunks}  ({len(chunk)} tasks)")
        lines.append(
            f"sbatch --array={ids_str}%{args.throttle} "
            f"{args.slurm_script} {tasks_path}"
        )
        lines.append("")

    out_path.write_text("\n".join(lines))
    out_path.chmod(0o755)

    print(f"Written: {out_path}  ({n_chunks} sbatch command(s))")
    print()
    print("Review and submit with:")
    print(f"  bash {out_path}")
    print()


if __name__ == "__main__":
    main()
