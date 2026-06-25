#!/usr/bin/env python
"""
find_missing_tasks.py
=====================
Compare workflow/tasks_custom.csv against the runs/ directory and print
the task IDs that have NOT been started yet (no run directory exists).

Usage
-----
    # Default: checks workflow/tasks_custom.csv vs ./runs/
    python find_missing_tasks.py

    # Custom paths
    python find_missing_tasks.py \\
        --tasks workflow/tasks_custom.csv \\
        --runs-dir /scratch/jcho5/goad-global-optimization/runs

    # Print as a Slurm array range (ready to paste into sbatch)
    python find_missing_tasks.py --slurm-array

Output examples
---------------
    Missing 312 tasks.
    Task IDs: 0,3,7,12,...

    # With --slurm-array:
    sbatch --array=0,3,7,12,...%20 goad_array_kestrel.slurm workflow/tasks_custom.csv
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Find task IDs not yet present in the runs/ directory."
    )
    parser.add_argument(
        "--tasks", default="workflow/tasks_custom.csv",
        help="Tasks CSV to check (default: workflow/tasks_custom.csv)"
    )
    parser.add_argument(
        "--runs-dir", default="runs",
        help="Path to the runs directory (default: ./runs)"
    )
    parser.add_argument(
        "--slurm-array", action="store_true",
        help="Also print a ready-to-use sbatch --array= line"
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print every missing task (surface, adsorbate, seed, calculator)"
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    runs_dir   = Path(args.runs_dir)

    if not tasks_path.exists():
        print(f"ERROR: {tasks_path} not found.")
        raise SystemExit(1)

    if not runs_dir.exists():
        print(f"WARNING: runs directory not found: {runs_dir}")
        print("Treating all tasks as missing.")

    # Read all tasks
    tasks = []
    with tasks_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(row)

    total = len(tasks)

    # Find missing ones
    missing = []
    finished = []
    for task in tasks:
        run_name = (
            f"{task['surface']}_{task['adsorbate']}"
            f"_seed{task['seed']}_{task['calculator']}"
        )
        run_path = runs_dir / run_name
        if run_path.exists():
            finished.append(task)
        else:
            missing.append(task)

    print(f"Tasks CSV:      {tasks_path}  ({total} total)")
    print(f"Runs directory: {runs_dir}")
    print()
    print(f"  Already in runs/ : {len(finished)}")
    print(f"  Missing (to run) : {len(missing)}")
    print()

    if not missing:
        print("Nothing missing — all tasks have a run directory.")
        return

    if args.show_all:
        print(f"{'task_id':<10} {'surface':<10} {'adsorbate':<15} {'seed':<6} {'calculator'}")
        print("-" * 60)
        for t in missing:
            print(
                f"{t['task_id']:<10} {t['surface']:<10} "
                f"{t['adsorbate']:<15} {t['seed']:<6} {t['calculator']}"
            )
        print()

    # Summary by calculator
    from collections import Counter
    by_calc = Counter(t["calculator"] for t in missing)
    print("Missing by calculator:")
    for calc, n in sorted(by_calc.items()):
        print(f"  {calc:<20}: {n} tasks")
    print()

    # Summary by surface
    by_surf = Counter(t["surface"] for t in missing)
    print("Missing by surface:")
    for surf, n in sorted(by_surf.items()):
        print(f"  {surf:<10}: {n} tasks")
    print()

    # Summary by molecule
    by_mol = Counter(t["adsorbate"] for t in missing)
    print("Missing by molecule:")
    for mol, n in sorted(by_mol.items()):
        print(f"  {mol:<15}: {n} tasks")
    print()

    # Comma-separated task IDs
    ids = [t["task_id"] for t in missing]
    ids_str = ",".join(ids)
    print(f"Missing task IDs ({len(ids)}):")
    print(ids_str)
    print()

    if args.slurm_array:
        tasks_arg = str(tasks_path)
        print("Submit missing tasks:")
        print(f"  sbatch --array={ids_str}%20 goad_array_kestrel.slurm {tasks_arg}")
        print()


if __name__ == "__main__":
    main()
