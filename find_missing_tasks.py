#!/usr/bin/env python
"""
find_missing_tasks.py
=====================
Compare workflow/tasks_custom.csv against the runs/ directory and print
the task IDs that have NOT been started yet (no run directory exists).

Automatically writes submit_missing.sh with sbatch commands chunked to
stay under Slurm's --array argument length limit (~4096 chars).

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

    # Also print every missing task in a table
    python find_missing_tasks.py --show-all

Output
------
    Missing 847 tasks.
    Missing by calculator:
      5m             : 648 tasks
      sevennet_omni  : 199 tasks
    ...
    Written: submit_missing.sh  (5 sbatch commands)
    Run with:  bash submit_missing.sh
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


# Slurm's hard limit on --array= argument is ~4096 chars.
# With 4-digit IDs and commas, ~200 IDs ≈ ~1000 chars — safely under.
DEFAULT_CHUNK  = 200
DEFAULT_THROTTLE = 20


def chunk_list(lst, size):
    """Split lst into sublists of at most `size` elements."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find task IDs not yet present in the runs/ directory and "
            "write a chunked submit_missing.sh ready to run with bash."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--show-all", action="store_true",
        help="Print every missing task in a table"
    )
    args = parser.parse_args()

    tasks_path  = Path(args.tasks)
    runs_dir    = Path(args.runs_dir)
    out_path    = Path(args.out)

    if not tasks_path.exists():
        print(f"ERROR: {tasks_path} not found. Run: python workflow/make_tasks_custom.py first.")
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

    # ---- Classify missing vs present ----
    missing  = []
    finished = []
    for task in tasks:
        run_name = (
            f"{task['surface']}_{task['adsorbate']}"
            f"_seed{task['seed']}_{task['calculator']}"
        )
        if (runs_dir / run_name).exists():
            finished.append(task)
        else:
            missing.append(task)

    # ---- Summary ----
    print(f"Tasks CSV      : {tasks_path}  ({total} total)")
    print(f"Runs directory : {runs_dir}")
    print()
    print(f"  Already in runs/ : {len(finished)}")
    print(f"  Missing (to run) : {len(missing)}")
    print()

    if not missing:
        print("Nothing missing — all tasks have a run directory. No script written.")
        return

    # ---- Optional full table ----
    if args.show_all:
        print(f"{'task_id':<10} {'surface':<10} {'adsorbate':<15} {'seed':<6} {'calculator'}")
        print("-" * 60)
        for t in missing:
            print(
                f"{t['task_id']:<10} {t['surface']:<10} "
                f"{t['adsorbate']:<15} {t['seed']:<6} {t['calculator']}"
            )
        print()

    # ---- Breakdowns ----
    by_calc = Counter(t["calculator"] for t in missing)
    print("Missing by calculator:")
    for calc, n in sorted(by_calc.items()):
        print(f"  {calc:<20}: {n} tasks")
    print()

    by_surf = Counter(t["surface"] for t in missing)
    print("Missing by surface:")
    for surf, n in sorted(by_surf.items()):
        print(f"  {surf:<10}: {n} tasks")
    print()

    by_mol = Counter(t["adsorbate"] for t in missing)
    print("Missing by molecule:")
    for mol, n in sorted(by_mol.items()):
        print(f"  {mol:<15}: {n} tasks")
    print()

    # ---- Build chunked sbatch commands ----
    ids = [t["task_id"] for t in missing]
    chunks = list(chunk_list(ids, args.chunk_size))
    n_chunks = len(chunks)

    lines = [
        "#!/bin/bash",
        "# Auto-generated by find_missing_tasks.py",
        f"# Submitting {len(ids)} missing tasks in {n_chunks} chunk(s)",
        f"# Chunk size : {args.chunk_size} IDs",
        f"# Throttle   : {args.throttle} concurrent jobs per chunk",
        f"# Tasks CSV  : {tasks_path}",
        f"# Slurm script: {args.slurm_script}",
        "",
        "set -euo pipefail",
        "",
    ]

    for i, chunk in enumerate(chunks):
        ids_str  = ",".join(chunk)
        comment  = f"# Chunk {i+1}/{n_chunks}  ({len(chunk)} tasks)"
        cmd = (
            f"sbatch --array={ids_str}%{args.throttle} "
            f"{args.slurm_script} {tasks_path}"
        )
        lines.append(comment)
        lines.append(cmd)
        lines.append("")

    script = "\n".join(lines)
    out_path.write_text(script)
    out_path.chmod(0o755)

    print(f"Written: {out_path}  ({n_chunks} sbatch command(s))")
    print()
    print("Review and submit with:")
    print(f"  bash {out_path}")
    print()


if __name__ == "__main__":
    main()
