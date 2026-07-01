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

    # Print every NOT-finished task (failed + missing) with their current state
    python find_missing_tasks.py --show-not-finished

    # Show not-finished filtered by calculator or surface
    python find_missing_tasks.py --show-not-finished --filter-calc sevennet_omni
    python find_missing_tasks.py --show-not-finished --filter-surface Cu111

Output
------
    Tasks CSV      : workflow/tasks_custom.csv  (1296 total)
    Runs directory : runs

      Finished              :  324
      Failed / not finished :  125   <- dir exists, state != finished
      Missing (no dir)      :  847   <- never started

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
        return "failed"
    try:
        data  = json.loads(status_file.read_text())
        state = data.get("state", "").strip()
        return "finished" if state == "finished" else state or "failed"
    except (json.JSONDecodeError, OSError):
        return "failed"


def show_not_finished(failed_tasks, missing_tasks,
                      filter_calc=None, filter_surface=None):
    """
    Print a detailed table of every not-finished task with its current state.
    Optionally filter by calculator and/or surface.
    """
    all_incomplete = [
        (t, "FAILED/RUNNING") for t in failed_tasks
    ] + [
        (t, "MISSING") for t in missing_tasks
    ]

    # Apply optional filters
    if filter_calc:
        all_incomplete = [(t, s) for t, s in all_incomplete
                          if t["calculator"] == filter_calc]
    if filter_surface:
        all_incomplete = [(t, s) for t, s in all_incomplete
                          if t["surface"] == filter_surface]

    if not all_incomplete:
        print("  (no not-finished tasks match the filter)")
        return

    # Summary counts per state
    state_counts = Counter(s for _, s in all_incomplete)
    print("Not-finished tasks{}: {} total".format(
        " [calc={}{]]".format(
            filter_calc or "",
            ", surface=" + filter_surface if filter_surface else ""
        ) if filter_calc or filter_surface else "",
        len(all_incomplete)
    ))
    for state, n in sorted(state_counts.items()):
        print("  {:<20}: {}".format(state, n))
    print()

    # Per-calculator breakdown
    by_calc = Counter(t["calculator"] for t, _ in all_incomplete)
    print("  By calculator:")
    for calc, n in sorted(by_calc.items()):
        print("    {:<20}: {}".format(calc, n))
    print()

    # Per-surface breakdown
    by_surf = Counter(t["surface"] for t, _ in all_incomplete)
    print("  By surface:")
    for surf, n in sorted(by_surf.items()):
        print("    {:<12}: {}".format(surf, n))
    print()

    # Per-molecule breakdown
    by_mol = Counter(t["adsorbate"] for t, _ in all_incomplete)
    print("  By molecule:")
    for mol, n in sorted(by_mol.items()):
        print("    {:<15}: {}".format(mol, n))
    print()

    # Full table
    col_w = [10, 10, 15, 6, 20, 16]
    header = "{:<{}} {:<{}} {:<{}} {:<{}} {:<{}} {:<{}}".format(
        "task_id",   col_w[0],
        "surface",   col_w[1],
        "adsorbate", col_w[2],
        "seed",      col_w[3],
        "calculator",col_w[4],
        "state",     col_w[5],
    )
    print(header)
    print("-" * (sum(col_w) + 5))
    for t, state in sorted(all_incomplete, key=lambda x: (
            x[1], x[0]["calculator"], x[0]["surface"], x[0]["adsorbate"])):
        print("{:<{}} {:<{}} {:<{}} {:<{}} {:<{}} {:<{}}".format(
            t.get("task_id", "?"), col_w[0],
            t["surface"],          col_w[1],
            t["adsorbate"],        col_w[2],
            t.get("seed", "?"),    col_w[3],
            t["calculator"],       col_w[4],
            state,                 col_w[5],
        ))
    print()


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
        help="Max task IDs per sbatch command (default: {})".format(DEFAULT_CHUNK)
    )
    parser.add_argument(
        "--throttle", type=int, default=DEFAULT_THROTTLE,
        help="Max simultaneous jobs per sbatch command (default: {})".format(
             DEFAULT_THROTTLE)
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
        help="Print every incomplete task in a compact table"
    )
    parser.add_argument(
        "--show-not-finished", action="store_true",
        help="Print detailed table of all not-finished tasks "
             "(failed + missing) with state, breakdowns by "
             "calculator / surface / molecule"
    )
    parser.add_argument(
        "--filter-calc", default=None,
        metavar="CALC",
        help="With --show-not-finished: only show tasks for this calculator"
    )
    parser.add_argument(
        "--filter-surface", default=None,
        metavar="SURFACE",
        help="With --show-not-finished: only show tasks for this surface"
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    runs_dir   = Path(args.runs_dir)
    out_path   = Path(args.out)

    if not tasks_path.exists():
        print("ERROR: {} not found.".format(tasks_path))
        print("Run:  python workflow/make_tasks_custom.py  first.")
        raise SystemExit(1)

    if not runs_dir.exists():
        print("WARNING: runs directory not found: {}".format(runs_dir))
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
            "{surface}_{adsorbate}_seed{seed}_{calculator}".format(**task)
        )
        state = get_run_state(runs_dir / run_name)
        if state == "finished":
            finished_tasks.append(task)
        elif state == "missing":
            missing_tasks.append(task)
        else:
            failed_tasks.append(task)

    # ---- Summary ----
    print("Tasks CSV      : {}  ({} total)".format(tasks_path, total))
    print("Runs directory : {}".format(runs_dir))
    print()
    print("  Finished              : {:>6}".format(len(finished_tasks)))
    print("  Failed / not finished : {:>6}   "
          "<- dir exists, state != finished".format(len(failed_tasks)))
    print("  Missing (no dir)      : {:>6}   "
          "<- never started".format(len(missing_tasks)))
    print()

    # ---- --show-not-finished ----
    if args.show_not_finished:
        show_not_finished(failed_tasks, missing_tasks,
                          filter_calc=args.filter_calc,
                          filter_surface=args.filter_surface)

    # Tasks to resubmit
    if args.missing_only:
        to_resubmit = missing_tasks
        label = "missing"
    else:
        to_resubmit = failed_tasks + missing_tasks
        label = "failed + missing"

    if not to_resubmit:
        print("Nothing to resubmit ({}) — all tasks finished!".format(label))
        return

    # ---- Optional compact table (original --show-all) ----
    if args.show_all:
        print("{:<10} {:<10} {:<15} {:<6} {:<20} {}".format(
              "task_id", "surface", "adsorbate", "seed", "calculator", "state"))
        print("-" * 72)
        for t in failed_tasks:
            print("{:<10} {:<10} {:<15} {:<6} {:<20} FAILED/RUNNING".format(
                  t.get("task_id", "?"), t["surface"], t["adsorbate"],
                  t.get("seed", "?"), t["calculator"]))
        for t in missing_tasks:
            print("{:<10} {:<10} {:<15} {:<6} {:<20} MISSING".format(
                  t.get("task_id", "?"), t["surface"], t["adsorbate"],
                  t.get("seed", "?"), t["calculator"]))
        print()

    # ---- Breakdowns for tasks to resubmit ----
    by_calc = Counter(t["calculator"] for t in to_resubmit)
    print("To resubmit ({}) by calculator:".format(label))
    for calc, n in sorted(by_calc.items()):
        print("  {:<20}: {} tasks".format(calc, n))
    print()

    by_surf = Counter(t["surface"] for t in to_resubmit)
    print("To resubmit by surface:")
    for surf, n in sorted(by_surf.items()):
        print("  {:<10}: {} tasks".format(surf, n))
    print()

    by_mol = Counter(t["adsorbate"] for t in to_resubmit)
    print("To resubmit by molecule:")
    for mol, n in sorted(by_mol.items()):
        print("  {:<15}: {} tasks".format(mol, n))
    print()

    # ---- Build chunked sbatch script ----
    ids      = [t["task_id"] for t in to_resubmit]
    chunks   = list(chunk_list(ids, args.chunk_size))
    n_chunks = len(chunks)

    lines = [
        "#!/bin/bash",
        "# Auto-generated by find_missing_tasks.py",
        "# Resubmitting {} tasks ({}) in {} chunk(s)".format(
            len(ids), label, n_chunks),
        "# Chunk size  : {}".format(args.chunk_size),
        "# Throttle    : {} concurrent jobs per chunk".format(args.throttle),
        "# Tasks CSV   : {}".format(tasks_path),
        "# Slurm script: {}".format(args.slurm_script),
        "",
        "set -euo pipefail",
        "",
    ]

    for i, chunk in enumerate(chunks):
        ids_str = ",".join(chunk)
        lines.append("# Chunk {}/{}  ({} tasks)".format(
            i + 1, n_chunks, len(chunk)))
        lines.append(
            "sbatch --array={}%{} {} {}".format(
                ids_str, args.throttle, args.slurm_script, tasks_path)
        )
        lines.append("")

    out_path.write_text("\n".join(lines))
    out_path.chmod(0o755)

    print("Written: {}  ({} sbatch command(s))".format(out_path, n_chunks))
    print()
    print("Review and submit with:")
    print("  bash {}".format(out_path))
    print()


if __name__ == "__main__":
    main()
