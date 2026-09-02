#!/usr/bin/env python
"""
pipeline.py -- one driver for the GOAD+SevenNet -> DFT benchmark
================================================================
A thin *conductor* that chains the existing pipeline scripts into named
stages so the whole benchmark can be driven with one command instead of
copy-pasting the recipe from PIPELINE.md.  It does **not** re-implement any
science -- every stage just shells out to the same scripts documented in
PIPELINE.md, so behaviour stays identical and there is a single source of
truth for each step.

    structure creation -> MLIP (GOAD/SevenNet) -> DFT geo-opt
        -> error detect & auto-fix -> binding energy -> MLIP-vs-DFT
        -> website refresh

Stages are tagged **local** (runs on the laptop, only needs `ase`) or
**cluster** (needs `sbatch` / VASP on Perlmutter or Kestrel).  On a machine
without `sbatch`, cluster stages are previewed (dry-run) instead of failing.

Usage
-----
    python pipeline.py list                 # show all stages
    python pipeline.py status               # coverage gate (what is missing?)
    python pipeline.py run structures       # run one stage
    python pipeline.py run dft-setup dft-submit --cluster perlmutter-cpu
    python pipeline.py all --dry-run        # print the whole plan, run nothing
    python pipeline.py all                   # run the canonical chain in order

Common flags
------------
    --dry-run            print every command, execute nothing
    --cluster NAME       perlmutter-cpu (default) or kestrel  (DFT stages)
    --jobs-dir DIR       staged DFT jobs tree            (default: dft_jobs)
    --gallery DIR        MLIP CIF gallery                (default: structure)
    --analysis-dir DIR   CSV / plot output dir           (default: analysis_out)
    --pages-dir DIR      local bond-distance-review clone (website stage)
    --continue-on-error  keep going past a failed/blocked stage
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

REPO = Path(__file__).resolve().parent
PY = sys.executable  # run every child with the same interpreter


# --------------------------------------------------------------------------- #
# Stage registry
# --------------------------------------------------------------------------- #
@dataclass
class Stage:
    id: str
    scope: str  # "local" or "cluster"
    summary: str
    build: Callable[[argparse.Namespace], List[List[str]]]
    notes: List[str] = field(default_factory=list)
    advisory: bool = False  # nonzero exit informs but never aborts the chain


def _funcs() -> List[str]:
    return ["pbe", "pbe-d3", "r2scan", "beef-vdw"]


def stage_structures(a) -> List[List[str]]:
    # generate_*_cifs.py take no args; they write CIFs into inputs/ (idempotent).
    return [[PY, "generate_surface_cifs.py"], [PY, "generate_molecule_cifs.py"]]


def stage_coverage(a) -> List[List[str]]:
    return [[PY, "workflow/check_dft_coverage.py"]]


def stage_mlip_tasks(a) -> List[List[str]]:
    return [
        [PY, "workflow/make_tasks_missing_sevennet.py"],
        [PY, "workflow/make_tasks_existing_gap.py"],
    ]


def stage_mlip_collect(a) -> List[List[str]]:
    return [
        [PY, "workflow/collect_missing_sevennet.py",
         "--tasks-csv", "workflow/tasks_missing_sevennet.csv"],
        [PY, "workflow/collect_missing_sevennet.py",
         "--tasks-csv", "workflow/tasks_existing_gap.csv"],
        [PY, "workflow/import_sevennet_to_gallery.py"],
        [PY, "workflow/stage_dft_poscars.py",
         "--manifest", "DFT_results/MANIFEST.csv",
         "--structure-dir", a.gallery, "--out-dir", a.jobs_dir,
         "--fix-bottom-layers", "2"],
        [PY, "workflow/check_dft_coverage.py"],
    ]


def stage_dft_setup(a) -> List[List[str]]:
    return [
        [PY, "setup_vasp_jobs.py", "--poscar-dir", a.jobs_dir,
         "--functional", f, "--cluster", a.cluster, "--skip-existing"]
        for f in _funcs()
    ]


def stage_dft_submit(a) -> List[List[str]]:
    # Emit a joblist per functional; the array submit itself is an sbatch line
    # printed by --emit-joblist / make_dft_joblist and left to the user's QOS.
    return [
        [PY, "workflow/make_dft_joblist.py", "--jobs-dir", a.jobs_dir,
         "--functional", f]
        for f in _funcs()
    ]


def stage_dft_fix(a) -> List[List[str]]:
    # Error DETECT + resubmit: classifies every OUTCAR (done/running/unconverged/
    # crashed/not_started) and resubmits only the ones that still need it.
    return [
        [PY, "workflow/resubmit_dft.py", "--jobs-dir", a.jobs_dir,
         "--functional", "all", "--restart-from-contcar", "--submit"]
    ]


def stage_auto(a) -> List[List[str]]:
    # Autonomous submit -> convergence-check -> rule-based INCAR patch -> retry
    # loop (automation/). One pass with --once; drop it for a continuous daemon.
    return [[PY, "automation/runner.py",
             "--config", "automation/config.yaml", "--once"]]


def stage_energies(a) -> List[List[str]]:
    ad = a.analysis_dir
    return [
        [PY, "calc_binding_energy.py",
         "--best-dirs", "poscar/best", "poscar/best2", a.jobs_dir,
         "--slab-dir", "vasp_slab", "--mol-dir", "vasp_mol",
         "--all-functionals", "--functionals", "PBE", "PBE_D3", "r2scan", "beef_vdw",
         "--calc-type", "fully-relaxed", "--only-complete",
         "--output", f"{ad}/dft_binding_energies_all.csv"],
        [PY, "plot_dft_vs_mlip.py",
         "--dft", f"{ad}/dft_binding_energies_all.csv",
         "--ml", "workflow/summary.csv",
         "--calculators", "sevennet_omni",
         "--csv-out", f"{ad}/dft_vs_mlip_pairs.csv",
         "--output", f"{ad}/dft_vs_mlip_all.png"],
        [PY, "analyze_dft_mlip_accuracy.py",
         "--pairs", f"{ad}/dft_vs_mlip_pairs.csv", "--out-dir", ad],
    ]


def stage_geometry(a) -> List[List[str]]:
    ad = a.analysis_dir
    return [
        [PY, "mlip_contact_geometry.py",
         "--gallery", a.gallery, "--out", f"{ad}/mlip_geom.csv"],
        [PY, "dft_contact_geometry.py",
         "--dft-jobs", "poscar/best", "--dft-jobs", "poscar/best2",
         "--dft-jobs", a.jobs_dir,
         "--targets", f"{ad}/dft_vs_mlip_pairs.csv",
         "--out", f"{ad}/dft_geom.csv"],
        [PY, "render_dft_structures.py",
         "--dft-jobs", "poscar/best", "--dft-jobs", "poscar/best2",
         "--dft-jobs", a.jobs_dir,
         "--targets", f"{ad}/dft_vs_mlip_pairs.csv", "--out-dir", "dft_png"],
    ]


def stage_website(a) -> List[List[str]]:
    pages = a.pages_dir
    if not pages:
        if a.dry_run:
            pages = "<pages-dir>"
        else:
            raise SystemExit(
                "website stage needs --pages-dir <local bond-distance-review clone>")
    ad = a.analysis_dir
    return [
        [PY, "build_dft_pages.py",
         "--analysis-dir", ad, "--gallery", a.gallery,
         "--out-dir", pages,
         "--struct-compare", f"{ad}/dft_geom.csv", "--dft-png-dir", "dft_png"],
        [PY, "build_validation_page.py",
         "--analysis-dir", a.analysis_dir_val,
         "--dft-geom", f"{ad}/dft_geom.csv", "--out-dir", pages],
    ]


STAGES: List[Stage] = [
    Stage("structures", "local",
          "Build surface + molecule input CIFs (inputs/)", stage_structures),
    Stage("coverage", "local",
          "Coverage gate: reconcile MANIFEST vs staged dft_jobs/", stage_coverage,
          advisory=True),
    Stage("mlip-tasks", "cluster",
          "Emit GOAD/SevenNet gap-fill task lists (Stage A)", stage_mlip_tasks,
          notes=["Then submit the GPU arrays, e.g.:",
                 "  sbatch --array=0-587%50 -t 00:20:00 "
                 "perlmutter/goad_array_perlmutter_gpu.slurm "
                 "workflow/tasks_missing_sevennet.csv"]),
    Stage("mlip-collect", "cluster",
          "Collect best seed -> import to gallery -> stage POSCARs -> verify",
          stage_mlip_collect),
    Stage("dft-setup", "cluster",
          "Write INCAR/KPOINTS/POTCAR/slurm for all 4 functionals (Stage B)",
          stage_dft_setup),
    Stage("dft-submit", "cluster",
          "Build the per-functional joblists for the VASP CPU array",
          stage_dft_submit,
          notes=["Submit each list, e.g.:",
                 "  N=$(grep -vc '^#' joblist_pbe.txt)",
                 "  sbatch --array=0-$((N-1))%20 "
                 "perlmutter/vasp_dft_array_cpu.slurm joblist_pbe.txt"]),
    Stage("dft-fix", "cluster",
          "Detect failed/unconverged DFT jobs and resubmit (error recovery)",
          stage_dft_fix),
    Stage("auto", "cluster",
          "Autonomous submit->check->INCAR-patch->retry loop (automation/)",
          stage_auto),
    Stage("energies", "cluster",
          "DFT E_ads -> MLIP-vs-DFT pairs + parity plot + accuracy report",
          stage_energies),
    Stage("geometry", "local",
          "MLIP + DFT contact geometry CSVs and DFT final-structure PNGs",
          stage_geometry),
    Stage("website", "local",
          "Rebuild the live DFT + method-validation pages", stage_website),
]

STAGE_BY_ID = {s.id: s for s in STAGES}

# Canonical order for `all` (stage ids).
ALL_ORDER = [s.id for s in STAGES]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def have_sbatch() -> bool:
    return shutil.which("sbatch") is not None


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


def run_stage(stage: Stage, a) -> int:
    """Return 0 ok, non-zero on failure, -1 = skipped/blocked."""
    cluster_blocked = stage.scope == "cluster" and not a.dry_run and not have_sbatch()
    header = f"[{stage.id}] ({stage.scope}) {stage.summary}"
    print("\n" + _bold(f"==> {header}"))

    try:
        cmds = stage.build(a)
    except SystemExit as e:
        print(f"  ! {e}")
        return 1

    if cluster_blocked:
        print("  · no `sbatch` here -> preview only (run this stage on the "
              "cluster):")

    for note in stage.notes:
        print(f"    {note}")

    rc = 0
    abort_on_fail = not (a.continue_on_error or stage.advisory)
    for cmd in cmds:
        printable = " ".join(cmd)
        if a.dry_run or cluster_blocked:
            print(f"    $ {printable}")
            continue
        print(f"  $ {printable}")
        proc = subprocess.run(cmd, cwd=str(REPO))
        if proc.returncode != 0:
            note = "reported gaps/nonzero" if stage.advisory else "command failed"
            print(f"  ! {note} (exit {proc.returncode}): {printable}")
            rc = proc.returncode
            if abort_on_fail:
                return rc
    if cluster_blocked:
        return -1
    if stage.advisory:
        return 0
    return rc


def cmd_list(a) -> int:
    print("Pipeline stages (canonical order):\n")
    for s in STAGES:
        tag = "🖥 local  " if s.scope == "local" else "☁ cluster"
        print(f"  {tag}  {s.id:<13} {s.summary}")
    print("\nRun one:   python pipeline.py run <stage> [<stage> ...]")
    print("Run all:   python pipeline.py all [--dry-run]")
    return 0


def cmd_status(a) -> int:
    print("Coverage / health gate:")
    subprocess.run([PY, "workflow/check_dft_coverage.py"], cwd=str(REPO))
    return 0  # a report, not a pass/fail signal


def cmd_run(a) -> int:
    unknown = [s for s in a.stages if s not in STAGE_BY_ID]
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(unknown)}\n"
                         f"known: {', '.join(STAGE_BY_ID)}")
    worst = 0
    for sid in a.stages:
        rc = run_stage(STAGE_BY_ID[sid], a)
        if rc > 0 and not a.continue_on_error:
            return rc
        if rc > 0:
            worst = rc
    return worst


def cmd_all(a) -> int:
    print("Canonical pipeline "
          + ("(dry-run preview)" if a.dry_run else "(executing)"))
    worst = 0
    for sid in ALL_ORDER:
        rc = run_stage(STAGE_BY_ID[sid], a)
        if rc == -1:  # cluster-blocked on a non-cluster host
            if a.continue_on_error:
                continue
            print("\nStopped at the first cluster stage (no `sbatch` on this "
                  "host).\nContinue on the cluster, or preview with "
                  "`python pipeline.py all --dry-run`.")
            return 0
        if rc > 0 and not a.continue_on_error:
            return rc
        if rc > 0:
            worst = rc
    return worst


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    # Shared flags live on a parent parser so they are accepted both before and
    # after the subcommand (`pipeline.py --dry-run all` and `pipeline.py all
    # --dry-run` both work).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true",
                        help="print every command, execute nothing")
    common.add_argument("--cluster", default="perlmutter-cpu",
                        choices=["perlmutter-cpu", "kestrel"],
                        help="target cluster for DFT stages (default: perlmutter-cpu)")
    common.add_argument("--jobs-dir", default="dft_jobs", help="staged DFT jobs tree")
    common.add_argument("--gallery", default="structure", help="MLIP CIF gallery dir")
    common.add_argument("--analysis-dir", default="analysis_out",
                        help="CSV / plot output dir")
    common.add_argument("--analysis-dir-val", default="analysis_nomag",
                        help="analysis dir for the validation page (non-magnetic set)")
    common.add_argument("--pages-dir", default=os.environ.get("PAGES_DIR"),
                        help="local bond-distance-review clone (website stage)")
    common.add_argument("--continue-on-error", action="store_true",
                        help="keep going past a failed or blocked stage")

    p = argparse.ArgumentParser(
        parents=[common],
        description="Conductor for the GOAD+SevenNet -> DFT benchmark pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See PIPELINE.md for the full manual recipe behind each stage.")

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", parents=[common],
                   help="list all stages").set_defaults(func=cmd_list)
    sub.add_parser("status", parents=[common],
                   help="coverage gate").set_defaults(func=cmd_status)
    pr = sub.add_parser("run", parents=[common],
                        help="run one or more named stages")
    pr.add_argument("stages", nargs="+", help="stage ids (see `list`)")
    pr.set_defaults(func=cmd_run)
    sub.add_parser("all", parents=[common],
                   help="run the canonical chain").set_defaults(func=cmd_all)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
