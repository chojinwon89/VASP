from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    return subprocess.run(
        [sys.executable, "pipeline.py", *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )


def test_pipeline_imports_and_registry_is_consistent():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import pipeline
    finally:
        sys.path.pop(0)
    # every id in the canonical order resolves, and scopes are known
    assert pipeline.ALL_ORDER, "no stages registered"
    for sid in pipeline.ALL_ORDER:
        stage = pipeline.STAGE_BY_ID[sid]
        assert stage.scope in {"local", "cluster"}
    # the conductor must not re-implement science: each stage builds argv lists
    class _Args:
        dry_run = True
        cluster = "perlmutter-cpu"
        jobs_dir = "dft_jobs"
        gallery = "structure"
        analysis_dir = "analysis_out"
        analysis_dir_val = "analysis_nomag"
        pages_dir = None
    for sid in pipeline.ALL_ORDER:
        cmds = pipeline.STAGE_BY_ID[sid].build(_Args())
        assert cmds and all(isinstance(c, list) for c in cmds)


def test_pipeline_list_runs():
    r = _run("list")
    assert r.returncode == 0, r.stderr
    assert "structures" in r.stdout and "website" in r.stdout


def test_pipeline_all_dry_run_plans_without_executing():
    r = _run("all", "--dry-run")
    assert r.returncode == 0, r.stderr
    # canonical order is previewed end-to-end
    for sid in ("structures", "dft-setup", "energies", "website"):
        assert f"[{sid}]" in r.stdout
    # dry-run must not actually shell out to a generator
    assert "generation complete" not in r.stdout.lower()


def test_unknown_stage_is_rejected():
    r = _run("run", "does-not-exist")
    assert r.returncode != 0
    assert "unknown stage" in (r.stdout + r.stderr).lower()
