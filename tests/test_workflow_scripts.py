import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_make_tasks_generates_expected_task_matrix(tmp_path):
    workflow_dir = tmp_path / "workflow"
    workflow_dir.mkdir()
    (workflow_dir / "make_tasks.py").write_text(
        (REPO_ROOT / "workflow" / "make_tasks.py").read_text()
    )

    subprocess.run(
        [sys.executable, "workflow/make_tasks.py"],
        cwd=tmp_path,
        check=True,
    )

    with (workflow_dir / "tasks.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 9
    assert [row["surface"] for row in rows[:3]] == ["Cu111", "Cu111", "Cu111"]
    assert [row["surface"] for row in rows[3:6]] == ["Cu110", "Cu110", "Cu110"]
    assert [row["surface"] for row in rows[6:9]] == ["Cu001", "Cu001", "Cu001"]
    assert {row["adsorbate"] for row in rows} == {"isopropanol"}
    assert {row["calculator"] for row in rows} == {"sevennet_omni"}


def test_collect_results_reads_per_task_result_json(tmp_path):
    run_dir = tmp_path / "runs" / "Cu001_isopropanol_seed2_sevennet_omni"
    run_dir.mkdir(parents=True)
    workflow_dir = tmp_path / "workflow"
    workflow_dir.mkdir()
    (tmp_path / "collect_results.py").write_text(
        (REPO_ROOT / "collect_results.py").read_text()
    )

    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "task_id": 8,
                "surface": "Cu001",
                "adsorbate": "isopropanol",
                "seed": 2,
                "calculator": "sevennet_omni",
                "run_dir": str(run_dir),
                "started_at": "2026-06-16T00:00:00",
                "finished_at": "2026-06-16T01:00:00",
                "state": "finished",
            }
        )
    )
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "system": "isopropanol_on_Cu001",
                "calculator": "sevennet_omni",
                "ga": {"best_seed": 2},
                "E_ads_eV": -1.23,
                "E_ads_pre_relax_eV": -0.98,
            }
        )
    )

    result = subprocess.run(
        [sys.executable, "collect_results.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    with (workflow_dir / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "task_id": "8",
            "surface": "Cu001",
            "adsorbate": "isopropanol",
            "seed": "2",
            "calculator": "sevennet_omni",
            "state": "finished",
            "E_ads_eV": "-1.23",
            "E_ads_pre_relax_eV": "-0.98",
            "best_seed": "2",
            "run_dir": str(run_dir),
            "started_at": "2026-06-16T00:00:00",
            "finished_at": "2026-06-16T01:00:00",
        }
    ]
    assert "task_id" in result.stdout
    assert "Cu001" in result.stdout
