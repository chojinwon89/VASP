import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_tasks_csv(path: Path):
    rows = [
        {
            "task_id": "0",
            "surface": "Cu111",
            "adsorbate": "methanol",
            "seed": "0",
            "calculator": "sevennet_omni",
        },
        {
            "task_id": "1",
            "surface": "Cu111",
            "adsorbate": "ethanol",
            "seed": "0",
            "calculator": "sevennet_omni",
        },
        {
            "task_id": "2",
            "surface": "Pt111",
            "adsorbate": "acetone",
            "seed": "1",
            "calculator": "5m",
        },
    ]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_id", "surface", "adsorbate", "seed", "calculator"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_finished_run(runs_dir: Path, row: dict):
    run_dir = runs_dir / (
        f"{row['surface']}_{row['adsorbate']}_seed{row['seed']}_{row['calculator']}"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"state": "finished"}))


def generate_submit_script(tmp_path: Path, *extra_args: str) -> Path:
    tasks_path = tmp_path / "workflow" / "tasks_custom.csv"
    tasks_path.parent.mkdir()
    write_tasks_csv(tasks_path)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    write_finished_run(
        runs_dir,
        {
            "surface": "Cu111",
            "adsorbate": "methanol",
            "seed": "0",
            "calculator": "sevennet_omni",
        },
    )

    out_path = tmp_path / "submit_missing.sh"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "find_missing_tasks.py"),
            "--tasks",
            str(tasks_path),
            "--runs-dir",
            str(runs_dir),
            "--out",
            str(out_path),
            "--chunk-size",
            "1",
            "--throttle",
            "7",
            *extra_args,
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["bash", "-n", str(out_path)], check=True, cwd=tmp_path)
    return out_path


def test_find_missing_tasks_generates_paced_submit_script_by_default(tmp_path):
    script_path = generate_submit_script(tmp_path)
    script = script_path.read_text()
    lines = script.splitlines()

    assert "# Max in-flight : 9000 (0 = pacing disabled)" in script
    assert "# Poll interval : 60s" in script
    assert "MAX_IN_FLIGHT=9000" in script
    assert "POLL_INTERVAL=60" in script
    assert "wait_for_headroom() {" in script
    assert 'in_flight=$(squeue -u "$USER" -h -t PENDING,RUNNING -r | wc -l)' in script

    sbatch_indexes = [i for i, line in enumerate(lines) if line.startswith("sbatch --array=")]
    assert len(sbatch_indexes) == 2
    for index in sbatch_indexes:
        assert lines[index - 1] == "wait_for_headroom"


def test_find_missing_tasks_can_disable_pacing(tmp_path):
    script_path = generate_submit_script(
        tmp_path,
        "--max-in-flight",
        "0",
        "--poll-interval",
        "15",
    )
    script = script_path.read_text()
    lines = script.splitlines()

    assert "# Max in-flight : 0 (0 = pacing disabled)" in script
    assert "# Poll interval : 15s" in script
    assert "wait_for_headroom() {" not in script
    assert "wait_for_headroom" not in script
    assert 'squeue -u "$USER" -h -t PENDING,RUNNING -r | wc -l' not in script
    assert "MAX_IN_FLIGHT=" not in script
    assert "POLL_INTERVAL=" not in script

    sbatch_indexes = [i for i, line in enumerate(lines) if line.startswith("sbatch --array=")]
    assert len(sbatch_indexes) == 2
    for index in sbatch_indexes:
        assert lines[index - 1].startswith("# Chunk ")
