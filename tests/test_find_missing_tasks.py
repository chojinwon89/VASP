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
    sys.path.insert(0, str(REPO_ROOT))
    from molecule_utils import carbon_count

    run_dir = runs_dir / f"C{carbon_count(row['adsorbate'])}" / (
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
        # Line before sbatch is TASK_IDS_N=(...), line before that is # Chunk ...
        assert lines[index - 1].startswith("TASK_IDS_")
        assert lines[index - 2].startswith("# Chunk ")


def test_find_missing_tasks_uses_contiguous_zero_based_array_indices(tmp_path):
    """--array= must use 0-based contiguous ranges, not raw task_ids."""
    script_path = generate_submit_script(tmp_path)
    script = script_path.read_text()
    lines = script.splitlines()

    sbatch_lines = [l for l in lines if l.startswith("sbatch --array=")]
    assert len(sbatch_lines) == 2

    for sbatch_line in sbatch_lines:
        # Extract the array spec between '--array=' and '%'
        array_spec = sbatch_line.split("--array=")[1].split("%")[0]
        # Must be a 0-N range, e.g. "0-0"
        assert "-" in array_spec, "Expected a range like 0-N, got: {}".format(array_spec)
        start, end = array_spec.split("-")
        assert start == "0", "Array range must start at 0, got: {}".format(start)
        assert end.isdigit(), "Array range end must be a digit, got: {}".format(end)
        # With --chunk-size 1, each chunk has 1 task → array 0-0
        assert end == "0"

    # The NOTE comment about 0-based indices must appear in the header
    assert "0-based indices" in script


def test_find_missing_tasks_provides_real_task_ids_via_env_var(tmp_path):
    """Real task_ids must be provided in TASK_IDS_N=(...) bash arrays."""
    script_path = generate_submit_script(tmp_path)
    script = script_path.read_text()
    lines = script.splitlines()

    # Two missing tasks (task_id 1 and 2); with --chunk-size 1 → two chunks
    task_id_lines = [l for l in lines if l.startswith("TASK_IDS_")]
    assert len(task_id_lines) == 2

    # Each TASK_IDS_N line must contain the real task_id (not just "0")
    # Task 1 → chunk 1, task 2 → chunk 2
    assert "TASK_IDS_1=(1)" in script
    assert "TASK_IDS_2=(2)" in script

    # sbatch lines must use --export=ALL,GOAD_TASK_ID_LIST=...
    sbatch_lines = [l for l in lines if l.startswith("sbatch --array=")]
    for sbatch_line in sbatch_lines:
        assert "GOAD_TASK_ID_LIST=" in sbatch_line


def test_find_missing_tasks_task_id_mapping_order_preserved(tmp_path):
    """Task ids must appear in order within each chunk's TASK_IDS variable."""
    # Use a larger chunk so multiple ids appear in one chunk
    tasks_path = tmp_path / "workflow" / "tasks_custom.csv"
    tasks_path.parent.mkdir()

    import csv as csv_mod
    rows = [
        {"task_id": str(i), "surface": "Cu111", "adsorbate": "methanol",
         "seed": str(i), "calculator": "sevennet_omni"}
        for i in range(5)
    ]
    with tasks_path.open("w", newline="") as f:
        writer = csv_mod.DictWriter(
            f, fieldnames=["task_id", "surface", "adsorbate", "seed", "calculator"])
        writer.writeheader()
        writer.writerows(rows)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    out_path = tmp_path / "submit_missing.sh"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "find_missing_tasks.py"),
            "--tasks", str(tasks_path),
            "--runs-dir", str(runs_dir),
            "--out", str(out_path),
            "--chunk-size", "3",
            "--throttle", "5",
            "--max-in-flight", "0",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["bash", "-n", str(out_path)], check=True, cwd=tmp_path)

    script = out_path.read_text()
    lines = script.splitlines()

    # 5 tasks, chunk-size 3 → 2 chunks (3 + 2)
    task_id_lines = [l for l in lines if l.startswith("TASK_IDS_")]
    assert len(task_id_lines) == 2

    # First chunk: ids 0,1,2 → TASK_IDS_1=(0 1 2)
    assert "TASK_IDS_1=(0 1 2)" in script
    # Second chunk: ids 3,4 → TASK_IDS_2=(3 4)
    assert "TASK_IDS_2=(3 4)" in script

    # Array ranges: chunk1 → 0-2, chunk2 → 0-1
    sbatch_lines = [l for l in lines if l.startswith("sbatch --array=")]
    assert len(sbatch_lines) == 2
    assert "0-2%" in sbatch_lines[0]
    assert "0-1%" in sbatch_lines[1]

    # No raw task_id > chunk_size appears inside --array=
    for sbatch_line in sbatch_lines:
        array_spec = sbatch_line.split("--array=")[1].split("%")[0]
        # Must be a simple range, not a comma-separated list
        assert "," not in array_spec


def test_find_missing_tasks_large_task_ids_stay_under_max_array_size(tmp_path):
    """task_ids far above MaxArraySize (e.g. 45539) must not appear in --array=."""
    tasks_path = tmp_path / "workflow" / "tasks_custom.csv"
    tasks_path.parent.mkdir()

    import csv as csv_mod
    # Simulate task_ids typical of large runs: just above common MaxArraySize
    large_ids = [1001, 4055, 22770, 45231, 45539]
    rows = [
        {"task_id": str(tid), "surface": "Cu111", "adsorbate": "methanol",
         "seed": "0", "calculator": "sevennet_omni"}
        for tid in large_ids
    ]
    with tasks_path.open("w", newline="") as f:
        writer = csv_mod.DictWriter(
            f, fieldnames=["task_id", "surface", "adsorbate", "seed", "calculator"])
        writer.writeheader()
        writer.writerows(rows)

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    out_path = tmp_path / "submit_missing.sh"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "find_missing_tasks.py"),
            "--tasks", str(tasks_path),
            "--runs-dir", str(runs_dir),
            "--out", str(out_path),
            "--chunk-size", "200",
            "--throttle", "20",
            "--max-in-flight", "0",
        ],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["bash", "-n", str(out_path)], check=True, cwd=tmp_path)

    script = out_path.read_text()
    lines = script.splitlines()

    sbatch_lines = [l for l in lines if l.startswith("sbatch --array=")]
    assert len(sbatch_lines) == 1  # all 5 fit in one chunk

    # Extract the array spec and verify it is a small 0-based range
    array_spec = sbatch_lines[0].split("--array=")[1].split("%")[0]
    start, end = array_spec.split("-")
    assert start == "0"
    assert int(end) == len(large_ids) - 1  # 0-4

    # No raw large task_id must appear inside --array=
    for tid in large_ids:
        assert str(tid) not in array_spec

    # Real task_ids must appear in the TASK_IDS variable
    assert "TASK_IDS_1=(1001 4055 22770 45231 45539)" in script
