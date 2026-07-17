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


def test_discover_classifies_molecules_not_surfaces(tmp_path):
    """
    Verify that short inorganic/diatomic molecule names (H2, O2, N2, etc.)
    that match the surface regex '^[A-Z][a-z]?\\d+$' are correctly classified
    as molecules — not surfaces — by discover_surfaces_and_molecules().

    This is the regression test for the H2 misclassification bug where the
    regex matched 'H2' (one uppercase letter + one digit) just like 'Fe2'.
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from workflow.make_tasks_custom import discover_surfaces_and_molecules

    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()

    # Create fake CIF files: real surfaces and known molecule names
    surface_stems = ["Cu111", "Pt111", "Ir111", "Fe110", "Ru0001"]
    molecule_stems = [
        "H2", "O2", "N2", "CO", "NH3", "NO", "NO2", "SO2", "H2S",
        "isopropanol", "glycerol", "ethanol",
    ]

    for stem in surface_stems + molecule_stems:
        (inputs_dir / f"{stem}.cif").write_text("# fake CIF\n")

    surfaces, molecules = discover_surfaces_and_molecules(inputs_dir)

    # All known molecule names must end up as molecules
    for mol in molecule_stems:
        assert mol not in surfaces, (
            f"'{mol}' was misclassified as a surface — fix the KNOWN_MOLECULE_NAMES check"
        )
        assert mol in molecules, f"'{mol}' not found in molecules dict"

    # Genuine surface names must end up as surfaces
    for surf in surface_stems:
        assert surf in surfaces, f"'{surf}' not found in surfaces list"
        assert surf not in molecules, f"'{surf}' was misclassified as a molecule"
