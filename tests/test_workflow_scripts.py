import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

NEW_BIO_OIL_MOLECULES = {
    "methyl_formate": "COC=O",
    "angelica_lactone": "CC1=CCC(=O)O1",
    "gamma_butyrolactone": "O=C1CCCO1",
    "ethylene_glycol": "OCCO",
    "glyoxal": "O=CC=O",
    "2-ethylphenol": "CCc1ccccc1O",
    "hydroquinone": "Oc1ccc(O)cc1",
    "guaiacol": "COc1ccccc1O",
    "4-methylguaiacol": "Cc1ccc(O)c(OC)c1",
    "eugenol": "C=CCc1ccc(O)c(OC)c1",
    "isoeugenol": "C/C=C/c1ccc(O)c(OC)c1",
    "syringol": "COc1cccc(OC)c1O",
    "propyl_syringol": "CCCc1cc(OC)c(O)c(OC)c1",
    "syringaldehyde": "COc1cc(C=O)cc(OC)c1O",
    "levoglucosan": "OC1C(O)C(O)C2COC1O2",
    "alpha-D-glucopyranose": "OCC1OC(O)C(O)C(O)C1O",
    "D-fructofuranose": "OCC1(O)OCC(O)C1O",
    "D-xylopyranose": "OC1COC(O)C(O)C1O",
    "1,6-anhydroglucofuranose": "OC1C2COC1OC2O",
    "2-furanone": "O=C1C=CCO1",
    "hydroxyacetaldehyde": "OCC=O",
    "acetal": "CC(OCC)OCC",
    "methylcyclopentenolone": "CC1=C(O)CCC1=O",
    "vanillin": "COc1cc(C=O)ccc1O",
}


def _write_minimal_poscar(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "test\n"
        "1.0\n"
        "1 0 0\n"
        "0 1 0\n"
        "0 0 1\n"
        "Cu H\n"
        "1 1\n"
        "Cartesian\n"
        "0 0 0\n"
        "0.5 0.5 0.5\n"
    )


def _run_setup_vasp_jobs(tmp_path: Path, calc_type: str | None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "setup_vasp_jobs.py").write_text(
        (REPO_ROOT / "setup_vasp_jobs.py").read_text()
    )
    poscar_root = tmp_path / "poscar" / "best"
    _write_minimal_poscar(poscar_root / "Cu001_CO" / "POSCAR")
    cmd = [
        sys.executable,
        "setup_vasp_jobs.py",
        "--poscar-dir",
        "poscar/best",
        "--functional",
        "pbe",
    ]
    if calc_type is not None:
        cmd.extend(["--calc-type", calc_type])
    subprocess.run(cmd, cwd=tmp_path, check=True)
    if calc_type == "single-point":
        return poscar_root / "Cu001_CO" / "singlepoint" / "PBE" / "INCAR"
    return poscar_root / "Cu001_CO" / "PBE" / "INCAR"


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


def test_new_bio_oil_molecules_added_consistently():
    import sys
    sys.path.insert(0, str(REPO_ROOT))

    from batch_isopropanol import MOLECULE_SMILES
    from setup_molecule_jobs import MOLECULE_REGISTRY

    for name, smiles in NEW_BIO_OIL_MOLECULES.items():
        assert MOLECULE_SMILES.get(name) == smiles
        assert MOLECULE_REGISTRY.get(name) == f"inputs/{name}.cif"


def test_new_bio_oil_carbon_counts():
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from batch_isopropanol import carbon_count

    assert carbon_count("eugenol") == 10
    assert carbon_count("glyoxal") == 2
    assert carbon_count("gamma_butyrolactone") == 4


def test_new_bio_oil_names_are_never_surface_classified(tmp_path):
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from workflow.make_tasks_custom import (
        _SURFACE_RE,
        KNOWN_MOLECULE_NAMES,
        discover_surfaces_and_molecules,
    )

    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()
    surface_stems = ["Cu111", "Pt111", "Ir111"]
    for stem in surface_stems + list(NEW_BIO_OIL_MOLECULES):
        (inputs_dir / f"{stem}.cif").write_text("# fake CIF\n")

    surfaces, molecules = discover_surfaces_and_molecules(inputs_dir)

    for name in NEW_BIO_OIL_MOLECULES:
        assert name in KNOWN_MOLECULE_NAMES
        assert name in molecules
        assert name not in surfaces
        assert _SURFACE_RE.match(name) is None


def test_setup_vasp_jobs_relax_default_and_explicit(tmp_path):
    incar_default = _run_setup_vasp_jobs(tmp_path / "default", calc_type=None)
    incar_explicit = _run_setup_vasp_jobs(tmp_path / "explicit_relax", calc_type="relax")

    default_text = incar_default.read_text()
    explicit_text = incar_explicit.read_text()

    assert "NSW    = 1000" in default_text
    assert "IBRION = 2" in default_text
    assert "EDIFFG = -5E-02" in default_text
    assert "! Exchange-correlation" in default_text
    assert "GGA = PE" in default_text
    assert default_text == explicit_text
    # relax mode must NOT use the singlepoint/ subdirectory
    assert "singlepoint" not in str(incar_default)
    assert "singlepoint" not in str(incar_explicit)


def test_setup_vasp_jobs_single_point_incar_and_subfolder(tmp_path):
    incar_path = _run_setup_vasp_jobs(tmp_path, calc_type="single-point")
    incar_text = incar_path.read_text()

    assert "NSW    = 0" in incar_text
    assert "IBRION = -1" in incar_text
    assert "EDIFFG" not in incar_text
    assert "! Exchange-correlation" in incar_text
    assert "GGA = PE" in incar_text
    # single-point must be nested under singlepoint/<Functional>/
    assert incar_path.parent.name == "PBE"
    assert incar_path.parent.parent.name == "singlepoint"
