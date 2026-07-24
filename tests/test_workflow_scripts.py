import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from ase import Atoms
from ase.io import write


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


def _write_fake_potcar_library(path: Path, elements: tuple[str, ...] = ("Cu", "H")):
    for element in elements:
        potcar = path / element / "POTCAR"
        potcar.parent.mkdir(parents=True, exist_ok=True)
        potcar.write_text(f"{element} POTCAR\n")


def _run_setup_vasp_jobs(
    tmp_path: Path,
    calc_type: str | None,
    *,
    functional: str = "pbe",
    vdw_kernel_path: Path | None = None,
    dry_run: bool = False,
    capture_output: bool = False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "setup_vasp_jobs.py").write_text(
        (REPO_ROOT / "setup_vasp_jobs.py").read_text()
    )
    poscar_root = tmp_path / "poscar" / "best"
    _write_minimal_poscar(poscar_root / "C1" / "Cu001_CO" / "POSCAR")
    pp_root = tmp_path / "fake_pp"
    _write_fake_potcar_library(pp_root)
    cmd = [
        sys.executable,
        "setup_vasp_jobs.py",
        "--poscar-dir",
        "poscar/best",
        "--functional",
        functional,
        "--pp-path",
        str(pp_root),
    ]
    if calc_type is not None:
        cmd.extend(["--calc-type", calc_type])
    if vdw_kernel_path is not None:
        cmd.extend(["--vdw-kernel-path", str(vdw_kernel_path)])
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        check=True,
        capture_output=capture_output,
        text=capture_output,
    )
    subfolder = {
        "pbe": "PBE",
        "pbe-d3": "PBE_D3",
        "r2scan": "r2scan",
        "beef-vdw": "beef_vdw",
    }[functional]
    if calc_type == "single-point":
        job_dir = poscar_root / "C1" / "Cu001_CO" / "singlepoint" / subfolder
    else:
        job_dir = poscar_root / "C1" / "Cu001_CO" / subfolder
    return result, job_dir


def _run_setup_slab_jobs(
    tmp_path: Path,
    *,
    functional: str = "pbe",
    force: bool = False,
    capture_output: bool = False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "setup_slab_jobs.py").write_text(
        (REPO_ROOT / "setup_slab_jobs.py").read_text()
    )
    pp_root = tmp_path / "fake_pp"
    _write_fake_potcar_library(pp_root, elements=("Cu",))
    cmd = [
        sys.executable,
        "setup_slab_jobs.py",
        "--functional",
        functional,
        "--surfaces",
        "Cu001",
        "--pp-path",
        str(pp_root),
    ]
    if force:
        cmd.append("--force")
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        check=True,
        capture_output=capture_output,
        text=capture_output,
    )
    subfolder = {
        "pbe": "PBE",
        "pbe-d3": "PBE_D3",
        "r2scan": "r2scan",
        "beef-vdw": "beef_vdw",
    }[functional]
    job_dir = tmp_path / "vasp_slab" / "Cu001" / subfolder
    return result, job_dir


def _run_setup_molecule_jobs(
    tmp_path: Path,
    *,
    functional: str = "pbe",
    force: bool = False,
    capture_output: bool = False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "setup_molecule_jobs.py").write_text(
        (REPO_ROOT / "setup_molecule_jobs.py").read_text()
    )
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (inputs_dir / "CO2.cif").write_bytes((REPO_ROOT / "inputs" / "CO2.cif").read_bytes())
    pp_root = tmp_path / "fake_pp"
    _write_fake_potcar_library(pp_root, elements=("C", "O"))
    cmd = [
        sys.executable,
        "setup_molecule_jobs.py",
        "--functional",
        functional,
        "--molecules",
        "CO2",
        "--pp-path",
        str(pp_root),
    ]
    if force:
        cmd.append("--force")
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        check=True,
        capture_output=capture_output,
        text=capture_output,
    )
    subfolder = {
        "pbe": "PBE",
        "pbe-d3": "PBE_D3",
        "r2scan": "r2scan",
        "beef-vdw": "beef_vdw",
    }[functional]
    job_dir = tmp_path / "vasp_mol" / "CO2" / subfolder
    return result, job_dir


def _write_tasks_custom_csv(path: Path, rows: list[dict[str, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_id", "surface", "adsorbate", "seed", "calculator"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_run_status(run_dir: Path, state: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps({"state": state}))


def _run_find_missing_tasks(
    tmp_path: Path,
    rows: list[dict[str, str]],
    runs_dir: Path,
):
    tasks_path = tmp_path / "workflow" / "tasks_custom.csv"
    _write_tasks_custom_csv(tasks_path, rows)
    out_path = tmp_path / "submit_missing.sh"
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "find_missing_tasks.py"),
            "--tasks",
            str(tasks_path),
            "--runs-dir",
            str(runs_dir),
            "--out",
            str(out_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_extract_poscar_run(
    run_dir: Path,
    surface: str,
    adsorbate: str,
    e_ads: float = -1.0,
    with_final_geometry: bool = True,
    status_state: str | None = "finished",
):
    run_dir.mkdir(parents=True, exist_ok=True)
    if with_final_geometry:
        geom_dir = run_dir / f"{adsorbate}_on_{surface}"
        geom_dir.mkdir(parents=True, exist_ok=True)
        atoms = Atoms("CuH", positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=[5, 5, 5], pbc=True)
        write(str(geom_dir / "final_adsorbed.cif"), atoms)
    if status_state is not None:
        (run_dir / "status.json").write_text(json.dumps({"state": status_state}))
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "E_ads_eV": e_ads,
                "surface_cif": f"inputs/{surface}.cif",
                "molecule_cif": f"inputs/{adsorbate}.cif",
            }
        )
    )


def _run_extract_poscar(runs_dir: Path, out_dir: Path, *extra_args: str):
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "extract_poscar.py"),
            "--runs-dir",
            str(runs_dir),
            "--out-dir",
            str(out_dir),
            *extra_args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _load_repo_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    _, default_job_dir = _run_setup_vasp_jobs(tmp_path / "default", calc_type=None)
    _, explicit_job_dir = _run_setup_vasp_jobs(tmp_path / "explicit_relax", calc_type="relax")
    incar_default = default_job_dir / "INCAR"
    incar_explicit = explicit_job_dir / "INCAR"

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
    _, job_dir = _run_setup_vasp_jobs(tmp_path, calc_type="single-point")
    incar_path = job_dir / "INCAR"
    incar_text = incar_path.read_text()

    assert "NSW    = 0" in incar_text
    assert "IBRION = -1" in incar_text
    assert "EDIFFG" not in incar_text
    assert "! Exchange-correlation" in incar_text
    assert "GGA = PE" in incar_text
    # single-point must be nested under singlepoint/<Functional>/
    assert incar_path.parent.name == "PBE"
    assert incar_path.parent.parent.name == "singlepoint"


def test_setup_vasp_jobs_beef_vdw_copies_vdw_kernel_in_relax_mode(tmp_path):
    source = tmp_path / "vdw_kernel.bindat"
    source.write_bytes(b"fake-vdw-kernel\n")

    result, job_dir = _run_setup_vasp_jobs(
        tmp_path / "relax_beef_vdw",
        calc_type="relax",
        functional="beef-vdw",
        vdw_kernel_path=source,
        capture_output=True,
    )

    copied = job_dir / "vdw_kernel.bindat"
    assert copied.exists()
    assert copied.read_bytes() == source.read_bytes()
    assert "written:  POSCAR, INCAR, KPOINTS, slm.vasp.kestrel, POTCAR, vdw_kernel.bindat" in result.stdout


def test_setup_vasp_jobs_beef_vdw_copies_vdw_kernel_in_single_point_mode(tmp_path):
    source = tmp_path / "singlepoint_vdw_kernel.bindat"
    source.write_bytes(b"single-point-vdw-kernel\n")

    _, job_dir = _run_setup_vasp_jobs(
        tmp_path / "singlepoint_beef_vdw",
        calc_type="single-point",
        functional="beef-vdw",
        vdw_kernel_path=source,
    )

    copied = job_dir / "vdw_kernel.bindat"
    assert job_dir.parent.name == "singlepoint"
    assert copied.exists()
    assert copied.read_bytes() == source.read_bytes()


def test_setup_vasp_jobs_non_beef_functional_does_not_copy_vdw_kernel(tmp_path):
    source = tmp_path / "unused_vdw_kernel.bindat"
    source.write_bytes(b"unused-vdw-kernel\n")

    _, job_dir = _run_setup_vasp_jobs(
        tmp_path / "pbe_run",
        calc_type="relax",
        functional="pbe",
        vdw_kernel_path=source,
    )

    assert not (job_dir / "vdw_kernel.bindat").exists()


def test_setup_vasp_jobs_missing_vdw_kernel_warns_without_crashing(tmp_path):
    missing = tmp_path / "does_not_exist" / "vdw_kernel.bindat"

    result, job_dir = _run_setup_vasp_jobs(
        tmp_path / "missing_vdw",
        calc_type="relax",
        functional="beef-vdw",
        vdw_kernel_path=missing,
        capture_output=True,
    )

    assert (job_dir / "INCAR").exists()
    assert (job_dir / "KPOINTS").exists()
    assert (job_dir / "POTCAR").exists()
    assert (job_dir / "slm.vasp.kestrel").exists()
    assert not (job_dir / "vdw_kernel.bindat").exists()
    assert "WARNING: vdw_kernel.bindat not found at:" in result.stdout
    assert "copy it manually" in result.stdout


def test_setup_vasp_jobs_dry_run_does_not_copy_vdw_kernel(tmp_path):
    source = tmp_path / "dry_run_vdw_kernel.bindat"
    source.write_bytes(b"dry-run-vdw-kernel\n")

    _, job_dir = _run_setup_vasp_jobs(
        tmp_path / "dry_run_beef_vdw",
        calc_type="relax",
        functional="beef-vdw",
        vdw_kernel_path=source,
        dry_run=True,
    )

    assert not job_dir.exists()


def test_setup_slab_jobs_beef_vdw_copies_vdw_kernel(tmp_path):
    module = _load_repo_module("setup_slab_jobs.py", "setup_slab_jobs_test_module")
    kernel_source = tmp_path / "vdw_kernel.bindat"
    kernel_source.write_bytes(b"slab-kernel\n")
    module.DEFAULT_VDW_KERNEL_PATH = str(kernel_source)

    pp_root = tmp_path / "fake_pp"
    _write_fake_potcar_library(pp_root, elements=("Cu",))
    result = module.setup_slab_dir(
        "Cu001",
        tmp_path / "vasp_slab",
        pp_root=pp_root,
        n_fixed=2,
        functional="beef-vdw",
    )

    copied = tmp_path / "vasp_slab" / "Cu001" / "beef_vdw" / "vdw_kernel.bindat"
    assert result["status"] == "ok"
    assert copied.exists()
    assert copied.read_bytes() == kernel_source.read_bytes()


def test_setup_slab_jobs_missing_vdw_kernel_warns(tmp_path):
    module = _load_repo_module("setup_slab_jobs.py", "setup_slab_jobs_missing_kernel")
    module.DEFAULT_VDW_KERNEL_PATH = str(tmp_path / "missing" / "vdw_kernel.bindat")

    pp_root = tmp_path / "fake_pp"
    _write_fake_potcar_library(pp_root, elements=("Cu",))
    result = module.setup_slab_dir(
        "Cu001",
        tmp_path / "vasp_slab",
        pp_root=pp_root,
        n_fixed=2,
        functional="beef-vdw",
    )

    assert result["status"] == "partial"
    assert any("vdw_kernel.bindat not found at:" in warning for warning in result["warnings"])


def test_setup_molecule_jobs_beef_vdw_copies_vdw_kernel(tmp_path):
    module = _load_repo_module("setup_molecule_jobs.py", "setup_molecule_jobs_test_module")
    kernel_source = tmp_path / "vdw_kernel.bindat"
    kernel_source.write_bytes(b"mol-kernel\n")
    module.DEFAULT_VDW_KERNEL_PATH = str(kernel_source)

    pp_root = tmp_path / "fake_pp"
    _write_fake_potcar_library(pp_root, elements=("C", "O"))
    result = module.setup_mol_dir(
        "CO2",
        str(REPO_ROOT / "inputs" / "CO2.cif"),
        tmp_path / "vasp_mol",
        pp_root=pp_root,
        functional="beef-vdw",
    )

    copied = tmp_path / "vasp_mol" / "CO2" / "beef_vdw" / "vdw_kernel.bindat"
    assert result["status"] == "ok"
    assert copied.exists()
    assert copied.read_bytes() == kernel_source.read_bytes()


def test_setup_molecule_jobs_missing_vdw_kernel_warns(tmp_path):
    module = _load_repo_module("setup_molecule_jobs.py", "setup_molecule_jobs_missing_kernel")
    module.DEFAULT_VDW_KERNEL_PATH = str(tmp_path / "missing" / "vdw_kernel.bindat")

    pp_root = tmp_path / "fake_pp"
    _write_fake_potcar_library(pp_root, elements=("C", "O"))
    result = module.setup_mol_dir(
        "CO2",
        str(REPO_ROOT / "inputs" / "CO2.cif"),
        tmp_path / "vasp_mol",
        pp_root=pp_root,
        functional="beef-vdw",
    )

    assert result["status"] == "partial"
    assert any("vdw_kernel.bindat not found at:" in warning for warning in result["warnings"])


def test_setup_slab_jobs_skip_finished_job_unless_force(tmp_path):
    _, job_dir = _run_setup_slab_jobs(tmp_path / "slab_skip_force", functional="pbe")
    outcar = job_dir / "OUTCAR"
    outcar.write_text("finished\n")
    poscar = job_dir / "POSCAR"
    poscar.write_text("sentinel-poscar\n")

    skip_result, _ = _run_setup_slab_jobs(
        tmp_path / "slab_skip_force",
        functional="pbe",
        capture_output=True,
    )
    assert "SKIPPED: OUTCAR already exists" in skip_result.stdout
    assert poscar.read_text() == "sentinel-poscar\n"

    _run_setup_slab_jobs(tmp_path / "slab_skip_force", functional="pbe", force=True)
    assert "sentinel-poscar" not in poscar.read_text()


def test_setup_molecule_jobs_skip_finished_job_unless_force(tmp_path):
    _, job_dir = _run_setup_molecule_jobs(tmp_path / "mol_skip_force", functional="pbe")
    outcar = job_dir / "OUTCAR"
    outcar.write_text("finished\n")
    poscar = job_dir / "POSCAR"
    poscar.write_text("sentinel-poscar\n")

    skip_result, _ = _run_setup_molecule_jobs(
        tmp_path / "mol_skip_force",
        functional="pbe",
        capture_output=True,
    )
    assert "SKIPPED: OUTCAR already exists" in skip_result.stdout
    assert poscar.read_text() == "sentinel-poscar\n"

    _run_setup_molecule_jobs(tmp_path / "mol_skip_force", functional="pbe", force=True)
    assert "sentinel-poscar" not in poscar.read_text()


def test_setup_vasp_jobs_processes_multiple_carbon_buckets(tmp_path):
    (tmp_path / "setup_vasp_jobs.py").write_text(
        (REPO_ROOT / "setup_vasp_jobs.py").read_text()
    )
    poscar_root = tmp_path / "poscar" / "best"
    _write_minimal_poscar(poscar_root / "C1" / "Cu001_CO" / "POSCAR")
    _write_minimal_poscar(poscar_root / "C3" / "Pt111_isopropanol" / "POSCAR")

    subprocess.run(
        [
            sys.executable,
            "setup_vasp_jobs.py",
            "--poscar-dir",
            "poscar/best",
            "--functional",
            "pbe",
        ],
        cwd=tmp_path,
        check=True,
    )

    assert (poscar_root / "C1" / "Cu001_CO" / "PBE" / "INCAR").exists()
    assert (poscar_root / "C3" / "Pt111_isopropanol" / "PBE" / "INCAR").exists()


def test_setup_vasp_jobs_accepts_single_bucket_poscar_dir(tmp_path):
    (tmp_path / "setup_vasp_jobs.py").write_text(
        (REPO_ROOT / "setup_vasp_jobs.py").read_text()
    )
    bucket_dir = tmp_path / "poscar" / "best" / "C1"
    _write_minimal_poscar(bucket_dir / "Cu001_CO" / "POSCAR")

    subprocess.run(
        [
            sys.executable,
            "setup_vasp_jobs.py",
            "--poscar-dir",
            "poscar/best/C1",
            "--functional",
            "pbe",
            "--calc-type",
            "single-point",
        ],
        cwd=tmp_path,
        check=True,
    )

    assert (bucket_dir / "Cu001_CO" / "singlepoint" / "PBE" / "INCAR").exists()


def test_find_missing_tasks_classifies_c1_finished_run_in_bucket(tmp_path):
    row = {
        "task_id": "0",
        "surface": "Ir100",
        "adsorbate": "formic_acid",
        "seed": "0",
        "calculator": "sevennet_omni",
    }
    run_name = "Ir100_formic_acid_seed0_sevennet_omni"
    runs_dir = tmp_path / "runs"
    _write_run_status(runs_dir / run_name, "failed")
    _write_run_status(runs_dir / "C1" / run_name, "finished")

    result = _run_find_missing_tasks(tmp_path, [row], runs_dir)

    assert "Finished              :      1" in result.stdout
    assert "Failed / not finished :      0" in result.stdout
    assert "Missing (no dir)      :      0" in result.stdout


def test_find_missing_tasks_looks_up_nonzero_carbon_bucket(tmp_path):
    row = {
        "task_id": "0",
        "surface": "Cu111",
        "adsorbate": "isopropanol",
        "seed": "2",
        "calculator": "sevennet_omni",
    }
    run_name = "Cu111_isopropanol_seed2_sevennet_omni"
    runs_dir = tmp_path / "runs"
    _write_run_status(runs_dir / "C3" / run_name, "finished")

    result = _run_find_missing_tasks(tmp_path, [row], runs_dir)

    assert "Finished              :      1" in result.stdout
    assert "Missing (no dir)      :      0" in result.stdout


def test_find_missing_tasks_keeps_missing_runs_missing_without_side_effects(tmp_path):
    row = {
        "task_id": "0",
        "surface": "Pt111",
        "adsorbate": "formic_acid",
        "seed": "0",
        "calculator": "5m",
    }
    runs_dir = tmp_path / "runs"

    result = _run_find_missing_tasks(tmp_path, [row], runs_dir)

    assert f"WARNING: runs directory not found: {runs_dir}" in result.stdout
    assert "Finished              :      0" in result.stdout
    assert "Missing (no dir)      :      1" in result.stdout
    assert not runs_dir.exists()


def test_carbon_count_imports_shared_and_batch_modules(tmp_path):
    sandbox_run_dir = tmp_path / "sandbox-run"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(REPO_ROOT)!r}); "
                "from molecule_utils import carbon_count as shared_count; "
                "from batch_isopropanol import carbon_count as batch_count; "
                "print(shared_count('formic_acid')); "
                "print(batch_count('formic_acid'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GOAD_RUN_DIR": str(sandbox_run_dir)},
    )

    assert result.stdout.strip().splitlines() == ["1", "1"]


def test_find_missing_tasks_import_and_helper_do_not_create_runs_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module_path = REPO_ROOT / "find_missing_tasks.py"
    spec = importlib.util.spec_from_file_location(
        "find_missing_tasks_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    run_dir = module.get_task_run_dir(
        tmp_path / "runs",
        {
            "task_id": "0",
            "surface": "Ir100",
            "adsorbate": "formic_acid",
            "seed": "0",
            "calculator": "sevennet_omni",
        },
    )

    assert run_dir == (
        tmp_path / "runs" / "C1" / "Ir100_formic_acid_seed0_sevennet_omni"
    )
    assert not (tmp_path / "runs").exists()


def test_extract_poscar_collects_bucketed_layout_and_writes_outputs(tmp_path):
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "poscar"
    run_dir = runs_dir / "C1" / "Ir100_formic_acid_seed1_sevennet_omni"
    _write_extract_poscar_run(run_dir, "Ir100", "formic_acid", e_ads=-1.23)

    result = _run_extract_poscar(runs_dir, out_dir)

    assert "Found 1 completed run(s)" in result.stdout
    assert "No completed runs" not in result.stdout
    assert (out_dir / "C1" / "Ir100_formic_acid_sevennet_omni_seed1" / "POSCAR").exists()
    assert (out_dir / "best" / "C1" / "Ir100_formic_acid" / "POSCAR").exists()


def test_extract_poscar_includes_finished_run_by_default(tmp_path):
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "poscar"
    _write_extract_poscar_run(
        runs_dir / "C1" / "Ir100_formic_acid_seed1_sevennet_omni",
        "Ir100",
        "formic_acid",
        status_state="finished",
    )

    result = _run_extract_poscar(runs_dir, out_dir)

    assert "Found 1 completed run(s)" in result.stdout
    assert "Skipped (not finished per status.json)" not in result.stdout
    assert (out_dir / "C1" / "Ir100_formic_acid_sevennet_omni_seed1" / "POSCAR").exists()


def test_extract_poscar_excludes_non_finished_run_by_default(tmp_path):
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "poscar"
    _write_extract_poscar_run(
        runs_dir / "C1" / "Ir100_formic_acid_seed1_sevennet_omni",
        "Ir100",
        "formic_acid",
        status_state="failed",
    )

    result = _run_extract_poscar(runs_dir, out_dir)

    assert "No completed runs" in result.stdout
    assert "Skipped (not finished per status.json): 1" in result.stdout
    assert not (out_dir / "C1" / "Ir100_formic_acid_sevennet_omni_seed1" / "POSCAR").exists()
    assert not (out_dir / "best" / "C1" / "Ir100_formic_acid" / "POSCAR").exists()


def test_extract_poscar_treats_missing_status_as_not_finished(tmp_path):
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "poscar"
    _write_extract_poscar_run(
        runs_dir / "C1" / "Ir100_formic_acid_seed1_sevennet_omni",
        "Ir100",
        "formic_acid",
        status_state=None,
    )

    result = _run_extract_poscar(runs_dir, out_dir)

    assert "No completed runs" in result.stdout
    assert "Skipped (not finished per status.json): 1" in result.stdout
    assert not (out_dir / "C1" / "Ir100_formic_acid_sevennet_omni_seed1" / "POSCAR").exists()


def test_extract_poscar_include_unfinished_restores_old_behavior(tmp_path):
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "poscar"
    _write_extract_poscar_run(
        runs_dir / "C1" / "Ir100_formic_acid_seed1_sevennet_omni",
        "Ir100",
        "formic_acid",
        status_state="running",
    )

    result = _run_extract_poscar(runs_dir, out_dir, "--include-unfinished")

    assert "Found 1 completed run(s)" in result.stdout
    assert "Skipped (not finished per status.json)" not in result.stdout
    assert (out_dir / "C1" / "Ir100_formic_acid_sevennet_omni_seed1" / "POSCAR").exists()


def test_extract_poscar_collects_runs_from_multiple_buckets(tmp_path):
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "poscar"
    _write_extract_poscar_run(
        runs_dir / "C1" / "Ir100_formic_acid_seed1_sevennet_omni",
        "Ir100",
        "formic_acid",
        e_ads=-1.23,
    )
    _write_extract_poscar_run(
        runs_dir / "C3" / "Cu111_propanol_seed0_5m",
        "Cu111",
        "propanol",
        e_ads=-0.80,
    )

    result = _run_extract_poscar(runs_dir, out_dir)

    assert "Found 2 completed run(s)" in result.stdout
    assert (out_dir / "C1" / "Ir100_formic_acid_sevennet_omni_seed1" / "POSCAR").exists()
    assert (out_dir / "C3" / "Cu111_propanol_5m_seed0" / "POSCAR").exists()
    assert (out_dir / "best" / "C1" / "Ir100_formic_acid" / "POSCAR").exists()
    assert (out_dir / "best" / "C3" / "Cu111_propanol" / "POSCAR").exists()


def test_extract_poscar_mixed_finished_and_unfinished_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    out_dir_default = tmp_path / "poscar_default"
    out_dir_include = tmp_path / "poscar_include"
    _write_extract_poscar_run(
        runs_dir / "C1" / "Ir100_formic_acid_seed1_sevennet_omni",
        "Ir100",
        "formic_acid",
        e_ads=-1.23,
        status_state="finished",
    )
    _write_extract_poscar_run(
        runs_dir / "C3" / "Cu111_propanol_seed0_5m",
        "Cu111",
        "propanol",
        e_ads=-0.80,
        status_state="running",
    )

    default_result = _run_extract_poscar(runs_dir, out_dir_default)
    include_result = _run_extract_poscar(
        runs_dir, out_dir_include, "--include-unfinished"
    )

    assert "Found 1 completed run(s)" in default_result.stdout
    assert "Skipped (not finished per status.json): 1" in default_result.stdout
    assert (out_dir_default / "C1" / "Ir100_formic_acid_sevennet_omni_seed1" / "POSCAR").exists()
    assert not (out_dir_default / "C3" / "Cu111_propanol_5m_seed0" / "POSCAR").exists()

    assert "Found 2 completed run(s)" in include_result.stdout
    assert "Skipped (not finished per status.json)" not in include_result.stdout
    assert (out_dir_include / "C1" / "Ir100_formic_acid_sevennet_omni_seed1" / "POSCAR").exists()
    assert (out_dir_include / "C3" / "Cu111_propanol_5m_seed0" / "POSCAR").exists()


def test_extract_poscar_collects_when_runs_dir_points_to_single_bucket_or_flat_layout(tmp_path):
    runs_dir = tmp_path / "runs_C1"
    out_dir = tmp_path / "poscar"
    _write_extract_poscar_run(
        runs_dir / "Ir100_formic_acid_seed1_sevennet_omni",
        "Ir100",
        "formic_acid",
        e_ads=-1.23,
    )

    result = _run_extract_poscar(runs_dir, out_dir)

    assert "Found 1 completed run(s)" in result.stdout
    assert (out_dir / "C1" / "Ir100_formic_acid_sevennet_omni_seed1" / "POSCAR").exists()
    assert (out_dir / "best" / "C1" / "Ir100_formic_acid" / "POSCAR").exists()


def test_extract_poscar_best_only_layout_is_bucketed(tmp_path):
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "poscar"
    _write_extract_poscar_run(
        runs_dir / "C1" / "Ir100_formic_acid_seed1_sevennet_omni",
        "Ir100",
        "formic_acid",
        e_ads=-1.23,
    )

    _run_extract_poscar(runs_dir, out_dir, "--best-only")

    assert (out_dir / "C1" / "Ir100_formic_acid" / "POSCAR").exists()
    assert not (out_dir / "best" / "C1" / "Ir100_formic_acid" / "POSCAR").exists()


def test_extract_poscar_empty_bucketed_layout_exits_cleanly_with_no_completed_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    out_dir = tmp_path / "poscar"
    _write_extract_poscar_run(
        runs_dir / "C1" / "Ir100_formic_acid_seed1_sevennet_omni",
        "Ir100",
        "formic_acid",
        with_final_geometry=False,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "extract_poscar.py"),
            "--runs-dir",
            str(runs_dir),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "No completed runs" in result.stdout
