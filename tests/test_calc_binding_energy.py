import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_outcar(path: Path, energy: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "header\n"
        f"  free  energy   TOTEN  =   {energy: .6f} eV\n"
    )


def _write_poscar(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("POSCAR\n")


def _run_calc(tmp_path: Path, *args: str) -> list[dict[str, str]]:
    output = tmp_path / "out.csv"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "calc_binding_energy.py"),
        *args,
        "--output",
        str(output),
    ]
    subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, text=True)
    with output.open() as handle:
        return list(csv.DictReader(handle))


def _run_calc_with_stdout(tmp_path: Path, *args: str) -> tuple[list[dict[str, str]], str]:
    output = tmp_path / "out.csv"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "calc_binding_energy.py"),
        *args,
        "--output",
        str(output),
    ]
    result = subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, text=True)
    with output.open() as handle:
        return list(csv.DictReader(handle)), result.stdout


def test_calc_binding_energy_discovers_bucketed_relax_layout(tmp_path):
    best_dir = tmp_path / "poscar" / "best"
    system_dir = best_dir / "C1" / "Cu001_CO"
    _write_poscar(system_dir / "POSCAR")
    _write_outcar(system_dir / "PBE" / "OUTCAR", -15.0)

    _write_outcar(tmp_path / "vasp_slab" / "Cu001" / "PBE" / "OUTCAR", -10.0)
    _write_outcar(tmp_path / "vasp_mol" / "CO" / "PBE" / "OUTCAR", -2.0)

    rows = _run_calc(
        tmp_path,
        "--best-dirs",
        str(best_dir),
        "--slab-dir",
        str(tmp_path / "vasp_slab"),
        "--mol-dir",
        str(tmp_path / "vasp_mol"),
        "--functional",
        "PBE",
    )

    assert len(rows) == 1
    assert rows[0]["system"] == "Cu001_CO"
    assert rows[0]["status"] == "ok"
    assert float(rows[0]["E_ads"]) == -3.0


def test_calc_binding_energy_discovers_bucketed_single_point_layout(tmp_path):
    best_dir = tmp_path / "poscar" / "best"
    system_dir = best_dir / "C1" / "Cu001_CO"
    _write_poscar(system_dir / "POSCAR")
    _write_outcar(system_dir / "singlepoint" / "PBE" / "OUTCAR", -16.0)

    _write_outcar(tmp_path / "vasp_slab" / "Cu001" / "PBE" / "OUTCAR", -10.0)
    _write_outcar(tmp_path / "vasp_mol" / "CO" / "PBE" / "OUTCAR", -2.0)

    rows = _run_calc(
        tmp_path,
        "--best-dirs",
        str(best_dir),
        "--slab-dir",
        str(tmp_path / "vasp_slab"),
        "--mol-dir",
        str(tmp_path / "vasp_mol"),
        "--functional",
        "PBE",
        "--calc-type",
        "single-point",
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert float(rows[0]["E_ads"]) == -4.0


def test_calc_binding_energy_discovers_multiple_buckets(tmp_path):
    best_dir = tmp_path / "poscar" / "best"
    cu_dir = best_dir / "C1" / "Cu001_CO"
    pt_dir = best_dir / "C2" / "Pt111_CO2"
    _write_poscar(cu_dir / "POSCAR")
    _write_poscar(pt_dir / "POSCAR")
    _write_outcar(cu_dir / "PBE" / "OUTCAR", -15.0)
    _write_outcar(pt_dir / "PBE" / "OUTCAR", -30.0)

    _write_outcar(tmp_path / "vasp_slab" / "Cu001" / "PBE" / "OUTCAR", -10.0)
    _write_outcar(tmp_path / "vasp_slab" / "Pt111" / "PBE" / "OUTCAR", -20.0)
    _write_outcar(tmp_path / "vasp_mol" / "CO" / "PBE" / "OUTCAR", -2.0)
    _write_outcar(tmp_path / "vasp_mol" / "CO2" / "PBE" / "OUTCAR", -5.0)

    rows = _run_calc(
        tmp_path,
        "--best-dirs",
        str(best_dir),
        "--slab-dir",
        str(tmp_path / "vasp_slab"),
        "--mol-dir",
        str(tmp_path / "vasp_mol"),
        "--functional",
        "PBE",
    )

    assert sorted(r["system"] for r in rows) == ["Cu001_CO", "Pt111_CO2"]
    assert all(r["status"] == "ok" for r in rows)


def test_calc_binding_energy_bucketed_root_ignores_stale_flat_leftover(tmp_path):
    best_dir = tmp_path / "poscar" / "best"
    bucketed_system_dir = best_dir / "C2" / "Au100_DMSO"
    stale_flat_dir = best_dir / "Au100_DMSO"

    _write_poscar(bucketed_system_dir / "POSCAR")
    _write_outcar(bucketed_system_dir / "singlepoint" / "PBE" / "OUTCAR", -25.0)

    _write_poscar(stale_flat_dir / "POSCAR")

    _write_outcar(tmp_path / "vasp_slab" / "Au100" / "PBE" / "OUTCAR", -20.0)
    _write_outcar(tmp_path / "vasp_mol" / "DMSO" / "PBE" / "OUTCAR", -3.0)

    rows, stdout = _run_calc_with_stdout(
        tmp_path,
        "--best-dirs",
        str(best_dir),
        "--slab-dir",
        str(tmp_path / "vasp_slab"),
        "--mol-dir",
        str(tmp_path / "vasp_mol"),
        "--functional",
        "PBE",
        "--calc-type",
        "single-point",
    )

    assert len(rows) == 1
    assert rows[0]["system"] == "Au100_DMSO"
    assert rows[0]["status"] == "ok"
    assert "WARNING: found stale non-bucketed system directory" in stdout
    assert "Au100_DMSO" in stdout
    assert "IGNORED" in stdout


def test_calc_binding_energy_flat_layout_fallback_without_bucket_warning(tmp_path):
    best_dir = tmp_path / "poscar" / "best"
    system_dir = best_dir / "Au100_DMSO"
    _write_poscar(system_dir / "POSCAR")
    _write_outcar(system_dir / "PBE" / "OUTCAR", -25.0)

    _write_outcar(tmp_path / "vasp_slab" / "Au100" / "PBE" / "OUTCAR", -20.0)
    _write_outcar(tmp_path / "vasp_mol" / "DMSO" / "PBE" / "OUTCAR", -3.0)

    rows, stdout = _run_calc_with_stdout(
        tmp_path,
        "--best-dirs",
        str(best_dir),
        "--slab-dir",
        str(tmp_path / "vasp_slab"),
        "--mol-dir",
        str(tmp_path / "vasp_mol"),
        "--functional",
        "PBE",
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert "WARNING: found stale non-bucketed system directory" not in stdout


def test_calc_binding_energy_accepts_single_bucket_or_direct_system_best_dir(tmp_path):
    best_root = tmp_path / "poscar" / "best"
    system_dir = best_root / "C1" / "Cu001_CO"
    _write_poscar(system_dir / "POSCAR")
    _write_outcar(system_dir / "PBE" / "OUTCAR", -15.0)

    _write_outcar(tmp_path / "vasp_slab" / "Cu001" / "PBE" / "OUTCAR", -10.0)
    _write_outcar(tmp_path / "vasp_mol" / "CO" / "PBE" / "OUTCAR", -2.0)

    common_args = (
        "--slab-dir",
        str(tmp_path / "vasp_slab"),
        "--mol-dir",
        str(tmp_path / "vasp_mol"),
        "--functional",
        "PBE",
    )

    bucket_rows = _run_calc(
        tmp_path,
        "--best-dirs",
        str(best_root / "C1"),
        *common_args,
    )
    direct_rows = _run_calc(
        tmp_path,
        "--best-dirs",
        str(system_dir),
        *common_args,
    )

    assert len(bucket_rows) == 1
    assert bucket_rows[0]["status"] == "ok"
    assert len(direct_rows) == 1
    assert direct_rows[0]["status"] == "ok"
