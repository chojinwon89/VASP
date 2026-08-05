import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _mk_run(runs: Path, rel: str, system: str, calc: str):
    d = runs / rel
    d.mkdir(parents=True)
    surface = system.split("_on_")[1]
    adsorbate = system.split("_on_")[0]
    (d / "result.json").write_text(json.dumps({
        "system": system,
        "E_ads_eV": -1.0,
        "calculator": calc,
        "ga": {"best_seed": 1},
    }))
    (d / "status.json").write_text(json.dumps({
        "surface": surface,
        "adsorbate": adsorbate,
        "calculator": calc,
        "state": "finished",
        "run_dir": str(d),
    }))


def test_collect_flat_and_bucketed(tmp_path):
    runs = tmp_path / "runs"
    _mk_run(runs, "flatrun", "H2_on_Ag100", "sevennet_omni")
    _mk_run(runs, "C0/bucketrun", "H2O_on_Ag100", "sevennet_omni")
    _mk_run(runs, "C3/propene_run", "propene_on_Pt111", "5m")

    out = tmp_path / "summary.csv"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "collect_results.py"),
         "-r", str(runs), "-o", str(out)],
        check=True, cwd=REPO_ROOT,
    )

    rows = list(csv.DictReader(out.open()))
    keys = {(r["surface"], r["adsorbate"]) for r in rows}
    assert keys == {
        ("Ag100", "H2"),
        ("Ag100", "H2O"),
        ("Pt111", "propene"),
    }
    assert len(rows) == 3
