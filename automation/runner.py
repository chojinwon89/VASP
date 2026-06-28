#!/usr/bin/env python
import argparse
import os
import re
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from error_handlers import detect_and_patch
from analysis import run_analysis

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_dir TEXT UNIQUE,
  job_type TEXT,
  surface TEXT,
  molecule TEXT,
  functional TEXT,
  slurm_id TEXT,
  status TEXT,
  retries INTEGER DEFAULT 0,
  energy REAL,
  updated_at TEXT
);
"""


def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def connect_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA_SQL)
    conn.commit()
    return conn


def classify_job(job_dir: Path):
    p = job_dir.as_posix().split("/")
    # expected examples:
    # campaigns/current/jobs/vasp_slab/Cu111/PBE
    # campaigns/current/jobs/vasp_mol/CO2/PBE
    # campaigns/current/jobs/poscar/best/Cu111_CO2/PBE
    job_type = "unknown"
    surface = None
    molecule = None
    functional = p[-1] if p else None

    if "vasp_slab" in p:
        job_type = "slab"
        surface = p[-2]
    elif "vasp_mol" in p:
        job_type = "molecule"
        molecule = p[-2]
    else:
        job_type = "adsorption"
        if len(p) >= 2:
            system = p[-2]
            if "_" in system:
                surface, molecule = system.split("_", 1)
            else:
                surface = system

    return job_type, surface, molecule, functional


def discover_job_dirs(jobs_root: Path):
    dirs = []
    for incar in jobs_root.rglob("INCAR"):
        d = incar.parent
        required = ["POSCAR", "INCAR", "KPOINTS", "slm.vasp.kestrel"]
        if all((d / x).exists() for x in required):
            dirs.append(d)
    return sorted(set(dirs))


def upsert_job(conn, job_dir: Path):
    job_type, surface, molecule, functional = classify_job(job_dir)
    conn.execute(
        """
        INSERT INTO jobs(job_dir, job_type, surface, molecule, functional, status, updated_at)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(job_dir) DO UPDATE SET
          job_type=excluded.job_type,
          surface=excluded.surface,
          molecule=excluded.molecule,
          functional=excluded.functional,
          updated_at=excluded.updated_at
        """,
        (str(job_dir), job_type, surface, molecule, functional, "new", now_iso()),
    )


def submit_job(job_dir: Path, submit_cmd: str):
    proc = subprocess.run(
        submit_cmd,
        cwd=job_dir,
        shell=True,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = re.search(r"Submitted batch job\s+(\d+)", out)
    slurm_id = m.group(1) if m else None
    ok = proc.returncode == 0 and slurm_id is not None
    return ok, slurm_id, out.strip()


def read_energy_from_oszicar(job_dir: Path):
    osz = job_dir / "OSZICAR"
    if not osz.exists():
        return None
    lines = osz.read_text(errors="ignore").splitlines()
    for line in reversed(lines):
        if "F=" in line:
            # e.g. " ... F= -.123456 E0= ..."
            m = re.search(r"F=\s*([\-0-9\.Ee+]+)", line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    pass
    return None


def likely_converged(job_dir: Path):
    outcar = job_dir / "OUTCAR"
    if not outcar.exists():
        return False
    txt = outcar.read_text(errors="ignore")[-200000:]
    return "reached required accuracy" in txt.lower()


def update_status_logic(conn, row, cfg):
    job_id, job_dir, status, retries = row
    job_dir = Path(job_dir)

    if status in ("done", "running", "queued"):
        # quick terminal check for done
        if status != "done" and likely_converged(job_dir):
            e = read_energy_from_oszicar(job_dir)
            conn.execute(
                "UPDATE jobs SET status=?, energy=?, updated_at=? WHERE id=?",
                ("done", e, now_iso(), job_id),
            )
        return

    if status in ("new", "failed", "patched"):
        if likely_converged(job_dir):
            e = read_energy_from_oszicar(job_dir)
            conn.execute(
                "UPDATE jobs SET status=?, energy=?, updated_at=? WHERE id=?",
                ("done", e, now_iso(), job_id),
            )
            return

        # try patch first if failed
        if status == "failed":
            patched = detect_and_patch(job_dir, cfg)
            if patched:
                conn.execute(
                    "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                    ("patched", now_iso(), job_id),
                )
                return

        if retries >= int(cfg["scheduler"].get("max_retries", 4)):
            return

        ok, slurm_id, msg = submit_job(job_dir, cfg["scheduler"]["submit_cmd"])
        if ok:
            conn.execute(
                "UPDATE jobs SET status=?, slurm_id=?, retries=?, updated_at=? WHERE id=?",
                ("queued", slurm_id, retries + 1, now_iso(), job_id),
            )
        else:
            conn.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                ("failed", now_iso(), job_id),
            )
            print(f"[submit-fail] {job_dir}: {msg}")


def refresh_queue_states(conn):
    # simple passive transitions (without calling sacct/squeue parser)
    # if OUTCAR appears and has convergence marker -> done
    cur = conn.execute("SELECT id, job_dir, status FROM jobs WHERE status IN ('queued','running')")
    for job_id, job_dir, status in cur.fetchall():
        d = Path(job_dir)
        if likely_converged(d):
            e = read_energy_from_oszicar(d)
            conn.execute(
                "UPDATE jobs SET status=?, energy=?, updated_at=? WHERE id=?",
                ("done", e, now_iso(), job_id),
            )
        else:
            outcar = d / "OUTCAR"
            if outcar.exists():
                conn.execute(
                    "UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                    ("running", now_iso(), job_id),
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    jobs_root = Path(cfg["paths"]["jobs_root"])
    db_path = Path(cfg["paths"]["state_db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = connect_db(db_path)

    def one_cycle():
        for d in discover_job_dirs(jobs_root):
            upsert_job(conn, d)
        conn.commit()

        refresh_queue_states(conn)
        conn.commit()

        cur = conn.execute("SELECT id, job_dir, status, retries FROM jobs")
        for row in cur.fetchall():
            update_status_logic(conn, row, cfg)
        conn.commit()

        run_analysis(conn, cfg)
        conn.commit()

        n_done = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='done'").fetchone()[0]
        n_all = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        print(f"[{now_iso()}] done={n_done}/{n_all}")

    if args.once:
        one_cycle()
        return

    poll_minutes = int(cfg["scheduler"].get("poll_minutes", 15))
    while True:
        one_cycle()
        time.sleep(poll_minutes * 60)


if __name__ == "__main__":
    main()
