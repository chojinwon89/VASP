#!/usr/bin/env python
from pathlib import Path
import csv


def run_analysis(conn, cfg):
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = conn.execute(
        """
        SELECT job_dir, job_type, surface, molecule, functional, status, energy
        FROM jobs
        """
    ).fetchall()

    summary_csv = reports_dir / "jobs_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_dir", "job_type", "surface", "molecule", "functional", "status", "energy"])
        w.writerows(rows)

    # Build adsorption energies when all components exist
    # Eads = E(slab+mol) - E(slab) - E(mol)
    slabs = {}
    mols = {}
    ads = []

    for job_dir, job_type, surface, molecule, functional, status, energy in rows:
        if status != "done" or energy is None:
            continue
        key = (functional, surface, molecule)
        if job_type == "slab":
            slabs[(functional, surface)] = energy
        elif job_type == "molecule":
            mols[(functional, molecule)] = energy
        elif job_type == "adsorption":
            ads.append((functional, surface, molecule, energy, job_dir))

    eads_rows = []
    for functional, surface, molecule, e_tot, job_dir in ads:
        if surface is None or molecule is None:
            continue
        e_slab = slabs.get((functional, surface))
        e_mol = mols.get((functional, molecule))
        if e_slab is None or e_mol is None:
            continue
        e_ads = e_tot - e_slab - e_mol
        eads_rows.append([functional, surface, molecule, e_tot, e_slab, e_mol, e_ads, job_dir])

    eads_csv = reports_dir / "adsorption_energies.csv"
    with eads_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["functional", "surface", "molecule", "E_slab+mol", "E_slab", "E_mol", "E_ads", "job_dir"])
        w.writerows(sorted(eads_rows))

    # Optional plotting if matplotlib installed
    try:
        import matplotlib.pyplot as plt
        if eads_rows:
            labels = [f"{r[0]}|{r[1]}|{r[2]}" for r in eads_rows]
            vals = [r[6] for r in eads_rows]
            plt.figure(figsize=(max(8, len(vals) * 0.45), 4.5))
            plt.bar(range(len(vals)), vals)
            plt.xticks(range(len(vals)), labels, rotation=90)
            plt.ylabel("E_ads (eV)")
            plt.title("Adsorption energies")
            plt.tight_layout()
            plt.savefig(reports_dir / "adsorption_energies.png", dpi=200)
            plt.close()
    except Exception:
        pass
