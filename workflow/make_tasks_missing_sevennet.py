#!/usr/bin/env python3
"""
make_tasks_missing_sevennet.py
==============================
Generate a SMALL, dedicated task list for the adsorbate/surface systems that
are MISSING from the SevenNet gallery (i.e. no *_sevennet_omni.cif exists yet).

Unlike make_tasks_custom.py -- which auto-discovers EVERY molecule in inputs/
and would emit >50k tasks -- this script emits only the gap-fill set:

    6 adsorbates  x  7 non-magnetic metals  x  3 facets  x  2 seeds  x  1 calc
      = 252 SevenNet-OMNI GOAD tasks.

Adsorbates (all C0-C2 -> 2 seeds each, per the standard tiered scheme):
    acetylene (C2)  methoxy (C1)  HCN (C1)  hydroxyl (C0)  atomicH (C0)  atomicO (C0)

These six have ZERO structure in the SevenNet gallery under any naming token,
so they must be generated from scratch (see molecule_utils.py /
generate_molecule_cifs.py, which now know how to build them).

Output CSV schema matches make_tasks_custom.py exactly:
    task_id,surface,adsorbate,seed,calculator,population_size,generations,n_carbon

Usage
-----
    python workflow/make_tasks_missing_sevennet.py

Then, on Perlmutter (after `python generate_surface_cifs.py` and
`python generate_molecule_cifs.py`):

    sbatch --array=0-251%50 perlmutter/goad_array_perlmutter_gpu.slurm \
        workflow/tasks_missing_sevennet.csv
"""
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from molecule_utils import carbon_count  # noqa: E402

# Non-magnetic metals only (Fe, Co, Ni, Cr excluded: spin-free foundation MLIPs
# are not on fair footing on magnetic surfaces).
METALS = ["Ag", "Au", "Cu", "Ir", "Pd", "Pt", "Rh"]
FACETS = ["100", "110", "111"]

# The six adsorbates with no SevenNet gallery structure.
ADSORBATES = ["acetylene", "methoxy", "HCN", "hydroxyl", "atomicH", "atomicO"]

CALCULATOR = "sevennet_omni"   # 7net-mf-ompa (omat24, PBE+D3) -- the gallery calc
POP = 60
GEN = 200


def seeds_for(n_c):
    """Tiered seed scheme, identical to make_tasks_custom.py."""
    if n_c <= 2:
        return [1, 2]
    elif n_c <= 4:
        return [1, 2, 3]
    return [1, 2, 3, 4, 5]


def main():
    out = REPO / "workflow" / "tasks_missing_sevennet.csv"
    rows = []
    task_id = 0
    for ads in ADSORBATES:
        n_c = carbon_count(ads)
        for metal in METALS:
            for facet in FACETS:
                surface = f"{metal}{facet}"
                for seed in seeds_for(n_c):
                    rows.append({
                        "task_id":         task_id,
                        "surface":         surface,
                        "adsorbate":       ads,
                        "seed":            seed,
                        "calculator":      CALCULATOR,
                        "population_size": POP,
                        "generations":     GEN,
                        "n_carbon":        n_c,
                    })
                    task_id += 1

    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {task_id} tasks to {out}")
    print(f"Submit with:  sbatch --array=0-{task_id - 1}%50 "
          f"perlmutter/goad_array_perlmutter_gpu.slurm "
          f"workflow/tasks_missing_sevennet.csv")
    print()
    print("Adsorbate breakdown:")
    counts = Counter(r["adsorbate"] for r in rows)
    for ads in ADSORBATES:
        print(f"  {ads:<12} C{carbon_count(ads)}  {counts[ads]:>3} tasks")


if __name__ == "__main__":
    main()
