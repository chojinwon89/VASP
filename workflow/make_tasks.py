#!/usr/bin/env python

import csv
from pathlib import Path

# Edit these lists for your production database
surfaces = [
    "Cu111",
    "Cu110",
    "Cu001",
    "Pt111",
    "Pt110",
    "Pt100",
    "Pd111",
    "Pd110",
    "Pd100",
    "Ni111",
    "Ni110",
    "Ni100",
    "Ag111",
    "Ag110",
    "Ag100",
    "Au111",
    "Au110",
    "Au100",
]

adsorbates = [
    "isopropanol",
    "CO2",
    "ethanol",
    "ethene",
    "ethane",
    "propane",
    "propene",
    "propanol",
    "glycerol",
]

seeds = [0, 1, 2]

calculator = "sevennet_omni"

out = Path("workflow/tasks.csv")
out.parent.mkdir(exist_ok=True)

with out.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "task_id",
            "surface",
            "adsorbate",
            "seed",
            "calculator",
        ],
    )
    writer.writeheader()

    task_id = 0
    for surface in surfaces:
        for adsorbate in adsorbates:
            for seed in seeds:
                writer.writerow(
                    {
                        "task_id": task_id,
                        "surface": surface,
                        "adsorbate": adsorbate,
                        "seed": seed,
                        "calculator": calculator,
                    }
                )
                task_id += 1

print(f"Wrote {task_id} tasks to {out}")