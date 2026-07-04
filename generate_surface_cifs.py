"""
generate_surface_cifs.py
========================
Generate CIF files for all metal surface slabs needed for the CCB project
adsorption workflow.

Slabs use a 3×3 supercell (3×2 for (110) facets) with 4 layers and ~15 Å
vacuum, matching the existing Cu111.cif / Cu100.cif convention.

Existing CIF files are **never overwritten** — the script skips them.

Output: one .cif file per surface in inputs/

Usage
-----
    python generate_surface_cifs.py
"""

import os
from ase.build import fcc111, fcc110, fcc100
from ase.io import write

os.makedirs("inputs", exist_ok=True)

# ---------------------------------------------------------------------------
# Surface specifications
# (element, builder_func, supercell_size, extra_kwargs, output_name)
# ---------------------------------------------------------------------------
# Lattice constants (Å)
A = {
    "Cu": 3.61,
    "Pd": 3.89,
    "Pt": 3.92,
    "Ni": 3.52,
    "Au": 4.08,
    "Ag": 4.09,
}

# (element, facet_func, size, extra_kwargs, name)
SLAB_SPECS = [
    # Cu (existing Cu111 & Cu100 will be skipped automatically)
    ("Cu", fcc111, (3, 3, 4), {}, "Cu111"),
    ("Cu", fcc110, (3, 2, 4), {}, "Cu110"),
    ("Cu", fcc100, (3, 3, 4), {}, "Cu100"),
    # Pd
    ("Pd", fcc111, (3, 3, 4), {}, "Pd111"),
    ("Pd", fcc110, (3, 2, 4), {}, "Pd110"),
    ("Pd", fcc100, (3, 3, 4), {}, "Pd100"),
    # Pt
    ("Pt", fcc111, (3, 3, 4), {}, "Pt111"),
    ("Pt", fcc110, (3, 2, 4), {}, "Pt110"),
    ("Pt", fcc100, (3, 3, 4), {}, "Pt100"),
    # Ni
    ("Ni", fcc111, (3, 3, 4), {}, "Ni111"),
    ("Ni", fcc100, (3, 3, 4), {}, "Ni100"),
    # Au
    ("Au", fcc111, (3, 3, 4), {}, "Au111"),
    # Ag
    ("Ag", fcc111, (3, 3, 4), {}, "Ag111"),
    ("Ag", fcc110, (3, 2, 4), {}, "Ag110"),
    ("Ag", fcc100, (3, 3, 4), {}, "Ag100"),
]

written = []
skipped = []

for element, builder, size, extra, name in SLAB_SPECS:
    out_path = f"inputs/{name}.cif"
    if os.path.exists(out_path):
        skipped.append(name)
        continue
    try:
        a = A[element]
        slab = builder(element, size=size, a=a, vacuum=15.0, **extra)
        write(out_path, slab)
        cell = slab.get_cell().lengths()
        written.append(
            f"{name}: {len(slab):3d} atoms  "
            f"cell = [{cell[0]:.2f} {cell[1]:.2f} {cell[2]:.2f}] Å"
        )
    except Exception as exc:
        print(f"  WARNING: could not build {name}: {exc}")

print(f"\n{'='*60}")
print(f"Surface CIF generation complete")
print(f"  Written : {len(written)}")
print(f"  Skipped : {len(skipped)}  (already existed)")
print(f"{'='*60}")
for entry in written:
    print(f"  + {entry}")
if skipped:
    print(f"\nSkipped (not overwritten):")
    for s in skipped:
        print(f"  ~ {s}")
