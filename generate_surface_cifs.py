"""
generate_surface_cifs.py
========================
Generate CIF files for all metal surface slabs needed for the CCB project
adsorption workflow.

Supported crystal structures and metals
---------------------------------------
FCC : Cu, Pd, Pt, Ni, Au, Ag, Ir, Rh
BCC : Fe, Cr, Mo
HCP : Ru, Co, Ti, Zn

Facets built per structure
--------------------------
FCC  ->  (111), (110), (100)
BCC  ->  (110), (100), (111)
HCP  ->  (0001), (10-10)  [hcp0001 and hcp10m10 in ASE]

Slabs use a 3x3 supercell (3x2 for open (110) facets) with 4 layers
and ~15 A vacuum.

Existing CIF files are **never overwritten** -- the script skips them.

Output: one .cif file per surface in inputs/

Usage
-----
    python generate_surface_cifs.py
"""

import os
from ase.build import (
    fcc111, fcc110, fcc100,
    bcc110, bcc100, bcc111,
    hcp0001, hcp10m10,
)
from ase.io import write

os.makedirs("inputs", exist_ok=True)

# ---------------------------------------------------------------------------
# Experimental lattice constants (Angstrom)
# FCC: a only
# BCC: a only
# HCP: (a, c)
# ---------------------------------------------------------------------------
A_FCC = {
    "Cu": 3.615,
    "Pd": 3.890,
    "Pt": 3.924,
    "Ni": 3.524,
    "Au": 4.078,
    "Ag": 4.086,
    "Ir": 3.840,
    "Rh": 3.803,
}

A_BCC = {
    "Fe": 2.870,
    "Cr": 2.885,
    "Mo": 3.147,
}

A_HCP = {
    # (a, c)
    "Ru": (2.706, 4.282),
    "Co": (2.507, 4.069),
    "Ti": (2.951, 4.686),
    "Zn": (2.665, 4.947),
}

# ---------------------------------------------------------------------------
# Build spec lists
# (element, builder_func, size, extra_kwargs, output_name)
# ---------------------------------------------------------------------------
SLAB_SPECS = []

# --- FCC ---
for el, a in A_FCC.items():
    SLAB_SPECS += [
        (el, fcc111, (3, 3, 4), {"a": a, "orthogonal": True}, f"{el}111"),
        (el, fcc110, (3, 2, 4), {"a": a},                     f"{el}110"),
        (el, fcc100, (3, 3, 4), {"a": a},                     f"{el}100"),
    ]

# Cu001 alias (same slab as Cu100, kept for backwards compat with task CSVs)
SLAB_SPECS.append(
    ("Cu", fcc100, (3, 3, 4), {"a": A_FCC["Cu"]}, "Cu001")
)

# --- BCC ---
for el, a in A_BCC.items():
    SLAB_SPECS += [
        (el, bcc110, (3, 2, 4), {"a": a},                     f"{el}110"),
        (el, bcc100, (3, 3, 4), {"a": a},                     f"{el}100"),
        (el, bcc111, (3, 3, 4), {"a": a, "orthogonal": True}, f"{el}111"),
    ]

# --- HCP ---
for el, (a, c) in A_HCP.items():
    SLAB_SPECS += [
        (el, hcp0001,  (3, 3, 4), {"a": a, "c": c, "orthogonal": True}, f"{el}0001"),
        (el, hcp10m10, (3, 2, 4), {"a": a, "c": c},                     f"{el}10m10"),
    ]

# ---------------------------------------------------------------------------
# Generate CIFs
# ---------------------------------------------------------------------------
written = []
skipped = []
errors  = []

for element, builder, size, kwargs, name in SLAB_SPECS:
    out_path = f"inputs/{name}.cif"
    if os.path.exists(out_path):
        skipped.append(name)
        continue
    try:
        slab = builder(element, size=size, vacuum=15.0, **kwargs)
        write(out_path, slab)
        cell = slab.get_cell().lengths()
        written.append(
            f"{name}: {len(slab):3d} atoms  "
            f"cell = [{cell[0]:.2f} {cell[1]:.2f} {cell[2]:.2f}] A"
        )
    except Exception as exc:
        errors.append(f"{name}: {exc}")

print(f"\n{'='*65}")
print("Surface CIF generation complete")
print(f"  Written : {len(written)}")
print(f"  Skipped : {len(skipped)}  (already existed)")
print(f"  Errors  : {len(errors)}")
print(f"{'='*65}")

if written:
    print("\nWritten:")
    for entry in written:
        print(f"  + {entry}")

if skipped:
    print("\nSkipped (not overwritten):")
    for s in skipped:
        print(f"  ~ {s}")

if errors:
    print("\nErrors:")
    for e in errors:
        print(f"  ! {e}")

print()
print("All surface CIFs are in inputs/")
