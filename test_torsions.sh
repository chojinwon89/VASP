#!/bin/bash
# Clear stale bytecode and test torsion detection

find goad_v1 -name __pycache__ -exec rm -rf {} +

python - <<'PY'
from ase.io import read
from goad_v1.utils.torsion_handler import TorsionHandler
m = read("inputs/isopropanol.cif")
th = TorsionHandler(m)
print("n_torsions =", th.n_torsions)
print("bonds      =", th.rotatable_bonds)
PY