#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd
from ase.io import read, write

SUMMARY_CSV = Path("workflow/summary.csv")
OUTPUT_DIR = Path("best_structures")
OUTPUT_DIR.mkdir(exist_ok=True)

def is_sevennet_seed_0_5(job_dir: str) -> bool:
    s = str(job_dir).lower()
    if "sevennet" not in s:
        return False
    m = re.search(r"seed[_\- ]?([0-9]+)", s)
    if not m:
        return False
    seed = int(m.group(1))
    return 0 <= seed <= 5

def find_energy_column(df: pd.DataFrame) -> str:
    for c in ["E_ads_eV", "energy", "E", "score"]:
        if c in df.columns:
            return c
    raise ValueError(f"No energy column found. Columns={list(df.columns)}")

def find_poscar_column(df: pd.DataFrame) -> str:
    for c in ["best_poscar_path", "poscar_path", "POSCAR_path", "path", "job_dir"]:
        if c in df.columns:
            return c
    raise ValueError(f"No structure/path column found. Columns={list(df.columns)}")

def build_system_key(row: pd.Series) -> str:
    if "surface" in row and "adsorbate" in row:
        return f"{row['surface']}_{row['adsorbate']}"
    src = str(row.get("job_dir", row.get("path", "")))
    parts = Path(src).parts
    return parts[-2] if len(parts) >= 2 else "unknown_system"

if not SUMMARY_CSV.exists():
    raise FileNotFoundError(f"Cannot find {SUMMARY_CSV.resolve()}")

df = pd.read_csv(SUMMARY_CSV)
if df.empty:
    raise RuntimeError("summary.csv is empty")

energy_col = find_energy_column(df)
path_col = find_poscar_column(df)
job_col = "job_dir" if "job_dir" in df.columns else path_col

mask = df[job_col].astype(str).apply(is_sevennet_seed_0_5)
df = df[mask].copy()
if df.empty:
    raise RuntimeError("No rows matched SevenNet seed0~5 pattern in summary.csv")

df[energy_col] = pd.to_numeric(df[energy_col], errors="coerce")
df = df.dropna(subset=[energy_col])

df["system_key"] = df.apply(build_system_key, axis=1)
idx = df.groupby("system_key")[energy_col].idxmin()
best = df.loc[idx].sort_values("system_key")

print(f"Selected {len(best)} best structures (SevenNet seed0~5 minima).")

n_ok, n_fail = 0, 0
for _, row in best.iterrows():
    system = str(row["system_key"]).replace("/", "_")
    p = Path(str(row[path_col]))

    if p.is_dir():
        p = p / "POSCAR"

    if not p.exists():
        print(f"[MISS] {system}: structure file not found -> {p}")
        n_fail += 1
        continue

    try:
        atoms = read(p)
        cif_path = OUTPUT_DIR / f"{system}.cif"
        png_path = OUTPUT_DIR / f"{system}.png"

        write(cif_path, atoms)
        write(png_path, atoms, rotation='-70x,20y,10z', show_unit_cell=2)

        print(f"[OK] {system} | E={row[energy_col]:.6f} eV")
        n_ok += 1
    except Exception as e:
        print(f"[ERR] {system}: {e}")
        n_fail += 1

print("\nDone.")
print(f"  Success: {n_ok}")
print(f"  Failed : {n_fail}")
print(f"  Output : {OUTPUT_DIR.resolve()}")

best_out = OUTPUT_DIR / "best_sevennet_seed0_5.csv"
best.to_csv(best_out, index=False)
print(f"  Summary: {best_out}")
