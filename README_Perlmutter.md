# README_Perlmutter — GOAD + SevenNet → DFT validation pipeline

End-to-end directions for the MLIP-vs-DFT benchmark, from producing the missing
SevenNet (GOAD) structures through running full VASP geometry optimizations and
collecting results. Written for **Perlmutter**; Kestrel differences are noted
inline.

The benchmark grid is **477 well-known physical systems** = 25 molecules × 7
non-magnetic metals {Ag, Au, Cu, Ir, Pd, Pt, Rh} × 3 facets {100, 110, 111}
(the 6 radicals C2H2/CH3O/H/HCN/O/OH only on their 13 in-scope surfaces). The
`deoxy` group is out of scope.

Two independent stages, each with its own job array:

```
Stage A: MLIP gap-fill (GOAD/SevenNet, GPU)   -> relaxed CIF per surface+molecule
Stage B: DFT geometry optimization (VASP, CPU) -> PBE / PBE-D3 / r2scan / beef-vdw
```

---

## 0. One-time setup (per login session)

```bash
cd /pscratch/sd/j/jcho5/VASP
git pull

# DFT (Stage B) environment:
module load vasp-tpc/6.4.2-cpu
export VASP_PP_PATH=/pscratch/sd/j/jcho5/paw64/potpaw_PBE_64
export VASP_VDW_KERNEL_PATH=/pscratch/sd/j/jcho5/vdw_kernel.bindat   # beef-vdw only
```

Local (Mac) Python has `ase`; the workflow scripts are pure-Python + `ase`.
`structure/` (the MLIP gallery, ~333 MB) and the GOAD `runs/` tree live on the
cluster / Mac, **not** in git — only the 2 MB `dft_jobs/` POSCAR tree and the
MANIFEST are committed.

| Cluster | module | POTCAR (`VASP_PP_PATH`) | vdw kernel |
|---|---|---|---|
| Perlmutter CPU | `vasp-tpc/6.4.2-cpu` | `/pscratch/sd/j/jcho5/paw64/potpaw_PBE_64` | `/pscratch/sd/j/jcho5/vdw_kernel.bindat` |
| Kestrel | `vasp/6.3.2_openMP+tpc` | `/projects/2dmgcat/paw64/potpaw_PBE_64` (default) | `/projects/2dmgcat/vdw_kernel.bindat` (default) |

On Kestrel the `setup_vasp_jobs.py` defaults already point at the right paths,
so no env vars are needed.

---

## 1. Check coverage — what is missing?

```bash
python workflow/check_dft_coverage.py                    # full report + dft_coverage.csv
python workflow/check_dft_coverage.py --system CH3_Pt111 # diagnose one system
python workflow/check_dft_coverage.py --missing-only     # just the gaps
```

Each gap is classified:
- `missing_no_cif` — no SevenNet structure yet → **Stage A** produces it.
- `missing_has_cif` — a gallery CIF exists but wasn't staged → a staging bug
  (re-run `stage_dft_poscars.py`).

---

## 2. Stage A — MLIP gap-fill (GOAD / SevenNet, GPU)

The gaps come in two buckets, each with its own task list.

```bash
# 2a. generate task lists
python workflow/make_tasks_missing_sevennet.py     # 588 radical tasks
python workflow/make_tasks_existing_gap.py         # ~44 tasks / ~22 literature systems (mostly Pt111)

# 2b. submit both GOAD arrays (GPU)
sbatch --array=0-587%50 -t 00:20:00 perlmutter/goad_array_perlmutter_gpu.slurm workflow/tasks_missing_sevennet.csv
sbatch --array=0-43%50  -t 00:20:00 perlmutter/goad_array_perlmutter_gpu.slurm workflow/tasks_existing_gap.csv
```

`-t 00:20:00` keeps the NERSC cost estimate small (these ≤C2 tasks finish in
≤~11 min on an A100). Prerequisite input CIFs come from
`python generate_surface_cifs.py` + `python generate_molecule_cifs.py`.

### 2c. Collect → import → stage

```bash
# collect best seed per system -> collected/sevennet_missing/
python workflow/collect_missing_sevennet.py --tasks-csv workflow/tasks_missing_sevennet.csv
python workflow/collect_missing_sevennet.py --tasks-csv workflow/tasks_existing_gap.csv

# import collected CIFs into structure/ + fill MANIFEST gallery_cif (dry-run first)
python workflow/import_sevennet_to_gallery.py --dry-run
python workflow/import_sevennet_to_gallery.py            # backs up MANIFEST.csv.bak

# stage POSCARs + verify coverage
python workflow/stage_dft_poscars.py --manifest DFT_results/MANIFEST.csv \
    --structure-dir structure --out-dir dft_jobs --fix-bottom-layers 2
python workflow/check_dft_coverage.py                    # target: 477/477
```

`collect_missing_sevennet.py` prints any system with no finished seed as
**MISSING** plus the exact `task_id`s to resubmit — re-run just those:
```bash
sbatch --array=<those_ids> -t 00:20:00 perlmutter/goad_array_perlmutter_gpu.slurm workflow/tasks_existing_gap.csv
```

`import_sevennet_to_gallery.py` resolves name aliases (`methanol↔CH3OH`,
`acetylene↔C2H2`, `DME↔CH3OCH3`) and fills one representative row per physical
system (adsorption-site variants like `CO_Pt111_atop/fcc` share one structure).

> **Note (Perlmutter):** re-staging prints `gallery_cif NAMED but FILE MISSING`
> for the original ~377 systems because their gallery CIFs live only on the Mac.
> This is harmless — those POSCARs are already committed in `dft_jobs/`, so
> staging leaves them intact and only writes the newly-imported systems.

---

## 3. Stage B — DFT geometry optimization (VASP, CPU)

### 3a. Generate VASP inputs per functional

```bash
for f in pbe pbe-d3 r2scan beef-vdw; do
    python setup_vasp_jobs.py --poscar-dir dft_jobs --functional $f \
        --cluster perlmutter-cpu --skip-existing
done
```

This writes `dft_jobs/<system>/<FUNC>/{INCAR,KPOINTS,POTCAR,slm.vasp.perlmutter}`
(FUNC = `PBE`, `PBE_D3`, `r2scan`, `beef_vdw`). INCAR defaults: full relax
`NSW=1000, IBRION=2, EDIFFG=-5E-02`, `ENCUT=450`, `ISPIN=2`, `EDIFF=1E-05`,
KPOINTS Monkhorst-Pack 2×2×1. beef-vdw copies `vdw_kernel.bindat` into each job.

**`--skip-existing` is the key to incremental waves:** it creates inputs only for
systems that don't already have an `INCAR`, so **already-converged / running jobs
are never touched**. Add new gap-fill systems, re-run the loop, and only the new
folders get set up. On Kestrel drop `--cluster` (default) — it writes
`slm.vasp.kestrel`.

### 3b. Submit the array (runs only non-converged jobs)

```bash
python workflow/make_dft_joblist.py --functional pbe > joblist_pbe.txt
N=$(grep -vc '^#' joblist_pbe.txt)
sbatch --array=0-$((N-1))%20 perlmutter/vasp_dft_array_cpu.slurm joblist_pbe.txt
```

`make_dft_joblist.py` **skips OUTCARs that already reached required accuracy**, so
the array only ever runs jobs that still need it. Repeat for `pbe-d3`, `r2scan`,
`beef-vdw` once PBE looks good (r2scan / beef-vdw are 2–3× slower). On Kestrel use
`vasp_dft_array_kestrel.slurm`.

### 3c. Only run the newly-added / failed jobs

You do **not** have to re-run converged systems. Two independent safeguards make
the pipeline incremental:

1. **Setup:** `setup_vasp_jobs.py --skip-existing` → inputs only for new systems.
2. **Submit:** `make_dft_joblist.py` and `resubmit_dft.py` → only non-converged
   OUTCARs enter the array.

So after a new MLIP wave: `stage_dft_poscars.py` → `setup_vasp_jobs.py
--skip-existing` (new inputs only) → `make_dft_joblist.py` (new/unfinished only)
→ `sbatch`. The already-converged 377 stay put.

---

## 4. Find & resubmit missing / failed DFT jobs

Once `squeue` is empty:

```bash
python workflow/resubmit_dft.py --jobs-dir dft_jobs --functional pbe            # preview
python workflow/resubmit_dft.py --jobs-dir dft_jobs --functional pbe --submit   # resubmit
python workflow/resubmit_dft.py --jobs-dir dft_jobs --functional pbe \
    --restart-from-contcar --submit    # continue crashed/unconverged from last geometry
python workflow/resubmit_dft.py --jobs-dir dft_jobs --functional all --submit   # all four
```

It scans the whole OUTCAR (not just the tail) and classifies every job as
`done` / `running` (both skipped) / `unconverged` / `crashed` / `not_started`,
resubmitting only the last three. On Kestrel add `--cluster kestrel`.

---

## 5. Analysis (after DFT finishes)

Parse the converged OUTCARs, fill the MANIFEST `dft_*` columns
(`dft_Eads_eV`, `dft_dMX_ang`, `dft_MX_pair`, `dft_dint_ang`), and compare against
the MLIP results — the adsorption energy and molecule–surface bond distance are
what this benchmark measures. `plot_dft_vs_mlip.py` renders the comparison.
*(The OUTCAR→MANIFEST parser is the last tool to add.)*

---

## Script reference

| Script | Purpose |
|---|---|
| `workflow/check_dft_coverage.py` | Reconcile in-scope MANIFEST vs staged `dft_jobs/`; classify gaps; `--system` lookup |
| `workflow/make_tasks_missing_sevennet.py` | Emit the 588 radical GOAD tasks |
| `workflow/make_tasks_existing_gap.py` | Emit GOAD tasks for the ~22 non-radical literature gaps |
| `workflow/collect_missing_sevennet.py` | Pick best seed per system → `collected/sevennet_missing/`; report re-run ids |
| `workflow/import_sevennet_to_gallery.py` | Copy collected CIFs into `structure/` + fill MANIFEST `gallery_cif` |
| `workflow/stage_dft_poscars.py` | Gallery CIF → species-sorted VASP5 POSCAR tree (`--fix-bottom-layers`) |
| `setup_vasp_jobs.py` | Write INCAR/KPOINTS/POTCAR/slurm per functional (`--skip-existing`, `--cluster`) |
| `workflow/make_dft_joblist.py` | List non-converged job dirs for one functional (array input) |
| `perlmutter/vasp_dft_array_cpu.slurm` (Perlmutter) / `vasp_dft_array_kestrel.slurm` (repo root, Kestrel) | DFT job-array submit scripts |
| `workflow/resubmit_dft.py` | Scan + resubmit failed/missing DFT jobs (`--restart-from-contcar`) |

## Quick recipe (steady state)

```bash
git pull
python workflow/check_dft_coverage.py                        # gaps?
# ... Stage A if any missing_no_cif (section 2) ...
for f in pbe pbe-d3 r2scan beef-vdw; do
  python setup_vasp_jobs.py --poscar-dir dft_jobs --functional $f --cluster perlmutter-cpu --skip-existing
done
python workflow/make_dft_joblist.py --functional pbe > joblist_pbe.txt
N=$(grep -vc '^#' joblist_pbe.txt)
sbatch --array=0-$((N-1))%20 perlmutter/vasp_dft_array_cpu.slurm joblist_pbe.txt
# after it drains:
python workflow/resubmit_dft.py --jobs-dir dft_jobs --functional pbe --submit
```
