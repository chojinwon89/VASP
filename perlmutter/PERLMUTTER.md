# Running GOAD on Perlmutter (NERSC)

Port of the GOAD MLIP global-optimization array workflow from **Kestrel (NREL)** to
**Perlmutter (NERSC)**. Each array task runs one `(surface, adsorbate, seed, calculator)`
genetic-algorithm relaxation with a torch MLIP (SevenNet-OMNI / MatterSim) and writes
`runs/C{n}/<name>/` with `final_adsorbed.cif`, `result.json`, `status.json`.

The Python workflow itself is unchanged — only the scheduler directives and the
software environment differ. Files in this folder:

| File | Purpose |
|------|---------|
| `goad_array_perlmutter_gpu.slurm` | **Primary.** 1 A100 per array task, `shared` QOS |
| `goad_array_perlmutter_cpu.slurm` | Fallback if you have no GPU allocation |
| `setup_perlmutter_env.sh` | One-time conda-env build |
| `PERLMUTTER.md` | This guide |

---

## Kestrel → Perlmutter: what changed

| | Kestrel | Perlmutter |
|---|---|---|
| Select node type | `--partition=standard` / `gpu-h100` | `-C cpu` / `-C gpu` (constraint) |
| Queue | partition implies it | `-q shared` / `regular` / `preempt` (**must set — default is `debug`, 30 min!**) |
| GPU request | `--gres=gpu:1` | `--gpus=1` |
| 1-GPU job | dedicated node | `-q shared` → ¼ node, ¼ charge |
| Account | `--account=ccpc` | `-A m5281` (CPU & GPU hours are **separate pools**) |
| GPU | H100 | A100 (sm_80, CUDA 12.1 wheels) |
| Cores/CPU node | 104 | 128 physical / 256 logical |
| Scratch | `/scratch/jcho5` | `$PSCRATCH` |
| Env module | `module load conda` | `module load conda` (same) |

---

## One-time setup

```bash
# 1. Get the code onto Perlmutter (run on a login node)
cd $PSCRATCH
git clone https://github.com/chojinwon89/VASP.git
cd VASP

# 2. Build the conda env (see options/paths inside the script)
export GOAD_ENV=/global/common/software/m5281/goad-env   # persistent; or $PSCRATCH/goad-env (purged ~8wk idle)
bash perlmutter/setup_perlmutter_env.sh

# 3. The two .slurm files are already set to  -A m5281  (your project).
#    To use a different project, edit the "#SBATCH -A" line in each.
```

> **Run `runs/` on `$PSCRATCH`, not `$HOME`.** The output tree has many files and
> `$HOME` is quota-limited (40 GB). Clone + run under `$PSCRATCH/VASP`.

---

## The task list

Production tasks already exist in `workflow/tasks_custom.csv`
(**52,440** tasks: `task_id, surface, adsorbate, seed, calculator, population_size, generations, n_carbon`).
Regenerate/extend with `workflow/make_tasks_custom.py` if needed.

### Gap-fill: the missing SevenNet structures (incl. open-shell radicals)

**14 adsorbates** have **no** SevenNet-OMNI gallery structure yet. A dedicated,
self-contained task list covers them on all 7 non-magnetic metals. Species that
*already* have gallery results (`CH2`, `CH3`, `HCO`, `O2`) are deliberately
**excluded** so we don't recompute them.

- Closed-shell / atomic gap-fill (6): `acetylene`, `HCN`, `methoxy`, `hydroxyl`,
  `atomicH`, `atomicO`.
- Open-shell radicals (8), well-known C/H/O validation set:
  `CH`, `atomicC` (CHx ladder); `C2H5`, `C2H3`, `C2H` (C2Hx ladder);
  `CH2OH`, `OOH`, `COOH` (oxygen + catalysis intermediates
  — `OOH`=hydroperoxyl is the ORR/OER `*OOH`; `COOH`=carboxyl is the CO₂RR `*COOH`).

- `workflow/tasks_missing_sevennet.csv` — **588 tasks**
  (14 adsorbates × {Ag,Au,Cu,Ir,Pd,Pt,Rh} × {100,110,111} × 2 seeds × `sevennet_omni`).
- Regenerate with `python workflow/make_tasks_missing_sevennet.py`.

Their gas-phase CIFs are committed under `inputs/`, and `molecule_utils.py` /
`generate_molecule_cifs.py` know how to (re)build them (ASE G2 geometries,
explicit coordinates for `C2H`/`CH2OH`/`OOH`/`COOH`, single-atom cells for the
atoms) — so this batch needs **no RDKit**.

> ⚠️ SevenNet-OMNI and MatterSim are spin-agnostic. For a fair DFT comparison,
> set the correct gas-phase multiplicities in your DFT references (triplet
> O₂/CH₂/C, doublet OH/CH₃/HCO/…). This spin mismatch is exactly what the
> open-shell benchmark is meant to probe.

Run it (from the repo root, after `conda activate $GOAD_ENV`):
```bash
python generate_surface_cifs.py       # ensures inputs/{Ag100..Rh111}.cif exist (idempotent)
python generate_molecule_cifs.py      # builds the adsorbate CIFs (skips existing)
mkdir -p slurm-logs runs              # Slurm needs the -o log dir to exist before submit
sbatch --array=0-587%50 -t 00:20:00 perlmutter/goad_array_perlmutter_gpu.slurm workflow/tasks_missing_sevennet.csv
```
Results land in `runs/C{n}/<surface>_<adsorbate>_seed<seed>_sevennet_omni/`.

### After the array finishes — collect the structures

`workflow/collect_missing_sevennet.py` walks `runs/`, picks the **best seed**
(lowest final `E_ads`) for each of the 14 × 21 = 294 systems, and extracts the
relaxed structure + energetics for evaluation:

```bash
python workflow/collect_missing_sevennet.py
```

It produces:
- `collected/sevennet_missing/<surface>_<adsorbate>_sevennet_omni.cif` — the
  best relaxed adsorbate-on-surface structure, named to match the SevenNet
  gallery convention (drop straight into `structure/` or link into `MANIFEST.csv`).
- `collected/sevennet_missing_summary.csv` — one row per system:
  `E_ads_eV`, `E_total/surface/molecule_eV`, and the **molecule–surface bond
  distance** (`min_surf_ads_dist_A`, `z_gap_A`) — the quantity this benchmark
  is about — plus `best_seed` and the source `run_dir`.
- a completeness report: any **MISSING** system prints its re-run `task_id`s,
  so you can resubmit just those with
  `sbatch --array=<ids> -t 00:20:00 perlmutter/goad_array_perlmutter_gpu.slurm workflow/tasks_missing_sevennet.csv`.


> 💰 **Wall time drives the NERSC cost gate.** The estimate = `N_tasks × -t × 0.25`
> (shared QOS, 1 A100 = ¼ node). The script's `#SBATCH -t 12:00:00` default is
> sized for the big `tasks_custom.csv` (large molecules) and would estimate this
> 588-task batch at 1764 node-hours — over a typical small balance. These
> adsorbates are all ≤ C2 and finish in **≤ ~11 min each** (measured on A100), so
> override with **`-t 00:20:00`** on the command line: estimate ≈ 49 node-hours,
> actual charge far less. Do **not** lower the in-script default — the full
> library run still needs the long wall.



---

## Submitting

Always submit **from the repo root** (the scripts `cd $SLURM_SUBMIT_DIR`).

### Smoke test first (a few tasks, short wall time)
```bash
export GOAD_ENV=$PSCRATCH/goad-env
sbatch --array=0-3 -t 0:30:00 perlmutter/goad_array_perlmutter_gpu.slurm workflow/tasks_custom.csv
```
Check `slurm-logs/` and `runs/C*/…/status.json` (`state: finished`) before scaling up.

### Scale up — let `find_missing_tasks.py` drive it (recommended)
This classifies finished/failed/missing and writes a self-paced `submit_missing.sh`
that only submits incomplete work (safe to re-run — finished tasks self-skip):

```bash
python find_missing_tasks.py \
    --slurm-script perlmutter/goad_array_perlmutter_gpu.slurm \
    --max-in-flight 4900          # stay under NERSC's 5000 submit limit
bash submit_missing.sh
```

It chunks tasks (default `--chunk 200`, well under NERSC's `MaxArraySize`≈1000) and
maps small **array indices → real `task_id`s** via `GOAD_TASK_ID_LIST`, which the
`.slurm` scripts already read. `--array=0-199%40` means 40 tasks run at once, each on
¼ of a GPU node.

### Or submit a raw contiguous range manually
```bash
sbatch --array=0-199%40 perlmutter/goad_array_perlmutter_gpu.slurm workflow/tasks_custom.csv
```

---

## Monitor / resume

```bash
squeue --me                              # running/pending
sacct -X -j <jobid> --format=JobID,State,Elapsed,MaxRSS
python find_missing_tasks.py             # re-classify; rewrites submit_missing.sh
bash submit_missing.sh                   # resubmit only what's left
python collect_results.py                # gather finished runs
```

---

## NERSC gotchas (read once)

- **QOS is mandatory.** Omitting `-q` lands you in `debug` (30-min, 8-node cap). The
  scripts set `-q shared`.
- **1–2 GPU jobs → `shared` QOS** (enforced by policy). `shared` charges only for the
  GPU fraction used: 1 A100 = ¼ node-hour.
- **Submit limit = 5000** jobs (`shared`/`regular`). Keep `--max-in-flight ≤ 4900`.
- **`MaxArraySize`** caps the array *index* (≈1000). The `--chunk 200` default keeps
  indices small; verify with `scontrol show config | grep MaxArraySize`.
- **CPU and GPU hours are separate allocations** — a project can have one and not the
  other. Check in [Iris](https://iris.nersc.gov).
- **`$PSCRATCH` is purged** after ~8 weeks of no access. Copy `runs/` results you want
  to keep to `$CFS/m5281/` or off-site (`collect_results.py` first).
- **`preempt` QOS** (`-q preempt`, add `--requeue`) is 4× cheaper after the first 2 h
  and a great fit here since tasks are idempotent/restartable — consider it for the big
  52k sweep once the GPU version is proven.

---

## Cost sketch

52,440 tasks × 1 A100 each in `shared` = ¼ node-hour per task-hour. If a typical task
finishes in ~10–20 min, that's ≈ 0.04–0.08 node-hr/task → order **2,000–4,000 GPU
node-hours** for a full sweep. Measure real per-task wall time from the smoke test and
right-size `-t` before the big submission (shared allows up to 48 h).

---

## VASP DFT on Perlmutter (full geo-opt from MLIP structures)

The MLIP (GOAD/SevenNet) relaxed geometries are the *starting point* for DFT. The
pipeline: **audit what DFT already exists → stage MLIP structures to POSCARs →
generate VASP inputs per functional → submit.** Functionals: `pbe`, `pbe-d3`,
`r2scan`, `beef-vdw`. Scope: the 25 well-known molecules × 7 non-magnetic metals ×
3 facets (378 systems; 377 already have an MLIP structure).

> VASP on Perlmutter is license-gated (the `vasp` unix group; request via a NERSC
> ticket). CPU build here: **`vasp-tpc/6.4.2-cpu`** (already set in the generated
> `slm.vasp.perlmutter`). POTCAR lib: `VASP_PP_PATH=/pscratch/sd/j/jcho5/paw64/potpaw_PBE_64`.
> beef-vdw kernel: `/pscratch/sd/j/jcho5/vdw_kernel.bindat`.

### 0. Audit existing (Kestrel) DFT — did we run full geo-opt or only single-point?
Run **on Kestrel**. Add `--manifest DFT_results/MANIFEST.csv --exclude-groups deoxy`
to focus the report on just the 478 well-known systems (instead of all ~5,400 in the
gallery):
```bash
python workflow/audit_dft_runs.py \
    --root /scratch/jcho5/goad-global-optimization/poscar/best \
    --manifest DFT_results/MANIFEST.csv --exclude-groups deoxy
```
Prints a system × functional table (`R+` relax-converged / `R-` relax-not-converged
/ `R.` running / `S` single-point / `.` none) and writes `dft_audit_matrix.csv` +
`dft_audit_jobs.csv`. Whatever shows `R+` is done; everything else needs running.

**Result (2026-08): the Kestrel runs are single-point only** — `sp_only≈2951`,
`relax_ok` 0–1 across pbe/pbe-d3/r2scan/beef-vdw. So the entire well-known scope
needs a full geo-opt here on Perlmutter.

### 1. Stage MLIP structures → POSCAR tree
**Already done for you** — `dft_jobs/` (377 POSCARs, bottom-2-layers fixed) is
committed in this repo, so on Perlmutter just `git pull` and skip to step 2. It was
staged locally because that needs `ase` + the 333 MB `structure/` gallery, neither
of which has to live on Perlmutter.

To (re)stage yourself — e.g. after the SevenNet gap-fill fills the 101 missing
systems (methoxy, acetylene, HCN, OH, atomic H/O …) — you need `structure/` +
`DFT_results/MANIFEST.csv` present, then:
```bash
python workflow/stage_dft_poscars.py \
    --manifest DFT_results/MANIFEST.csv --structure-dir structure \
    --out-dir dft_jobs --fix-bottom-layers 2
```
`--fix-bottom-layers 2` freezes the bottom 2 slab layers (Selective dynamics `F F F`)
for a proper geo-opt, matching the GOAD 4-layer/bottom-2-fixed slab.

### 2. Generate VASP inputs per functional (relax, Perlmutter CPU submit script)
```bash
export VASP_PP_PATH=/pscratch/sd/j/jcho5/paw64/potpaw_PBE_64   # your PBE PAW library
for f in pbe pbe-d3 r2scan beef-vdw; do
    python setup_vasp_jobs.py --poscar-dir dft_jobs --functional $f \
        --cluster perlmutter-cpu
done
```
This writes `dft_jobs/<system>/<FUNC>/{INCAR,KPOINTS,POTCAR,slm.vasp.perlmutter}`
(relax: `NSW=1000, IBRION=2, EDIFFG=-5E-02`). Edit the `module load vasp/...` line
in the template to your build. (`--calc-type single-point` inserts a `singlepoint/`
level instead, so relax and SP never collide.)

> **beef-vdw only:** needs `vdw_kernel.bindat`. Pass
> `--vdw-kernel-path /pscratch/sd/j/jcho5/vdw_kernel.bindat` (the setup script copies
> it into each beef-vdw job dir):
> ```bash
> python setup_vasp_jobs.py --poscar-dir dft_jobs --functional beef-vdw \
>     --cluster perlmutter-cpu \
>     --vdw-kernel-path /pscratch/sd/j/jcho5/vdw_kernel.bindat
> ```

### 3. Submit
```bash
for d in dft_jobs/*/{PBE,PBE_D3,r2scan,beef_vdw}/; do
    (cd "$d" && sbatch slm.vasp.perlmutter)
done
```
**Cost:** 377 × 4 functionals full slab relaxations is large. Start with **PBE
only** on a handful, measure wall time, then decide. For these small slabs the
`shared` QOS (fractional-node charging) is far cheaper than a full `regular` node —
see the commented block in `slm.vasp.perlmutter`.

