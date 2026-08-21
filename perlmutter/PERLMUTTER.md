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

### Gap-fill: the 6 missing SevenNet structures

Six adsorbates have **no** SevenNet-OMNI gallery structure yet:
`acetylene`, `methoxy`, `HCN`, `hydroxyl`, `atomicH`, `atomicO`.
A dedicated, self-contained task list covers them on all 7 non-magnetic metals:

- `workflow/tasks_missing_sevennet.csv` — **252 tasks**
  (6 adsorbates × {Ag,Au,Cu,Ir,Pd,Pt,Rh} × {100,110,111} × 2 seeds × `sevennet_omni`).
- Regenerate with `python workflow/make_tasks_missing_sevennet.py`.

Their gas-phase CIFs are already committed under `inputs/`, and
`molecule_utils.py` / `generate_molecule_cifs.py` know how to (re)build them
(ASE G2 for the 4 molecular species, single-atom `Atoms()` for H/O).

Run it (from the repo root, after `conda activate $GOAD_ENV`):
```bash
python generate_surface_cifs.py       # ensures inputs/{Ag100..Rh111}.cif exist (idempotent)
python generate_molecule_cifs.py      # builds the 6 adsorbate CIFs (skips existing)
sbatch --array=0-251%50 perlmutter/goad_array_perlmutter_gpu.slurm workflow/tasks_missing_sevennet.csv
```
Results land in `runs/C{n}/<surface>_<adsorbate>_seed<seed>_sevennet_omni/`.


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

## Also need VASP DFT here?

This port covers the **GOAD (MLIP)** workflow only. The repo's `setup_vasp_jobs.py` /
`extract_poscar.py` path targets DFT — NERSC provides `module load vasp/6.x.x-gpu`, but
access is restricted to license-verified users (the `vasp` unix group; request via a
NERSC ticket). Ask and I'll add matching `*_perlmutter_vasp.slurm` scripts.
