# HANDOFF — GOAD + SevenNet → DFT validation benchmark

Portable context for continuing this work in **VS Code GitHub Chat** (or any new
Copilot session). Everything needed is committed to
`github.com/chojinwon89/VASP` — clone it and this file + `README_Perlmutter.md`
give a new chat the full picture.

Last updated: 2026-08-25 · VASP repo `main` @ `5c60dc5`.

---

## 1. What this project is

A validation benchmark that checks whether MLIP-relaxed adsorbate/surface
geometries (GOAD driving **SevenNet**, compared against **MatterSim**) match DFT.
The grid is **477 physical systems** = 25 molecules × 7 non-magnetic metals
{Ag, Au, Cu, Ir, Pd, Pt, Rh} × 3 facets {100,110,111} (6 radicals only on their
13 in-scope surfaces). Magnetic metals (Ni/Co/Fe/Cr) and the `deoxy` group are
out of scope.

Two stages:
- **Stage A — MLIP (GOAD/SevenNet, GPU):** relax each system → one CIF per
  surface+molecule, collected into the gallery.
- **Stage B — DFT (VASP, CPU):** full geometry optimization with PBE / PBE-D3 /
  r2scan / BEEF-vdW to validate the MLIP geometries and adsorption energies.

**Current status:** 476/477 systems staged for DFT. Only `C2H6_Pt111` (ethane on
Pt111) is missing — its 2 SevenNet seeds need re-running (see §6). DFT input
generation + submission tooling is complete and pushed; VASP runs are being
launched on Perlmutter.

---

## 2. Where everything lives

### GitHub repositories (source of truth)
| Repo | URL | Purpose |
|---|---|---|
| **VASP** | `github.com/chojinwon89/VASP` (`main`) | All workflow code, DFT tooling, staged POSCARs, MANIFEST |
| **bond-distance-review** | `github.com/chojinwon89/bond-distance-review` → `https://chojinwon89.github.io/bond-distance-review/` | The visual website (MLIP benchmark gallery + tables) |

### Local clones / working dirs on this Mac
| Path | What |
|---|---|
| `/Users/jcho2/.copilot/repos/VASP` | Local clone of the VASP repo (the code) |
| `/tmp/bond-pages` | Local clone of the website repo (`bond-distance-review`); site is published from here |
| `/Users/jcho2/Desktop/Pymatgen/GOAD+Sevennet Structures` | Workspace (symlink → `GOAD+Sevennet_Structures`); scratch CIFs/PNGs/reports |
| `~/.copilot/session-state/<session>/files/` | MLIP benchmark generators + data: `mlip_benchmark.py`, `generate_mlip_html.py`, `mlip_bench/` (results JSON + figures) |

### Clusters (remote)
| Cluster | Working dir | Role |
|---|---|---|
| **Perlmutter** (NERSC) | `/pscratch/sd/j/jcho5/VASP` | New DFT full-relax runs (Stage B) + MLIP gap-fill |
| **Kestrel** (NREL) | `/scratch/jcho5/goad-global-optimization` | Older single-point DFT + some RPBE relaxes (audited) |

### Key cluster paths (Perlmutter)
- VASP module: `vasp-tpc/6.4.2-cpu`
- POTCAR library: `/pscratch/sd/j/jcho5/paw64/potpaw_PBE_64`
- vdW kernel (BEEF-vdW): `/pscratch/sd/j/jcho5/vdw_kernel.bindat`

These POTCAR/vdW paths are now the **built-in `--cluster perlmutter-cpu`
defaults**, so no env-var export is required.

---

## 3. Continuing in VS Code GitHub Chat

```bash
# clone the code repo (has everything: tooling + README_Perlmutter + this file)
git clone https://github.com/chojinwon89/VASP.git
code VASP
```

Then in GitHub Chat, point it at the repo and read `README_Perlmutter.md`
(full runbook) and this `HANDOFF.md`. The new chat has **full repo access** the
moment the repo is open in the workspace — all work described here is already on
`main` (`git log --oneline -8` shows the session's commits, §5).

> The website repo is separate. To update the site, clone it too:
> `git clone https://github.com/chojinwon89/bond-distance-review.git`.
> Its build/publish flow lives in the session `files/` dir (see §7).

---

## 4. The workflow (short form)

Full detail: **`README_Perlmutter.md`**. In brief:

```bash
cd /pscratch/sd/j/jcho5/VASP && git pull
module load vasp-tpc/6.4.2-cpu

# 1. what's missing?
python workflow/check_dft_coverage.py                         # -> dft_coverage.csv

# 2. Stage A: MLIP gap-fill (only if something is missing_no_cif)
python workflow/make_tasks_missing_sevennet.py                # radicals
python workflow/make_tasks_existing_gap.py                    # literature systems
sbatch --array=... -t 00:20:00 perlmutter/goad_array_perlmutter_gpu.slurm workflow/<tasks>.csv
python workflow/collect_missing_sevennet.py --tasks-csv workflow/<tasks>.csv
python workflow/import_sevennet_to_gallery.py                 # collected -> structure/ + MANIFEST
python workflow/stage_dft_poscars.py --manifest DFT_results/MANIFEST.csv \
    --structure-dir structure --out-dir dft_jobs --fix-bottom-layers 2
python workflow/check_dft_coverage.py                         # target 477/477

# 3. Stage B: DFT inputs (incremental — only new/incomplete systems)
for f in pbe pbe-d3 r2scan beef-vdw; do
  python setup_vasp_jobs.py --poscar-dir dft_jobs --functional $f \
      --cluster perlmutter-cpu --skip-existing
done

# 4. submit (auto-skips already-converged jobs)
mkdir -p slurm-logs
python workflow/make_dft_joblist.py --functional pbe > joblist_pbe.txt
N=$(grep -vc '^#' joblist_pbe.txt)
sbatch --array=0-$((N-1))%20 perlmutter/vasp_dft_array_cpu.slurm joblist_pbe.txt

# 4b. OR submit exactly the wave a setup run just created:
python setup_vasp_jobs.py --poscar-dir dft_jobs --functional pbe \
    --cluster perlmutter-cpu --skip-existing --emit-joblist joblist_new_pbe.txt
N=$(grep -vc '^#' joblist_new_pbe.txt)
sbatch --array=0-$((N-1))%20 perlmutter/vasp_dft_array_cpu.slurm joblist_new_pbe.txt

# 5. after a batch drains, resubmit any crashed/unconverged
python workflow/resubmit_dft.py --jobs-dir dft_jobs --functional pbe --submit
```

**Incremental guarantee (two layers, so converged DFT is never disturbed):**
1. `setup_vasp_jobs.py --skip-existing` — writes inputs only for systems that
   aren't already fully set up (have both INCAR + POTCAR).
2. `make_dft_joblist.py` / `resubmit_dft.py` — only non-converged OUTCARs enter
   the array (whole-OUTCAR scan for `reached required accuracy`).

---

## 5. Tooling added this session (all on VASP `main`)

| Commit | What |
|---|---|
| `e6183eb` | Fix DFT convergence detection: scan the **whole** OUTCAR (mmap), not just the tail, in `make_dft_joblist.py` + `resubmit_dft.py`. Converged jobs were mislabeled crashed. |
| `9c65e81` | `workflow/check_dft_coverage.py` — reconcile in-scope MANIFEST vs staged `dft_jobs/`; classify gaps (`missing_no_cif` vs staging bug); `--system` lookup. |
| `37be2bb` | `workflow/import_sevennet_to_gallery.py` (collected CIF → `structure/` + fill MANIFEST) + `workflow/make_tasks_existing_gap.py` (tasks for ~22 non-radical gaps). |
| `a88b653` | `setup_vasp_jobs.py --skip-existing` + this repo's `README_Perlmutter.md`. |
| `0296fa5` | Per-cluster POTCAR/vdW defaults (Perlmutter writes POTCAR automatically); `--skip-existing` regenerates a dir missing its POTCAR. |
| `5c60dc5` | `setup_vasp_jobs.py --emit-joblist` — record exactly the dirs created this run and print the `sbatch` line. |

Script reference table + quick recipe: bottom of `README_Perlmutter.md`.

---

## 6. Outstanding / next steps

1. **Close the last gap (476 → 477):** re-run the 2 `C2H6_Pt111` SevenNet tasks
   in `workflow/tasks_existing_gap.csv`, then `collect → import → stage`.
2. **Run the DFT arrays** on Perlmutter for all four functionals (§4).
3. **Phase 7 — analysis (not yet built):** parse converged OUTCARs → fill the
   MANIFEST `dft_*` columns (`dft_Eads_eV`, `dft_dMX_ang`, `dft_MX_pair`,
   `dft_dint_ang`) → `plot_dft_vs_mlip.py` renders MLIP-vs-DFT. This is the
   natural next tool to write.
4. **Website:** decide what the site should show for the DFT phase (see §7).

---

## 7. The visual website

- **Published:** `https://chojinwon89.github.io/bond-distance-review/`
  (main page `mlip_benchmark.html`).
- **Repo:** `github.com/chojinwon89/bond-distance-review`, local clone
  `/tmp/bond-pages`.
- **Built by:** `generate_mlip_html.py` (in the session `files/` dir), which
  reads `mlip_bench/*.json` (SevenNet/MatterSim results + figures) and writes the
  HTML + assets into `/tmp/bond-pages/`. Publish = commit + push that repo.
- **Current content:** MLIP benchmark (SevenNet vs MatterSim geometry/energy
  tables + per-system image gallery) over the 28 benchmarked non-magnetic
  systems. A **"DFT validation in progress on Perlmutter"** status banner (added
  2026-08-25, pages commit `dd556b8`) links to `README_Perlmutter.md`. No DFT
  numbers yet (Stage B still running); the banner is the placeholder until
  converged OUTCARs populate DFT columns.

To refresh/publish:
```bash
python ~/.copilot/session-state/<session>/files/generate_mlip_html.py   # -> /tmp/bond-pages/
cd /tmp/bond-pages && git add -A && git commit -m "..." && git push
```
