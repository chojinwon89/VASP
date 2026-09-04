# `ml/` — Fine-tuning & active-learning starter kit

Goal: **improve MLIP (SevenNet) accuracy on our own GOAD adsorption systems** by
fine-tuning on our DFT data, then closing an active-learning loop so DFT is spent
only where the model is unsure. Everything here is a starting scaffold — each
script has a small, testable core function and a CLI.

The benchmark work showed SevenNet keeps molecular *shapes* correct but its
surface *binding* (and especially magnetic 3d metals) drifts from DFT. Fine-tuning
on our labelled relaxations is the direct fix.

```
ml/
  env.yml            conda environment (torch, sevenn, ase, sklearn, BO, rdkit)
  finetune.yaml      SevenNet fine-tune config template (verify vs your sevenn)
  scripts/
    extract_vasp.py     vasprun.xml trajectories -> labelled train.extxyz
    split_by_system.py  split by SYSTEM (no frame leakage) -> train/valid
    validate.py         energy/force MAE/RMSE + parity plot (base vs fine-tuned)
    active_learning.py  ensemble-disagreement ranking -> to_label.extxyz
    descriptors.py      cheminformatics features + error stratification
  data/   models/       (git-ignored artifacts you generate)
```

---

## Phase 0 — Data foundation (do this first)

MLIPs learn from **per-structure energy + forces**, not from one binding energy
per system. The gold source is the VASP relaxation *trajectories* in `dft_jobs/`
— every ionic step is a free labelled frame.

**Pick ONE functional.** Mixing PBE / PBE+D3 / r²SCAN / BEEF-vdW in one training
set teaches contradictory labels. PBE+D3 matches our production SevenNet modal
(`omat24`) best, so start there.

```bash
# 1. trajectories -> one labelled dataset (tag the functional!)
python ml/scripts/extract_vasp.py --jobs dft_jobs --functional pbe_d3 \
    --stride 2 --max-frames-per-system 60 --out ml/data/all.extxyz

# 2. split by SYSTEM, not by frame
python ml/scripts/split_by_system.py --in ml/data/all.extxyz --train 0.9

# 3. zero-shot baseline error of the CURRENT model (the number to beat)
python ml/scripts/validate.py --ref ml/data/valid.extxyz \
    --model 7net-mf-ompa --modal omat24 --csv-out ml/base.csv --plot ml/parity_base.png
```

## Phase 1 — Fine-tune SevenNet

```bash
# build graphs at the SAME cutoff used in finetune.yaml (5.0 A)
sevenn_graph_build ml/data/train.extxyz 5.0 -o ml/data/train.sevenn_data
sevenn_graph_build ml/data/valid.extxyz 5.0 -o ml/data/valid.sevenn_data

# get a schema-correct config for YOUR sevenn, then merge finetune.yaml values in
sevenn_preset fine_tune > ml/_preset.yaml   # compare keys, then edit finetune.yaml

sevenn ml/finetune.yaml -s                  # train (continues from 7net-omat)
sevenn_get_model 7net checkpoint_best.pth -o ml/models/ft_pbe_d3.pt

# did it help? compare to ml/base.csv on the SAME held-out systems
python ml/scripts/validate.py --ref ml/data/valid.extxyz \
    --model ml/models/ft_pbe_d3.pt --csv-out ml/ft.csv --plot ml/parity_ft.png
```

Deploy by adding a `get_sevennet_finetuned()` to `goad_v1/calculator_manager.py`
that points `SevenNetCalculator` at `ml/models/ft_pbe_d3.pt`, then run GOAD with
that calculator.

> ⚠️ `sevenn` CLI flag/key names drift between releases. Always regenerate the
> preset with your installed version and reconcile before a long run.

## Phase 2 — Active learning (spend DFT where the model is unsure)

Train 2–3 checkpoints with different seeds, then rank GOAD-proposed structures by
ensemble disagreement and DFT-label only the top-K.

```bash
python ml/scripts/active_learning.py --pool ml/data/candidates.extxyz \
    --models ml/models/ft_seed1.pt ml/models/ft_seed2.pt ml/models/ft_seed3.pt \
    --top-k 50 --out ml/data/to_label.extxyz
# -> DFT single-point those 50 -> extract_vasp -> add to train -> retrain -> repeat
```

## Phase 3 — Bayesian optimization & cheminformatics

* **Descriptors / error stratification** — find which chemistries are worst:
  ```bash
  python ml/scripts/descriptors.py --systems ml/data/valid.extxyz \
      --errors ml/per_system_err.csv --out ml/desc.csv
  ```
* **Bayesian optimization** (`scikit-optimize` / `ax-platform`, in `env.yml`) —
  use a GP surrogate over adsorption-site / geometry parameters to propose the
  next configuration to evaluate instead of brute-forcing the GA population.

## Phase 4 — AI hypothesis generation

Once `descriptors.py` + validation errors are joined, feed the "worst
chemistries / systematic residual" table to an LLM to propose *testable*
hypotheses (e.g. "under-binds oxophilic O-adsorbates on Cr/Fe") and let the
active-learning loop confirm or refute them with new DFT.

---

### Local testing without DFT / GPU

The pure functions (`prepare_frames`, `split_systems`, `compute_errors`,
`ensemble_uncertainty`, `parse_system`) are unit-testable with synthetic ASE
`Atoms` + the built-in `EMT` calculator — no VASP, sevenn, or CUDA required. See
`scripts/tests/test_pipeline.py`.
