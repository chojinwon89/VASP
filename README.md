# GOAD + SevenNet → DFT adsorption benchmark

Does a fast machine-learning interatomic potential (**GOAD** global optimization
driving the **SevenNet-OMNI** MLIP) reproduce DFT adsorption geometries and
energies well enough to replace high-throughput DFT screening? This repo is the
tooling that answers that across a grid of **metal-surface + molecule** systems,
and publishes the result as a live website:

**→ <https://chojinwon89.github.io/bond-distance-review/>**

The repo holds two things:

| | |
|---|---|
| **The benchmark pipeline** | structure creation → MLIP relaxation → DFT geometry optimization → binding energy → MLIP-vs-DFT comparison → website. Documented in **[`PIPELINE.md`](PIPELINE.md)**. |
| **The GOAD engine** | the upstream Tkinter genetic-algorithm app that produces the adsorption structures. Documented in **[`GOAD_ENGINE.md`](GOAD_ENGINE.md)** (see also [`QUICK_START.md`](QUICK_START.md)). How the GA finds the lowest-energy structure — and how that differs from a DFT relaxation — is explained in **[`GOAD_ALGORITHM.md`](GOAD_ALGORITHM.md)**. |

## Quick start

Everything runs through one conductor, `pipeline.py`, which chains the individual
tools into named stages (full manual recipe in [`PIPELINE.md`](PIPELINE.md)):

```bash
python pipeline.py list          # show every stage (local vs cluster)
python pipeline.py status         # coverage gate: what DFT systems are missing?
python pipeline.py all --dry-run  # print the whole plan without running anything
python pipeline.py all            # run the canonical chain (local stages here,
                                  # full chain on the cluster)
```

Local (laptop) stages only need `ase`; the DFT stages need `sbatch` + VASP on a
cluster (Perlmutter or Kestrel) and are previewed instead of run when `sbatch`
is absent.

## Layout

```
pipeline.py            # the one-command conductor over every stage
PIPELINE.md            # the full benchmark manual (Stage A, Stage B, analysis, website)
GOAD_ENGINE.md         # the upstream GOAD genetic-algorithm app
generate_*_cifs.py     # build surface + molecule input CIFs (inputs/)
mol_canon.py           # shared molecule-name map (ethane≡C2H6) — edit species here once
setup_vasp_jobs.py     # write INCAR/KPOINTS/POTCAR/slurm per functional
calc_binding_energy.py # E_ads = E(slab+mol) − E(slab) − E(gas)
plot_dft_vs_mlip.py    # MLIP-vs-DFT parity plot + pairs CSV
build_dft_pages.py     # generate the live DFT website pages
build_validation_page.py # the "can MLIP replace DFT?" method-validation page
workflow/              # cluster pipeline steps (coverage, staging, joblists, resubmit)
automation/            # autonomous submit → check → INCAR-patch → retry loop
perlmutter/            # Perlmutter Slurm array scripts + env setup
goad_v1/               # the GOAD engine package (see GOAD_ENGINE.md)
tests/                 # pytest suite
```

## Tests

```bash
python -m pytest -q tests/
```

## Credits

The GOAD engine is by Urbiztondo, Castro-Palacio, Grau-Crespo & Hamad
([Zenodo 10.5281/zenodo.17904742](https://doi.org/10.5281/zenodo.17904742));
see [`GOAD_ENGINE.md`](GOAD_ENGINE.md) and [`CITATION.cff`](CITATION.cff). This
benchmark pipeline is built on top of it. Licensed under the MIT License
([`LICENSE`](LICENSE)).
