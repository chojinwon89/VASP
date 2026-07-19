# GOAD-MLIP Workflow — Quick Start

## Prerequisites
- Create and activate the conda environment you use for GOAD on Kestrel.
- Install the Python requirements:
  - `pip install -r requirements.txt`
- Install RDKit with conda if it is not already available:
  - `conda install -c conda-forge rdkit`

## Supported surfaces

| Metal | Structure | (111/0001) | (110) | (100) |
|-------|-----------|------------|-------|-------|
| Cu    | FCC       | Cu111      | Cu110 | Cu001 |
| Pt    | FCC       | Pt111      | Pt110 | Pt100 |
| Pd    | FCC       | Pd111      | Pd110 | Pd100 |
| Ni    | FCC       | Ni111      | Ni110 | Ni100 |
| Ag    | FCC       | Ag111      | Ag110 | Ag100 |
| Au    | FCC       | Au111      | Au110 | Au100 |
| Ir    | FCC       | Ir111      | Ir110 | Ir100 |
| Rh    | FCC       | Rh111      | Rh110 | Rh100 |
| Fe    | BCC       | Fe111      | Fe110 | Fe100 |
| Cr    | BCC       | Cr111      | Cr110 | Cr100 |
| Mo    | BCC       | Mo111      | Mo110 | Mo100 |
| Ru    | HCP       | Ru0001     | —     | —     |
| Co    | HCP       | Co0001     | —     | —     |
| Ti    | HCP       | Ti0001     | —     | —     |
| Zn    | HCP       | Zn0001     | —     | —     |

All slabs: 4×4×4, 15 Å vacuum, built with ASE using experimental lattice constants.

## Supported molecules

### Alkane
| Name       | SMILES       | Carbon |
|------------|--------------|--------|
| methane    | `C`          | C1     |
| ethane     | `CC`         | C2     |
| propane    | `CCC`        | C3     |
| butane     | `CCCC`       | C4     |
| isobutane  | `CC(C)C`     | C4     |
| pentane    | `CCCCC`      | C5     |
| isopentane | `CC(C)CC`    | C5     |
| hexane     | `CCCCCC`     | C6     |
| heptane    | `CCCCCCC`    | C7     |
| octane     | `CCCCCCCC`   | C8     |

### Alkene
| Name      | SMILES       | Carbon |
|-----------|--------------|--------|
| ethene    | `C=C`        | C2     |
| propene   | `CC=C`       | C3     |
| 1-butene  | `CCC=C`      | C4     |
| 2-butene  | `CC=CC`      | C4     |
| isobutene | `CC(=C)C`    | C4     |
| 1-pentene | `CCCC=C`     | C5     |
| butadiene | `C=CC=C`     | C4     |
| isoprene  | `CC(=C)C=C`  | C5     |

### Aromatic
| Name        | SMILES            | Carbon |
|-------------|-------------------|--------|
| benzene     | `c1ccccc1`        | C6     |
| toluene     | `Cc1ccccc1`       | C7     |
| furan       | `c1ccoc1`         | C4     |
| pyrrole     | `c1cc[nH]c1`      | C4     |
| thiophene   | `c1ccsc1`         | C4     |
| styrene     | `C=Cc1ccccc1`     | C8     |
| xylene      | `Cc1ccc(C)cc1`    | C8     |
| phenol      | `Oc1ccccc1`       | C6     |
| aniline     | `Nc1ccccc1`       | C6     |
| naphthalene | `c1ccc2ccccc2c1`  | C10    |

### Furan
| Name       | SMILES        | Carbon |
|------------|---------------|--------|
| 2-furanone | `O=C1C=CCO1`  | C4     |

### Alcohol
| Name        | SMILES                | Carbon |
|-------------|-----------------------|--------|
| methanol    | `CO`                  | C1     |
| ethanol     | `CCO`                 | C2     |
| ethylene_glycol | `OCCO`            | C2     |
| isopropanol | `CC(C)O`              | C3     |
| propanol    | `CCCO`                | C3     |
| glycerol    | `OCC(O)CO`            | C3     |
| 1-butanol   | `CCCCO`               | C4     |
| 2-butanol   | `CCC(O)C`             | C4     |
| pentanol    | `CCCCCO`              | C5     |
| sorbitol    | `OCC(O)C(O)C(O)C(O)CO` | C6   |
| xylitol     | `OCC(O)C(O)C(O)CO`    | C5     |

### Aldehyde
| Name             | SMILES          | Carbon |
|------------------|-----------------|--------|
| formaldehyde     | `C=O`           | C1     |
| acetaldehyde     | `CC=O`          | C2     |
| propanal         | `CCC=O`         | C3     |
| butanal          | `CCCC=O`        | C4     |
| furfural         | `O=Cc1ccco1`    | C5     |
| valeraldehyde    | `CCCCC=O`       | C5     |
| hexanal          | `CCCCCC=O`      | C6     |
| 5-HMF            | `OCc1ccc(C=O)o1` | C6    |
| 5-methylfurfural | `Cc1ccc(C=O)o1` | C6     |
| benzaldehyde     | `O=Cc1ccccc1`   | C7     |

### Carbonyl
| Name    | SMILES   | Carbon |
|---------|----------|--------|
| glyoxal | `O=CC=O` | C2     |

### Ketone
| Name              | SMILES             | Carbon |
|-------------------|--------------------|--------|
| acetone           | `CC(=O)C`          | C3     |
| methylethylketone | `CCC(=O)C`         | C4     |
| cyclobutanone     | `O=C1CCC1`         | C4     |
| 2-pentanone       | `CCCC(=O)C`        | C5     |
| cyclopentanone    | `O=C1CCCC1`        | C5     |
| 2-hexanone        | `CCCCC(=O)C`       | C6     |
| cyclohexanone     | `O=C1CCCCC1`       | C6     |
| 5-heptanone       | `CCCCC(=O)CC`      | C7     |
| 2-heptanone       | `CCCCCC(=O)C`      | C7     |
| acetophenone      | `CC(=O)c1ccccc1`   | C8     |

### Carboxylic acid
| Name          | SMILES              | Carbon |
|---------------|---------------------|--------|
| formic_acid   | `OC=O`              | C1     |
| acetic_acid   | `CC(=O)O`           | C2     |
| oxalic_acid   | `OC(=O)C(=O)O`      | C2     |
| propionic_acid | `CCC(=O)O`         | C3     |
| malonic_acid  | `OC(=O)CC(=O)O`     | C3     |
| butyric_acid  | `CCCC(=O)O`         | C4     |
| succinic_acid | `OC(=O)CCC(=O)O`    | C4     |
| valeric_acid  | `CCCCC(=O)O`        | C5     |
| glutaric_acid | `OC(=O)CCCC(=O)O`   | C5     |
| caproic_acid  | `CCCCCC(=O)O`       | C6     |

### Hydroxy/keto acid
| Name                    | SMILES                             | Carbon |
|-------------------------|------------------------------------|--------|
| glycolic_acid           | `OCC(=O)O`                         | C2     |
| lactic_acid             | `CC(O)C(=O)O`                      | C3     |
| pyruvic_acid            | `CC(=O)C(=O)O`                     | C3     |
| 3-hydroxypropionic_acid | `OCCC(=O)O`                        | C3     |
| malic_acid              | `OC(CC(=O)O)C(=O)O`                | C4     |
| tartaric_acid           | `OC(C(O)C(=O)O)C(=O)O`             | C4     |
| itaconic_acid           | `OC(=O)CC(=C)C(=O)O`               | C5     |
| levulinic_acid          | `CC(=O)CCC(=O)O`                   | C5     |
| citric_acid             | `OC(=O)CC(O)(C(=O)O)CC(=O)O`       | C6     |
| gluconic_acid           | `OCC(O)C(O)C(O)C(O)C(=O)O`         | C6     |
| muconic_acid            | `OC(=O)C=CC=CC(=O)O`               | C6     |

### Phenols
| Name          | SMILES          | Carbon |
|---------------|-----------------|--------|
| 2-ethylphenol | `CCc1ccccc1O`   | C8     |
| hydroquinone  | `Oc1ccc(O)cc1`  | C6     |

### Guaiacols
| Name            | SMILES                  | Carbon |
|-----------------|-------------------------|--------|
| guaiacol        | `COc1ccccc1O`           | C7     |
| 4-methylguaiacol | `Cc1ccc(O)c(OC)c1`     | C8     |
| eugenol         | `C=CCc1ccc(O)c(OC)c1`   | C10    |
| isoeugenol      | `C/C=C/c1ccc(O)c(OC)c1` | C10    |

### Syringols
| Name             | SMILES                        | Carbon |
|------------------|-------------------------------|--------|
| syringol         | `COc1cccc(OC)c1O`             | C8     |
| propyl_syringol  | `CCCc1cc(OC)c(O)c(OC)c1`      | C11    |
| syringaldehyde   | `COc1cc(C=O)cc(OC)c1O`        | C9     |

### Sugars
| Name                     | SMILES                 | Carbon |
|--------------------------|------------------------|--------|
| levoglucosan             | `OC1C(O)C(O)C2COC1O2`  | C6     |
| alpha-D-glucopyranose    | `OCC1OC(O)C(O)C(O)C1O` | C6     |
| D-fructofuranose         | `OCC1(O)OCC(O)C1O`     | C6     |
| D-xylopyranose           | `OC1COC(O)C(O)C1O`     | C5     |
| 1,6-anhydroglucofuranose | `OC1C2COC1OC2O`        | C6     |

### Ester/ether
| Name                | SMILES                  | Carbon |
|---------------------|-------------------------|--------|
| DME                 | `COC`                   | C2     |
| DMSO                | `CS(=O)C`               | C2     |
| methyl_formate      | `COC=O`                 | C2     |
| 3-MTHF              | `CC1CCCO1`              | C5     |
| methylmethacrylate  | `COC(=O)C(=C)C`         | C5     |
| angelica_lactone    | `CC1=CCC(=O)O1`         | C5     |
| gamma_butyrolactone | `O=C1CCCO1`             | C4     |
| diethyl_ether       | `CCOCC`                 | C4     |
| THF                 | `C1CCOC1`               | C4     |
| ethyl_acetate       | `CC(=O)OCC`             | C4     |
| furfuryl_alcohol    | `OCc1ccco1`             | C5     |
| gamma_valerolactone | `CC1CCC(=O)O1`          | C5     |
| dimethyl_succinate  | `COC(=O)CCC(=O)OC`      | C6     |

### Oxygenates
| Name                  | SMILES              | Carbon |
|-----------------------|---------------------|--------|
| hydroxyacetaldehyde   | `OCC=O`             | C2     |
| acetal                | `CC(OCC)OCC`        | C6     |
| methylcyclopentenolone | `CC1=C(O)CCC1=O`   | C6     |
| vanillin              | `COc1cc(C=O)ccc1O`  | C8     |

### C1 reference
| Name         | Notes                                    |
|--------------|------------------------------------------|
| CO           | carbon monoxide                          |
| CO2          | carbon dioxide                           |
| methane / CH4 | same structure                          |
| methanol     | simplest alcohol                         |
| formaldehyde | simplest aldehyde                        |
| formate      | carboxylate anion (approximation)        |

### Inorganic
| Name | Formula | Notes                  |
|------|---------|------------------------|
| H2   | H₂      | hydrogen gas           |
| H2O  | H₂O     | water                  |
| N2   | N₂      | nitrogen gas           |
| O2   | O₂      | oxygen gas             |
| NH3  | NH₃     | ammonia                |
| NO   | NO      | nitric oxide           |
| NO2  | NO₂     | nitrogen dioxide       |
| SO2  | SO₂     | sulfur dioxide         |
| H2S  | H₂S     | hydrogen sulfide       |

## Skip-if-finished behaviour

`workflow/run_one_task.py` automatically skips any task whose run directory
already contains a `status.json` with `state == "finished"`, exiting with
code 0 so Slurm treats it as success.  This makes bulk re-submission via
`submit_missing.sh` (generated by `find_missing_tasks.py`) safe to repeat
without wasting compute on already-completed work.

Use `--force` to override this check:

```bash
python workflow/run_one_task.py --task-id 42 --force
```

`find_missing_tasks.py` now also generates a self-pacing `submit_missing.sh`
by default: before each chunked `sbatch` call, the script polls `squeue` and
waits until the user's pending+running array-task count drops below
`--max-in-flight` (default: `9000`), checking every `--poll-interval` seconds
(default: `60`). This helps avoid cluster-wide QOS submit caps such as
`MaxSubmitPU`.

```bash
# Use a lower cap if your cluster's QOS MaxSubmitPU is smaller than 9000
python find_missing_tasks.py --max-in-flight 3000 --poll-interval 30

# Disable pacing entirely (e.g. no such QOS restriction on your cluster)
python find_missing_tasks.py --max-in-flight 0
```

## Step-by-step

### Step 1 — Generate input structures
```bash
cd /scratch/jcho5/goad-global-optimization
python generate_surface_cifs.py
python generate_molecule_cifs.py
```
Builds all surface CIFs and all molecule CIFs under `inputs/`.

### Step 2 — Generate task table
```bash
python workflow/make_tasks_custom.py
```
Writes `workflow/tasks_custom.csv`.

### Step 3 — Test one task interactively
```bash
python workflow/run_one_task.py --task-id 0 --tasks-csv workflow/tasks_custom.csv
```

### Step 4 — Submit full Slurm array
```bash
mkdir -p slurm-logs runs
sbatch --array=0-<N>%50 goad_array_kestrel.slurm workflow/tasks_custom.csv
```

### Step 5 — Monitor
```bash
squeue -u $USER
tail -f slurm-logs/goad_array-*.out
```

### Step 6 — Collect results
```bash
python collect_results.py
column -s, -t < workflow/summary.csv
```

### Step 7 — Extract best geometries
```bash
python extract_poscar.py --verbose
```

### Step 8 — Set up DFT adsorbed jobs (all 3 seeds)
```bash
export VASP_PP_PATH=/home/jcho5/project/paw64/potpaw_PBE_64
python setup_vasp_jobs.py --all-seeds
```

`setup_vasp_jobs.py` supports:
- `--calc-type relax` (default): full ionic relaxation (`NSW=1000`, `IBRION=2`, `EDIFFG=-5E-02`)
- `--calc-type single-point`: single-point energy/forces on the input geometry (`NSW=0`, `IBRION=-1`, no `EDIFFG`)

Single-point DFT across all 4 supported functionals on best-seed GOAD+MLIP geometries:

```bash
python extract_poscar.py --best-only --out-dir poscar/best

for func in pbe pbe-d3 r2scan beef-vdw; do
    python setup_vasp_jobs.py --poscar-dir poscar/best \
                               --functional $func \
                               --calc-type single-point
done

for d in poscar/best/*/*/singlepoint/{PBE,PBE_D3,r2scan,beef_vdw}/; do
    (cd "$d" && sbatch slm.vasp.kestrel)
done
```

### Step 9 — Set up DFT bare-slab reference jobs
```bash
python setup_slab_jobs.py
```
Creates `vasp_slab/<SurfaceName>/` with POSCAR (Selective Dynamics), INCAR, KPOINTS, POTCAR, Slurm script.
Supports all metals listed in the Supported surfaces table above (FCC, BCC, HCP).

### Step 10 — Set up DFT gas-phase molecule jobs
```bash
python setup_molecule_jobs.py
```
Creates `vasp_mol/<MoleculeName>/` with POSCAR (20×20×20 Å box), INCAR (ISMEAR=0, Gamma-point), KPOINTS (1×1×1), POTCAR, Slurm script.

## Adsorption energy formula

```
E_ads = E_total(slab+mol) - E_surf(slab) - E_mol(gas)
```

## Submit all DFT jobs

```bash
# Adsorbed systems
for d in poscar/*/; do (cd "$d" && sbatch slm.vasp.kestrel); done

# Bare slabs
for d in vasp_slab/*/; do (cd "$d" && sbatch slm.vasp.kestrel); done

# Gas molecules
for d in vasp_mol/*/; do (cd "$d" && sbatch slm.vasp.kestrel); done
```
