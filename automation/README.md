# Automation workflow (sleep mode)

This folder provides a lightweight autonomous loop for VASP campaigns:

- discovers jobs under `jobs_root`
- submits new/failed jobs via `sbatch`
- checks convergence from `OUTCAR`
- applies rule-based INCAR patches on failures
- re-submits up to `max_retries`
- writes reports and adsorption energies automatically

## Files

- `config.yaml` — runtime configuration
- `runner.py` — main loop
- `error_handlers.py` — failure detection + INCAR patching
- `analysis.py` — summary CSV + adsorption energy table + optional PNG plot

## Quick start

```bash
python automation/runner.py --config automation/config.yaml --once
```

Continuous mode:

```bash
python automation/runner.py --config automation/config.yaml
```

## Notes

- Ensure `jobs_root` points to the directory containing your generated job folders.
- `runner.py` classifies jobs by path patterns (`vasp_slab`, `vasp_mol`, otherwise adsorption).
- For robust queue tracking, you can later extend this with `squeue/sacct` parsing.
