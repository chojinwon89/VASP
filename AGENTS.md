# AGENTS.md

## Repository overview
- Project name: **GOAD / VASP workflow utilities**
- Primary language: **Python**
- Main purposes:
  - GUI-driven adsorption optimization workflow in `goad_v1/`
  - Workflow and job-preparation utilities in `workflow/`, `automation/`, and top-level scripts
  - Analysis and plotting helpers plus regression tests in `tests/`

## Important paths
- `/home/runner/work/VASP/VASP/goad_v1/` — core package for analysis, GA, GUI, and utilities
- `/home/runner/work/VASP/VASP/workflow/` — task generation and execution helpers
- `/home/runner/work/VASP/VASP/automation/` — automation runner, analysis, and config
- `/home/runner/work/VASP/VASP/tests/` — pytest suite
- `/home/runner/work/VASP/VASP/run_goad_v1.py` — main application entry point

## Environment and dependencies
- Python 3.8+ is expected by the docs.
- Install dependencies with:
  - `pip install -r requirements.txt`
- Notes from repo docs:
  - `mattersim` is optional in some workflows
  - `rdkit` may need conda installation: `conda install -c conda-forge rdkit`

## Common commands
- Run the GUI application:
  - `python3 run_goad_v1.py`
- Run the test suite:
  - `pytest`
- Run a specific test file:
  - `pytest tests/test_find_missing_tasks.py`
- Basic syntax validation used in repo docs:
  - `python3 -m py_compile goad_v1/**/*.py`

## Development guidance
- Make focused changes and keep existing script-oriented structure intact.
- Prefer updating existing utilities and tests rather than introducing new frameworks.
- Avoid adding dependencies unless required.
- Preserve compatibility with current data/workflow files in `inputs/`, `runs/`, and the workflow scripts.

## Testing expectations
- For documentation-only changes, no functional tests are usually needed.
- For Python code changes, run the smallest relevant `pytest` scope first, then broaden if needed.
- If touching core package behavior, run `pytest` for the affected tests and ensure no obvious import/syntax regressions.

## Notes for future agents
- There is no existing `AGENTS.md`; this file is the repository-specific starting point.
- The repository mixes end-user docs, GUI code, scientific workflow scripts, and tests, so verify the target area before editing.
