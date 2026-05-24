# Repository Guidelines

## Project Structure & Module Organization
`verl/` contains the core library: trainers, rollout workers, model merger utilities, dataset helpers, and distributed/runtime support. `recipe/` holds runnable task pipelines and examples such as KL training, PPO, QAT, and reward-model workflows. `tests/` mirrors the package layout with CPU-focused unit tests and recipe-level regression tests. `docs/` contains Sphinx documentation sources, and top-level `requirements*.txt` files define environment variants.

## Build, Test, and Development Commands
- `pip install -e .` installs the repository in editable mode.
- `pip install -e .[test]` or `pip install -r requirements-test.txt` installs pytest and pre-commit tooling.
- `pytest tests/utils/test_tokenizer_normalize_on_cpu.py -q` runs a targeted test file.
- `pytest tests -q` runs the broader test suite.
- `python -m py_compile path/to/file.py` is a fast syntax check for edited modules.
- `bash recipe/kl_training/run_kl_training.sh` runs the KL training recipe end to end.
- `make html -C docs` builds the documentation site from `docs/`.

## Coding Style & Naming Conventions
Use 4-space indentation and follow standard Python style. Prefer clear, explicit names: `snake_case` for functions, variables, and modules; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants. Try your best not to modify essential core code under `verl/`; prefer limiting changes to `recipe/...` whenever the task can be solved there. Only change lower-level `verl/...` modules when the fix clearly belongs to shared infrastructure. Match existing logging style and keep comments short and technical.

## Testing Guidelines
Pytest is the default test framework. Add tests near the affected area, following the existing naming pattern `test_<feature>*.py`. Prefer focused tests for utilities and CPU-safe regression tests for recipe behavior. When touching distributed, checkpoint, or rollout code, run at least one targeted test plus a syntax check on modified files.

## Commit & Pull Request Guidelines
Recent history uses short prefixes such as `ADD: ...` and scoped conventional messages like `[ckpt] fix: ...` or `[model,doc] feat: ...`. Keep commits narrowly scoped and descriptive. Pull requests should explain the user-facing impact, note any environment assumptions, list validation commands run, and attach logs or screenshots when changing docs, dashboards, or long-running recipes.

## Security & Configuration Tips
Do not commit secrets, tokens, or machine-specific paths. Prefer environment variables for dataset roots, model paths, and W&B settings. Large checkpoints and generated results belong outside git; keep reproducible scripts under `recipe/` and document required GPUs, nodes, and external services in the PR description.

## Slurm Resource Handling
For Slurm ML jobs, do not leave completed allocations idle. A top-level `sbatch` or top-level `srun` should exit normally after successful completion so Slurm releases its resources. If a recovery command is run inside an existing allocation with `srun --jobid=<jobid> --overlap` or a generated `rerun_env.sh`, it should preserve its exit code and release the parent allocation when it exits, normally with `scancel ""`.

Before releasing resources after success, verify the final marker/checkpoint/model exists when feasible. The default rule is: successful completion releases the allocation, and failures should exit nonzero so Slurm also releases the allocation. Do not keep failed jobs alive for debugging.
