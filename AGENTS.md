# Repository Guidelines

## Project Structure & Module Organization

Application code lives under `src/xberg_pipe/`. The package entry point is
`src/xberg_pipe/__init__.py`; `scan_extractor.py` handles recursive file discovery,
`models.py` defines SQLAlchemy entities, `repository.py` contains persistence
queries, and `db.py` configures the SQLite database. Runtime data is created under
`data/` and should not be committed. `pyproject.toml` defines package metadata,
dependencies, and the `xberg-pipe` console command. The root `test.py` is currently
a small exploratory script, not a complete automated test suite.

## Build, Test, and Development Commands

This project uses Python 3.14 and `uv`.

- `uv sync` creates or updates `.venv` from `uv.lock`.
- `uv run xberg-pipe` runs the installed console entry point.
- `uv run python test.py` runs the current smoke script.
- `uv build` produces source and wheel distributions in `dist/`.
- `uvx ruff check .` checks Python style; add `--fix` for safe automatic fixes.
- `uvx ruff format .` formats Python files.

Run commands from the repository root so relative paths such as `data/app.db`
resolve consistently.

## Coding Style & Naming Conventions

Use four-space indentation, modern Python type annotations, and `pathlib.Path`
for filesystem work. Follow standard Python naming: `snake_case` for functions and
modules, `PascalCase` for classes, and `UPPER_CASE` for constants. Keep database
operations in `repository.py`, ORM declarations in `models.py`, and scanning or
extraction logic in focused service modules. Ruff is configured in
`pyproject.toml`; timezone lint (`DTZ`) is intentionally disabled. Existing
docstrings are Japanese, so preserve the surrounding file's language and tone.

## Testing Guidelines

There is no configured test framework or coverage threshold yet. New behavior
should add `pytest` tests under `tests/`, named `test_<module>.py`, with test
functions named `test_<behavior>`. Prefer temporary directories and temporary
SQLite databases; do not let tests modify `data/app.db`. Once pytest is added,
run the suite with `uv run pytest`.

## Commit & Pull Request Guidelines

History currently contains only the short commit `poc phase1`, so no strong
convention is established. Use concise, imperative subjects such as
`Add document scan batching`. Keep commits focused. Pull requests should explain
the behavior change, list verification commands, link relevant issues, and call
out schema or dependency changes. Include screenshots only for user-visible
output.
