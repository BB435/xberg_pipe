# Repository Guidelines

## Project Structure & Processing Flow

Source code lives in `src/xberg_pipe/`. `cli.py` defines commands,
`scan_extractor.py` saves extracted text, and `models.py` contains the schema.
Post-processing is separate: `chunking.py` creates Japanese-aware chunks, `keywording.py` runs
KeyBERT, `summarization.py` calls an OpenAI-compatible local LLM, and
`vector_store.py` manages sqlite-vec embeddings and search. Database setup and
queries belong in `db.py` and `repository.py`. Tests live under `tests/`.
Runtime databases and downloaded models must not be committed.

The expected pipeline is `scan` → `chunk` → optional `keywords`, `summarize`,
and `embed` → `search`. Keep extraction independent from expensive downstream
processing. When extracted text changes, derived chunks, keywords, summaries,
and embedding metadata must be invalidated.

## Build, Test, and Development Commands

This project requires Python 3.14 and uses `uv`.

- `uv sync --group dev` installs core and development dependencies.
- `uv sync --extra all --group dev` also installs KeyBERT and sqlite-vec support.
- `uv run xberg-pipe scan <directory>` extracts text into `data/app.db`.
- `uv run xberg-pipe chunk` generates chunks as a separate step.
- `uv run pytest -q` runs the test suite.
- `uv run ruff check src tests` checks lint rules.
- `uv run ruff format --check src tests` verifies formatting.
- `uv build` creates distributions in `dist/`.

Use `--database <path>` before the subcommand when testing a non-default DB.

## Coding Style & Naming Conventions

Use four spaces, type annotations, `pathlib.Path`, and typed dataclasses.
Follow `snake_case` for functions and
modules, `PascalCase` for classes, and `UPPER_CASE` for constants. Keep heavy ML
imports lazy so basic CLI commands work without optional extras. Preserve the
existing concise Japanese docstring and user-message style. Ruff configuration
lives in `pyproject.toml`.

## Testing Guidelines

Write pytest files as `tests/test_<module>.py` and functions as
`test_<behavior>`. Use `tmp_path`, in-memory SQLite, and fake LLM/embedding
implementations; never write tests to `data/app.db` or download production
models. Mark optional-extension tests with `pytest.importorskip`. Validate both
stored rows and invalidation behavior. Run sqlite-vec integration separately
when needed: `uv run --with sqlite-vec pytest -q tests/test_vector_store.py`.

## Commits & Pull Requests

History uses short Conventional Commit-style subjects such as `feat: add cli
tool` and `chore: add pytest dependency`. Use `feat:`, `fix:`, `test:`, or
`chore:` with one focused change per commit. Pull requests should describe the
pipeline impact, schema or dependency changes, verification commands, and any
model download or migration requirements. Link relevant issues.
