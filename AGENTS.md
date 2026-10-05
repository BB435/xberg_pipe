# Repository Guidelines

## Structure and Pipeline

Source code lives in `src/xberg_pipe/`; tests live in `tests/`. `cli.py` defines
the commands, `scan_extractor.py` handles file discovery and extraction,
`models.py` defines the SQLite schema, and `db.py` and `repository.py` manage
database setup and shared persistence. `chunking.py` creates Japanese-aware
chunks, `keywording.py` provides lightweight YAKE and optional KeyBERT
keywords, `provisional_summary.py` creates extractive summaries,
`summarization.py` creates Ollama summaries, `vector_store.py` stores and
searches sqlite-vec embeddings, and `server.py` serves the search UI and API.

`scan` extracts changed files and saves chunks, lightweight keywords, and
provisional summaries in the same workflow. The expensive steps are separate:
`refine-keywords`, `summarize`, and `embed`; then `search` or `serve` queries the
results. Those three commands process only items missing the selected result by
default; `--all` rebuilds them. Keep the basic scan usable without optional ML,
LLM, vector, or server dependencies. When extracted text changes, invalidate
its chunks, keywords, summaries, and embedding metadata and vectors together.
Do not commit runtime databases, downloaded models, or generated distributions.

## Development Commands

Python 3.14 and `uv` are required. Put `--database <path>` before the
subcommand when using a database other than `data/app.db`.

- `uv sync --group dev`: core and development dependencies.
- `uv sync --extra all --group dev`: all optional features. Individual extras
  are `keywords`, `vectors`, `llm`, and `vis`.
- `uv run xberg-pipe scan <directory>`: extract and save lightweight results.
- `uv run xberg-pipe refine-keywords`, `uv run xberg-pipe summarize`, and
  `uv run xberg-pipe embed`: run optional downstream work. Add `--all` to
  regenerate existing results.
- `uv run xberg-pipe search <query>` and `uv run xberg-pipe serve`: query the
  stored results. `init-db` and `stats` are available for database maintenance.
- `uv run pytest -q`: run tests.
- `uv run ruff check src tests` and `uv run ruff format --check src tests`:
  check lint and formatting.
- `uv build`: build distributions in `dist/`.

## Coding and Testing

Use four spaces, type annotations, `pathlib.Path`, and typed dataclasses.
Use `snake_case` for functions and modules, `PascalCase` for classes, and
`UPPER_CASE` for constants. Keep heavy optional imports lazy. Preserve the
concise Japanese docstring and user-message style. Ruff settings are in
`pyproject.toml`.

Name pytest files `tests/test_<module>.py` and tests `test_<behavior>`. Use
`tmp_path`, in-memory SQLite where appropriate, and fake LLM and embedding
implementations; never write tests to `data/app.db` or download production
models. Use `pytest.importorskip` for optional extensions. Check stored rows,
invalidation on changed text, incremental reruns, and `--all` behavior when
changing downstream processing. Run sqlite-vec integration separately when
needed: `uv run --with sqlite-vec pytest -q tests/test_vector_store.py`.

## Commits and Pull Requests

Use short, focused Conventional Commit subjects such as `feat: add search
filter` or `fix: invalidate stale vectors`. Pull requests should describe
pipeline effects, schema or dependency changes, verification commands, and
any model download or migration requirements. Link relevant issues.
