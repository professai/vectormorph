# VectorMorph

A lightweight vector database: a FastAPI server (`vectormorph/vector_morph.py`)
backed by hnswlib, plus a Python client (`vectormorph/client.py`).

## Git workflow

- Push work to `main` (owner preference: no long-lived divergent branches).
  If a session requires developing on a feature branch, fast-forward `main`
  to it before finishing so the branches never diverge.

## Development

- Install: `pip install -e '.[dev]'`
- Tests: `pytest` (suite in `tests/`)
- Lint: `ruff check .` — must pass; CI runs it on every push
- Run the server locally: `BEARER_TOKEN=<token> vectormorph` (port 4440)

## Conventions

- Packaging lives in `pyproject.toml` (no setup.py); bump `version` there and
  `__version__` in `vectormorph/vector_morph.py` together.
- All endpoints except `/health/` require bearer-token auth.
- The database dimensionality is fixed by the first vector added.
