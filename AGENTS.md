# Repository Guidelines

## Project Structure & Module Organization

This is a Python `src`-layout package for RAG document parsing and indexing. Core code lives in `src/rag_system/`: `parsing/` holds parser backends, chunking, caching, models, and the parser CLI; `indexing/` holds embedding clients, vector stores, manifests, and indexing services; `web/` contains the local FastAPI app; `settings.py` manages DashScope and embedding configuration. Tests are in `tests/`, with heavier checks in `tests/integration/`. Design notes are in `docs/`, sample documents in `samples/`, and local runtime state in `data/`.

## Build, Test, and Development Commands

- `.\.venv\Scripts\python.exe -m pip install -e ".[test,web,indexing,pdf,ocr,html]"`: install the package in editable mode with common development extras.
- `.\.venv\Scripts\python.exe -m pip install -e ".[heavy]"`: add Docling and Unstructured parser backends when needed.
- `.\.venv\Scripts\python.exe -m pytest`: run the configured pytest suite under `tests/`.
- `.\.venv\Scripts\python.exe -m rag_system.parsing --backend auto --document samples\web_cases`: parse the sample corpus and print structured JSON.
- `.\.venv\Scripts\python.exe -m rag_system.web`: start the local web UI/API on `127.0.0.1:8000`.

- `./scripts/run_linux.sh web`: on Linux, reuse the available compatible environment and start the local web UI/API.
- `./scripts/run_linux.sh doctor`: verify dependencies, local models, configuration, and sample data without sending API requests.
- `./scripts/install_all_linux.sh`: create the project-local Linux environment and install every declared optional dependency.
- `./scripts/start_linux.sh`: start the Linux web service in the background and keep its PID and log under `data/runtime/`.
- `./scripts/stop_linux.sh`: stop the background Linux web service.

## Coding Style & Naming Conventions

Use Python 3.10+ syntax, four-space indentation, and type hints for public APIs. Existing modules favor `from __future__ import annotations`, `pathlib.Path`, dataclasses, and dependency-injected service classes. Name modules and functions in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Use explicit UTF-8 for text I/O.

## Testing Guidelines

Pytest is configured in `pyproject.toml` with `pythonpath = ["src"]` and `testpaths = ["tests"]`. Add tests as `test_*.py`; keep fast unit tests in `tests/` and heavier parser or optional-backend coverage in `tests/integration/`. Prefer `tmp_path`, `monkeypatch`, and in-memory test doubles over shared `data/`. No coverage threshold is configured, so cover new parser branches, web endpoints, settings paths, and indexing edge cases directly.

## Commit & Pull Request Guidelines

This checkout has no local Git history, so no project-specific convention can be inferred. Use short imperative subjects such as `Add parser cache diagnostics` and explain behavior changes in the body. Pull requests should describe the changed parser/indexing/web path, list commands run, call out optional dependencies, and link issues or sample files. Include screenshots only for visible web UI changes.

## Security & Configuration Tips

Do not commit real API keys, generated indexes, parser caches, or uploads from `data/`. Prefer `DASHSCOPE_API_KEY` or the local settings UI; saved keys live in `data/settings.json` and should remain machine-local.
