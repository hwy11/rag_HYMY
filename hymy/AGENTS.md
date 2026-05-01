# Repository Guidelines

## Project Structure & Module Organization

This repository now has two layers: a local ZSXQ crawler and the existing post-processing pipeline. Core logic lives in `hymy/`: `pipeline.py` handles markdown/json cleanup, `zsxq_client.py` calls the Knowledge Planet API, `zsxq_service.py` manages incremental crawl state, `zsxq_pipeline.py` converts raw topics into the repository JSON/Markdown format, and `paths.py` / `storage.py` centralize filesystem paths. `app.py` exposes the local FastAPI web console, with UI files under `app/templates` and `app/static`. Raw crawler output is stored in `data/raw/`; generated artifacts are stored in `output/`.

## Build, Test, and Development Commands

Use a local virtual environment and install dependencies first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the local web console:

```bash
python run_web.py
```

Open `http://127.0.0.1:8011` and paste your `Authorization`, `User-Agent`, and group info. For the legacy pipeline, run each stage from the repository root:

```bash
python process_kb.py
python enrich_data.py
python reconstruct_kb.py
python verify_kb.py
```

`process_kb.py` extracts structured entries from existing markdown, `enrich_data.py` adds summaries and keywords, `reconstruct_kb.py` rebuilds markdown into `output/`, and `verify_kb.py` prints entry and answer counts. `python -m compileall app.py hymy` is the quickest syntax sanity check before commits.

## Coding Style & Naming Conventions

Follow PEP 8 with 4-space indentation and descriptive snake_case names. Keep crawler, state, and export logic inside `hymy/` modules; keep root scripts thin. Prefer `pathlib.Path` over hard-coded paths, and route persistent files through `paths.py` or `storage.py`. When changing API behavior, keep Knowledge Planet fetching isolated from export formatting.

## Testing Guidelines

There is no formal test suite yet, so verify changes by running the web app locally and checking both `data/raw/` and `output/`. For pipeline-only changes, treat `verify_kb.py` as the minimum regression check. If you add tests, use `pytest`, place them under `tests/`, and name files `test_<module>.py`.

## Commit & Pull Request Guidelines

The Git history is minimal, so use short, imperative commit messages such as `add zsxq local console` or `refine incremental crawl state`. Pull requests should state whether they affect crawling, state persistence, or output formatting, and include sample file paths when generated artifacts change.

## Security & Configuration Tips

Do not commit real `Authorization` headers, `User-Agent` values, or `.env` secrets. Keep `.env.example` updated when new configuration is introduced. Avoid embedding machine-specific absolute paths, and never store crawler state outside `data/` unless the change is intentional.
