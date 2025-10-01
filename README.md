LangExtract Service (uv-managed)

This repo provides:
- A FastAPI service that adapts LangExtract for Ollama models (`langextract_csv_doc.py`).
- A simple CSV client to batch-extract via the API (`send_csv.py`).

It’s configured to use `uv` for dependency management and virtualenvs.

**Prerequisites**
- Install `uv`: https://docs.astral.sh/uv/
- Python 3.10+ (uv can auto-manage Python if needed).
- Optionally, an Ollama server (`OLLAMA_URL` or `OLLAMA_HOST`).

**Setup**
```
uv sync
```
Creates `.venv` and installs dependencies from `pyproject.toml`/`uv.lock`.

If the `langextract` package is not on PyPI for you, configure its source:
```
[tool.uv.sources]
# Use either a local path or a Git URL
langextract = { path = "../langextract", editable = true }
# or
# langextract = { git = "https://github.com/your-org/langextract.git", rev = "main" }
```
Then run `uv sync` again.

**Run the API**
```
uv run uvicorn langextract_csv_doc:app --reload --host 0.0.0.0 --port 8000
```

Environment variables:
- `OLLAMA_URL` or `OLLAMA_HOST` (default `http://localhost:11434`)
- `OLLAMA_MODEL_ID` (default `gemma3:4b`)
- `OLLAMA_TEMPERATURE` (default `0`)
- `SAVE_ARTIFACTS` (default `1`) — saves JSONL and HTML under `outputs/`
- `PUBLIC_BASE_URL` — base URL if behind a reverse proxy
- `INSTRUCTION_CSV_PATH` — path to your instruction CSV

**Use the CSV client**
With the API running on `http://localhost:8000`:
```
uv run python send_csv.py multi_doc_contracts.csv
```
- Adjust `DOC_ID_COLUMN` in `send_csv.py` if your ID column differs.
- Results are saved under `results/` (per-doc JSON and consolidated file).

**Notes**
- Artifacts (JSONL + HTML) are under `outputs/`. The service serves HTML reports at `/artifacts/<name>.html` if `SAVE_ARTIFACTS=1`.
- To pin Python explicitly, create `.python-version` (e.g., `3.11`) and run `uv python install`.
# vigilant-octo-broccoli
# vigilant-octo-broccoli
