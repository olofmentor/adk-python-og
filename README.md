# Statistical Analysis Agent Service

A FastAPI-based service that accepts a natural-language prompt describing a statistical analysis to perform on a configured CSV dataset, executes the analysis, and writes a structured result into a SQLite database table.

## Features
- Natural language prompt to analysis mapping (basic patterns + safe eval of expressions)
- Pandas-based statistics and plotting (summaries, correlations, regressions)
- Results persisted to SQLite `results` table
- Simple REST API

## Quickstart
1. Create a virtual environment and install dependencies:
   ```bash
   uv venv && source .venv/bin/activate || python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Place your dataset CSV at `data/dataset.csv` (or set `DATASET_PATH`).
3. Run the server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## API
- `POST /analyze`:
  - Body:
    ```json
    { "prompt": "compute mean of sepal_length grouped by species" }
    ```
  - Response: `{ "result_id": 1 }`

- `GET /results/{id}`: fetch a single result row
- `GET /results`: list recent results

## Configuration
- `DATASET_PATH`: path to CSV (default `data/dataset.csv`)
- `DATABASE_URL`: SQLite URL (default `sqlite:///data/results.db`)

## Notes
- The NL-to-analysis layer uses simple pattern/rule-based parsing with guarded evaluation. For complex use, extend `app/agent.py`.
