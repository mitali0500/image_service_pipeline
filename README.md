# Image Intake Service

FastAPI service that validates a product image, runs code-based quality checks, asks Gemini
for image facts, and lets plain code decide the outcome.

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # set GEMINI_API_KEY
.venv/bin/uvicorn app.main:app --reload
.venv/bin/python -m pytest -q
```

## Endpoints
- `GET /health`
- `POST /intake` — multipart `file` (jpg/jpeg/png, ≤10MB)
- `GET /logs?limit=200&request_id=<id>&level=ERROR`

## Pipeline
1. Validate upload (extension, content type, real JPEG/PNG bytes via PIL) → store in SQLite (`data/intake.db`).
2. Parallel code checks: shorter side ≥ 1000px; Laplacian variance ≥ 50.
3. If both pass → Gemini (`gemini-3.8-flash`) returns facts; validated with Pydantic `ImageFacts`,
   resent with the validation error up to 3 attempts.
4. `decide()` (pure function, see `tests/test_decide.py`) → `pass | fail | needs_review` + experience.

Logs: `logs/app.log`, each line tagged with `request_id`.
