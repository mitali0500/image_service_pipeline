import io
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from app import db
from app.checks import run_checks
from app.config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    ALLOWED_PIL_FORMATS,
    LOG_PATH,
    MAX_UPLOAD_BYTES,
)
from app.decide import decide
from app.llm import LLMError, extract_facts
from app.logging_setup import get_logger, request_id_var, setup_logging
from app.schemas import IntakeResponse, Scores

setup_logging()
log = get_logger("api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    log.info("Service started")
    yield


app = FastAPI(title="Image Intake Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def validate_upload(file: UploadFile, data: bytes) -> str:
    """Basic JPG/PNG validation. Returns the real MIME type detected from the bytes."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file extension '{ext}'. Allowed: jpg, jpeg, png")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported content type '{file.content_type}'. Allowed: image/jpeg, image/png")
    if not data:
        raise HTTPException(400, "Uploaded file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large ({len(data)} bytes). Max {MAX_UPLOAD_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format
            img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(400, "File is not a valid JPEG/PNG image")
    if fmt not in ALLOWED_PIL_FORMATS:
        raise HTTPException(400, f"File content is {fmt}, not JPEG/PNG")
    return "image/jpeg" if fmt == "JPEG" else "image/png"


@app.post("/intake", response_model=IntakeResponse)
async def intake(file: UploadFile = File(...)) -> IntakeResponse:
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    started = time.perf_counter()
    log.info("Intake received: filename=%s content_type=%s", file.filename, file.content_type)

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    try:
        mime_type = validate_upload(file, data)
    except HTTPException as exc:
        log.warning("Upload rejected: %s", exc.detail)
        raise

    created_at = datetime.now(timezone.utc).isoformat()
    db.insert_image(request_id, data, file.filename or "", mime_type, created_at)

    # Code checks (parallel)
    checks = await run_checks(data)
    by_name = {c.name: c for c in checks}
    scores = Scores(resolution_score=by_name["resolution"].score, sharpness_score=by_name["sharpness"].score)

    # LLM layer, only when the code checks pass
    facts = None
    llm_failed = False
    if all(c.passed for c in checks):
        log.info("Code checks passed, calling LLM")
        try:
            facts = await extract_facts(data, mime_type)
        except LLMError as exc:
            llm_failed = True
            log.error("LLM failed: %s", exc)
    else:
        log.info("Code checks failed (%s), skipping LLM", [c.reason for c in checks if not c.passed])

    decision = decide(checks, facts, llm_failed)
    response = IntakeResponse(
        request_id=request_id,
        status=decision.status,
        reasons=decision.reasons,
        score=scores,
        facts=facts,
        experience=decision.experience,
        created_at=created_at,
    )
    db.save_result(request_id, response.model_dump())
    log.info(
        "Decision: status=%s reasons=%s experience=%s (%.2fs total)",
        decision.status, decision.reasons, decision.experience, time.perf_counter() - started,
    )
    return response


def _parse_line(line: str) -> dict:
    parts = line.rstrip("\n").split(" | ", 4)
    if len(parts) != 5:
        return {"raw": line.rstrip("\n")}
    ts, level, rid, logger, message = parts
    return {"timestamp": ts, "level": level, "request_id": rid, "logger": logger, "message": message}


@app.get("/logs")
def get_logs(
    limit: int = Query(200, ge=1, le=5000),
    request_id: str | None = None,
    level: str | None = Query(None, description="INFO, WARNING, ERROR"),
) -> dict:
    if not LOG_PATH.exists():
        return {"count": 0, "logs": []}
    matched: deque[dict] = deque(maxlen=limit)
    with LOG_PATH.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            entry = _parse_line(line)
            if request_id and entry.get("request_id") != request_id:
                continue
            if level and entry.get("level") != level.upper():
                continue
            matched.append(entry)
    return {"count": len(matched), "logs": list(matched)}
