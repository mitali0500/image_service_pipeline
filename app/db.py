import json
import sqlite3
from contextlib import contextmanager

from app.config import DB_PATH
from app.logging_setup import get_logger

log = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS intake (
    request_id   TEXT PRIMARY KEY,
    image        BLOB NOT NULL,
    filename     TEXT,
    content_type TEXT,
    created_at   TEXT NOT NULL,
    result_json  TEXT
)
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(SCHEMA)
    log.info("SQLite ready at %s", DB_PATH)


def insert_image(request_id: str, image: bytes, filename: str, content_type: str, created_at: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO intake (request_id, image, filename, content_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (request_id, image, filename, content_type, created_at),
        )
    log.info("Stored image (%d bytes, %s)", len(image), filename)


def save_result(request_id: str, result: dict) -> None:
    with _connect() as conn:
        conn.execute("UPDATE intake SET result_json = ? WHERE request_id = ?", (json.dumps(result), request_id))
    log.info("Stored final result")
