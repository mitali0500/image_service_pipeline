import contextvars
import logging
from logging.handlers import RotatingFileHandler

from app.config import LOG_PATH

# Set per request so every log line carries the request_id without passing it around.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(request_id)s | %(name)s | %(message)s"


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("intake")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3)
    console_handler = logging.StreamHandler()
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        handler.addFilter(RequestIdFilter())
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"intake.{name}")
