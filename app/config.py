import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Storage
DB_PATH = Path(os.getenv("DB_PATH", BASE_DIR / "data" / "intake.db"))
LOG_PATH = Path(os.getenv("LOG_PATH", BASE_DIR / "logs" / "app.log"))

# Upload validation
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

# Code checks
MIN_SHORT_SIDE_PX = int(os.getenv("MIN_SHORT_SIDE_PX", 1000))
MIN_SHARPNESS = float(os.getenv("MIN_SHARPNESS", 50.0))

# LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", 3))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", 30))
LLM_BACKOFF_SECONDS = float(os.getenv("LLM_BACKOFF_SECONDS", 1.0))

# Decision
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.6))
