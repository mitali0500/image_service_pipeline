import asyncio
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from app.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_BACKOFF_SECONDS,
    LLM_MAX_ATTEMPTS,
    LLM_TIMEOUT_SECONDS,
)
from app.logging_setup import get_logger
from app.schemas import ImageFacts

log = get_logger("llm")

PROMPT = """You are a product image inspector. Look at the image and report FACTS only.
Do not judge whether the image is acceptable. Reply with a single JSON object and nothing else:

{
  "single_product": bool,   // true only if exactly one product is shown
  "clean_packshot": bool,   // plain background, no text or watermark, product not being worn
  "product_type": string,   // short lowercase noun, e.g. "earrings", "sofa", "watch"
  "worn_on": "ear" | "neck" | "wrist" | "finger" | "face" | "feet" | "none",  // where the product is worn; "none" if not wearable
  "sits_in_room": bool,     // true if it is furniture or decor
  "confidence": number      // 0 to 1, how confident you are in these facts
}"""


class LLMError(Exception):
    pass


_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


async def _call_gemini(image_bytes: bytes, mime_type: str, feedback: str | None) -> str:
    prompt = PROMPT
    if feedback:
        prompt += (
            "\n\nYour previous reply failed validation:\n"
            f"{feedback}\n"
            "Return ONLY a corrected JSON object that matches the structure exactly."
        )
    response = await asyncio.wait_for(
        _get_client().aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
        ),
        timeout=LLM_TIMEOUT_SECONDS,
    )
    return response.text or ""


async def extract_facts(image_bytes: bytes, mime_type: str) -> ImageFacts:
    """Ask Gemini for image facts, validate with Pydantic, resend with the error on failure."""
    feedback: str | None = None
    last_error = "unknown error"

    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            raw = await _call_gemini(image_bytes, mime_type, feedback)
        except LLMError:
            raise
        except genai_errors.ClientError as exc:
            # 4xx (bad model, bad key, bad request) won't fix itself on retry; 429 rate limit will
            last_error = f"{type(exc).__name__}: {exc}"
            if exc.code != 429:
                log.error("LLM attempt %d/%d non-retryable error: %s", attempt, LLM_MAX_ATTEMPTS, last_error)
                raise LLMError(last_error) from exc
            log.warning("LLM attempt %d/%d rate limited: %s", attempt, LLM_MAX_ATTEMPTS, last_error)
        except Exception as exc:  # network, 5xx, timeout, SDK errors
            last_error = f"{type(exc).__name__}: {exc}"
            log.warning("LLM attempt %d/%d call failed: %s", attempt, LLM_MAX_ATTEMPTS, last_error)
        else:
            elapsed = time.perf_counter() - started
            log.info("LLM attempt %d/%d replied in %.2fs: %s", attempt, LLM_MAX_ATTEMPTS, elapsed, raw[:500])
            try:
                facts = ImageFacts.model_validate_json(raw)
                log.info("LLM output valid on attempt %d", attempt)
                return facts
            except ValidationError as exc:
                last_error = str(exc)
                feedback = f"Reply: {raw[:1000]}\nErrors: {exc}"
                log.warning("LLM attempt %d/%d invalid output: %s", attempt, LLM_MAX_ATTEMPTS, exc.errors())

        if attempt < LLM_MAX_ATTEMPTS:
            await asyncio.sleep(LLM_BACKOFF_SECONDS * attempt)

    raise LLMError(f"No valid output after {LLM_MAX_ATTEMPTS} attempts: {last_error}")
