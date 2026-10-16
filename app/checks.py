import asyncio
import io

import cv2
import numpy as np
from PIL import Image

from app.config import MIN_SHARPNESS, MIN_SHORT_SIDE_PX
from app.logging_setup import get_logger
from app.schemas import CheckResult

log = get_logger("checks")


def check_resolution(image_bytes: bytes) -> CheckResult:
    with Image.open(io.BytesIO(image_bytes)) as img:
        width, height = img.size
    short_side = min(width, height)
    passed = short_side >= MIN_SHORT_SIDE_PX
    log.info("Resolution %dx%d, short side %d (min %d) -> %s", width, height, short_side, MIN_SHORT_SIDE_PX, passed)
    return CheckResult(
        name="resolution",
        passed=passed,
        score=short_side,
        reason=None if passed else "low_resolution",
        detail=f"{width}x{height}px; shorter side must be at least {MIN_SHORT_SIDE_PX}px",
    )


def check_sharpness(image_bytes: bytes) -> CheckResult:
    gray = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("OpenCV could not decode the image")
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    passed = score >= MIN_SHARPNESS
    log.info("Sharpness (Laplacian variance) %.2f (min %.1f) -> %s", score, MIN_SHARPNESS, passed)
    return CheckResult(
        name="sharpness",
        passed=passed,
        score=round(score, 2),
        reason=None if passed else "blurry",
        detail=f"Laplacian variance must be at least {MIN_SHARPNESS}",
    )


async def run_checks(image_bytes: bytes) -> list[CheckResult]:
    """Run both checks in parallel; both always run so every failure reason is reported."""
    return list(
        await asyncio.gather(
            asyncio.to_thread(check_resolution, image_bytes),
            asyncio.to_thread(check_sharpness, image_bytes),
        )
    )
