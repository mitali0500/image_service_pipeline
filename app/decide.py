"""Pure decision logic: no I/O, no LLM. The LLM reports facts; this code decides."""

from app.config import MIN_CONFIDENCE
from app.schemas import CheckResult, Decision, ImageFacts


def decide(checks: list[CheckResult], facts: ImageFacts | None, llm_failed: bool = False) -> Decision:
    # 1. Code checks failed -> fail with low_resolution and/or blurry
    check_reasons = [c.reason for c in checks if not c.passed and c.reason]
    if check_reasons:
        return Decision(status="fail", reasons=check_reasons)

    # 2. LLM call failed or output invalid -> needs_review
    if llm_failed or facts is None:
        return Decision(status="needs_review", reasons=["llm_error"])

    # 3. Content rules
    content_reasons = []
    if not facts.single_product:
        content_reasons.append("not_single_product")
    if not facts.clean_packshot:
        content_reasons.append("not_packshot")
    if content_reasons:
        return Decision(status="fail", reasons=content_reasons)

    # 4. Low confidence -> needs_review
    if facts.confidence < MIN_CONFIDENCE:
        return Decision(status="needs_review", reasons=["low_confidence"])

    # 5. Pass, pick the experience
    if facts.worn_on != "none":
        experience = "virtual_tryon"
    elif facts.sits_in_room:
        experience = "ar_placement"
    else:
        experience = "3d_viewer"
    return Decision(status="pass", experience=experience)
