import asyncio

import pytest

from app import llm

VALID = ('{"single_product": true, "clean_packshot": true, "product_type": "earrings", '
         '"worn_on": "ear", "sits_in_room": false, "confidence": 0.92}')


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(llm, "LLM_BACKOFF_SECONDS", 0)


def fake_gemini(replies, calls):
    async def _call(image_bytes, mime_type, feedback):
        calls.append(feedback)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    return _call


def test_resends_with_feedback_after_invalid_output(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_call_gemini", fake_gemini(['{"worn_on": "hand"}', VALID], calls))
    facts = asyncio.run(llm.extract_facts(b"img", "image/jpeg"))
    assert facts.worn_on == "ear"
    assert calls[0] is None and "failed validation" not in (calls[0] or "")
    assert "worn_on" in calls[1]  # validation error fed back on the resend


def test_recovers_from_api_exception(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_call_gemini", fake_gemini([TimeoutError(), VALID], calls))
    assert asyncio.run(llm.extract_facts(b"img", "image/jpeg")).product_type == "earrings"


def test_raises_after_max_attempts(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_call_gemini", fake_gemini(["not json"] * 3, calls))
    with pytest.raises(llm.LLMError):
        asyncio.run(llm.extract_facts(b"img", "image/jpeg"))
    assert len(calls) == llm.LLM_MAX_ATTEMPTS
