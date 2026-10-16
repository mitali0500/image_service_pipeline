import pytest

from app.decide import decide
from app.schemas import CheckResult, ImageFacts

RES_OK = CheckResult(name="resolution", passed=True, score=1200)
RES_BAD = CheckResult(name="resolution", passed=False, score=600, reason="low_resolution")
SHARP_OK = CheckResult(name="sharpness", passed=True, score=120.0)
SHARP_BAD = CheckResult(name="sharpness", passed=False, score=12.0, reason="blurry")


def facts(**overrides) -> ImageFacts:
    base = dict(
        single_product=True,
        clean_packshot=True,
        product_type="earrings",
        worn_on="none",
        sits_in_room=False,
        confidence=0.9,
    )
    return ImageFacts(**(base | overrides))


@pytest.mark.parametrize(
    "checks, f, llm_failed, status, reasons, experience",
    [
        ([RES_BAD, SHARP_OK], None, False, "fail", ["low_resolution"], None),
        ([RES_OK, SHARP_BAD], None, False, "fail", ["blurry"], None),
        ([RES_BAD, SHARP_BAD], None, False, "fail", ["low_resolution", "blurry"], None),
        ([RES_OK, SHARP_OK], None, True, "needs_review", ["llm_error"], None),
        ([RES_OK, SHARP_OK], facts(single_product=False), False, "fail", ["not_single_product"], None),
        ([RES_OK, SHARP_OK], facts(clean_packshot=False), False, "fail", ["not_packshot"], None),
        ([RES_OK, SHARP_OK], facts(single_product=False, clean_packshot=False), False, "fail",
         ["not_single_product", "not_packshot"], None),
        ([RES_OK, SHARP_OK], facts(confidence=0.59), False, "needs_review", ["low_confidence"], None),
        ([RES_OK, SHARP_OK], facts(confidence=0.6), False, "pass", None, "3d_viewer"),
        ([RES_OK, SHARP_OK], facts(worn_on="ear"), False, "pass", None, "virtual_tryon"),
        ([RES_OK, SHARP_OK], facts(worn_on="wrist", sits_in_room=True), False, "pass", None, "virtual_tryon"),
        ([RES_OK, SHARP_OK], facts(product_type="sofa", sits_in_room=True), False, "pass", None, "ar_placement"),
        ([RES_OK, SHARP_OK], facts(product_type="mug"), False, "pass", None, "3d_viewer"),
    ],
)
def test_decide(checks, f, llm_failed, status, reasons, experience):
    d = decide(checks, f, llm_failed)
    assert (d.status, d.reasons, d.experience) == (status, reasons, experience)


def test_code_fail_beats_everything():
    d = decide([RES_BAD, SHARP_OK], facts(single_product=False), llm_failed=True)
    assert d.status == "fail" and d.reasons == ["low_resolution"]


def test_schema_rejects_bad_llm_output():
    from pydantic import ValidationError

    for bad in [
        '{"single_product": true}',
        '{"single_product": true, "clean_packshot": true, "product_type": "x", "worn_on": "hand", '
        '"sits_in_room": false, "confidence": 0.9}',
        '{"single_product": true, "clean_packshot": true, "product_type": "x", "worn_on": "none", '
        '"sits_in_room": false, "confidence": 1.5}',
    ]:
        with pytest.raises(ValidationError):
            ImageFacts.model_validate_json(bad)
