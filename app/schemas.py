from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

WornOn = Literal["ear", "neck", "wrist", "finger", "face", "none"]
Status = Literal["pass", "fail", "needs_review"]
Experience = Literal["virtual_tryon", "ar_placement", "3d_viewer"]


class ImageFacts(BaseModel):
    """Contract for the vision LLM output. Facts only, no decisions."""

    model_config = ConfigDict(extra="forbid")

    single_product: bool = Field(description="Exactly one product in the image")
    clean_packshot: bool = Field(description="Plain background, no text or watermark, not worn")
    product_type: str = Field(min_length=1, description='e.g. "earrings", "sofa"')
    worn_on: WornOn = Field(description="Body part the product is worn on, or none")
    sits_in_room: bool = Field(description="Furniture or decor that sits in a room")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in these facts, 0 to 1")


class CheckResult(BaseModel):
    name: Literal["resolution", "sharpness"]
    passed: bool
    score: float
    reason: str | None = None  # "low_resolution" / "blurry" on failure
    detail: str | None = None


class Scores(BaseModel):
    resolution_score: float | None = None
    sharpness_score: float | None = None


class Decision(BaseModel):
    status: Status
    reasons: list[str] | None = None
    experience: Experience | None = None


class IntakeResponse(BaseModel):
    request_id: str
    status: Status
    reasons: list[str] | None
    score: Scores
    facts: ImageFacts | None
    experience: Experience | None
    created_at: str
