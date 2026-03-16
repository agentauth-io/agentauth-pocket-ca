from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from pocket_ca.formatting import build_prompt as shared_build_prompt
from pocket_ca.formatting import parse_response as shared_parse_response


class DecisionRequest(BaseModel):
    instruction: str = Field(..., description="Task instruction for the model.")
    context: dict[str, Any] = Field(
        ..., description="Structured financial context for the decision."
    )


class DecisionResponse(BaseModel):
    decision: Literal["approve", "reject", "review"]
    explanation: str
    confidence: float


def build_prompt(instruction: str, context: dict[str, Any]) -> str:
    return shared_build_prompt(instruction, context)


def parse_response(response_text: str) -> dict[str, Any]:
    parsed = shared_parse_response(response_text)
    parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
    return parsed
