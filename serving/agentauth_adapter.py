"""
AgentAuth Adapter -- Pydantic models and thin wrappers for the serving layer
=============================================================================

This module sits between the FastAPI endpoints (``inference_api.py``) and the
core prompt / parsing logic in ``pocket_ca.formatting``.  Its responsibilities:

  * Define strict **Pydantic request/response schemas** so that the API
    contract is enforced and auto-documented by FastAPI's OpenAPI generator.
  * Re-export ``build_prompt`` and ``parse_response`` with a serving-specific
    guard: the raw confidence score returned by the model (or estimated by the
    heuristic in ``pocket_ca.formatting.estimate_confidence``) is clamped to
    the ``[0.0, 1.0]`` range before it reaches the caller.

Decision normalisation
~~~~~~~~~~~~~~~~~~~~~~
The model may output many synonyms (APPROVE, ALLOW, BUY, PASS, ...).  The
shared ``parse_response`` in ``pocket_ca.formatting`` canonicalises them into
exactly three API-level labels: ``approve``, ``reject``, or ``review``.  See
``APPROVE_LABELS`` / ``REJECT_LABELS`` / ``REVIEW_LABELS`` in that module for
the full mapping.  Any unrecognised label falls through to ``review``.

Confidence estimation heuristic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When the model does not emit an explicit ``Confidence:`` line (most cases),
``pocket_ca.formatting.estimate_confidence`` synthesises a score from:
  - whether a decision was extracted at all (+0.14 bonus),
  - explanation length (longer = higher confidence),
  - presence of hedging terms like "maybe" or "possibly" (-0.12 penalty),
  - whether the final decision is ``review`` (slightly lower confidence).
The result is rounded and clamped to ``[0.05, 0.99]`` by that function; this
module additionally clamps to ``[0.0, 1.0]`` as a safety net.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from pocket_ca.formatting import build_prompt as shared_build_prompt
from pocket_ca.formatting import parse_response as shared_parse_response


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    """Incoming payload for all decision endpoints.

    ``instruction`` is the human-readable task (e.g. "Evaluate this wire
    transfer").  ``context`` is an arbitrary JSON object carrying the
    financial data the model needs (amounts, accounts, policy limits, etc.).
    """
    instruction: str = Field(..., description="Task instruction for the model.")
    context: dict[str, Any] = Field(
        ..., description="Structured financial context for the decision."
    )


class DecisionResponse(BaseModel):
    """Outgoing payload returned by all decision endpoints.

    ``decision`` is one of three normalised labels.  ``confidence`` is a
    float in [0, 1] -- either extracted from the model output or estimated
    heuristically.
    """
    decision: Literal["approve", "reject", "review"]
    explanation: str
    confidence: float


# ---------------------------------------------------------------------------
# Thin wrappers around shared formatting utilities
# ---------------------------------------------------------------------------

def build_prompt(instruction: str, context: dict[str, Any]) -> str:
    """Construct a Llama-3 chat-template prompt from instruction + context."""
    return shared_build_prompt(instruction, context)


def parse_response(response_text: str) -> dict[str, Any]:
    """Parse the model's raw text output into a structured dict.

    Delegates to ``pocket_ca.formatting.parse_response`` which extracts the
    ``Decision:`` and ``Reason:`` lines, then clamps the confidence value to
    [0.0, 1.0] as a defensive measure -- the upstream heuristic already caps
    at 0.99, but an explicit ``Confidence:`` line from the model could exceed
    that range.
    """
    parsed = shared_parse_response(response_text)
    # Clamp confidence to a valid probability range.
    parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
    return parsed
