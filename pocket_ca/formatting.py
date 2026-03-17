"""Prompt construction, decision normalisation, and response parsing for
the Pocket CA model.

This module implements three core capabilities:

1. **Prompt building** -- Formats instruction/context pairs into Llama 3
   chat-template strings that the fine-tuned model expects at inference.
2. **Decision normalisation** -- Maps the wide variety of raw decision
   labels (APPROVE, ALLOW, DEDUCTIBLE, ...) into three canonical buckets:
   ``approve``, ``reject``, ``review``.
3. **Response parsing** -- Extracts structured fields (decision, reason,
   confidence) from the model's free-text output.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pocket_ca.data_utils import normalize_text


# ── system prompt ───────────────────────────────────────────────────────
# The system prompt constrains the model to return exactly two lines.
# This rigid format ("Decision: ..." / "Reason: ...") makes downstream
# regex parsing reliable and avoids free-form verbosity.
SYSTEM_PROMPT = (
    "You are Pocket CA, AgentAuth's financial reasoning model. "
    "Follow policy, compare financial numbers carefully, and return exactly two "
    "lines: 'Decision: ...' and 'Reason: ...'."
)

# ── decision label buckets ──────────────────────────────────────────────
# Synthetic data generators produce many different decision labels across
# financial domains (expense approval, credit risk, M&A analysis, etc.).
# These three sets normalise them into a universal three-way taxonomy so
# that metrics code only needs to compare ``approve / reject / review``.
APPROVE_LABELS = {
    "APPROVE",
    "ALLOW",
    "DEDUCTIBLE",
    "NO_ALERT",
    "NORMAL",
    "HEALTHY",
    "BUY",
    "PASS",
    "ACCRETIVE",
    "WITHIN_BUDGET",
}
REJECT_LABELS = {
    "REJECT",
    "DENY",
    "DECLINE",
    "NON_DEDUCTIBLE",
    "ALERT",
    "OUTLIER",
    "AVOID",
    "FAIL",
    "DISTRESSED",
    "DILUTIVE",
    "FRAUD",
}
REVIEW_LABELS = {
    "REVIEW",
    "MANUAL_REVIEW",
    "HOLD",
    "ANSWER",     # generic catch-all from wrapped outputs
    "UNKNOWN",
}

# Words in the model's explanation that suggest the answer is uncertain.
# Their presence lowers the estimated confidence score.
HEDGING_TERMS = {
    "maybe",
    "unclear",
    "likely",
    "possibly",
    "appears",
    "suggests",
    "estimate",
}



# ── prompt construction ─────────────────────────────────────────────────

def build_prompt(instruction: str, context: dict[str, Any]) -> str:
    """Build a Llama 3 chat-template prompt string.

    Llama 3 uses a specific special-token protocol:
    * ``<|begin_of_text|>`` -- marks the very start of the sequence.
    * ``<|start_header_id|>role<|end_header_id|>`` -- opens each turn.
    * ``<|eot_id|>`` -- signals end-of-turn to the model.

    The final ``assistant`` header is left open (no ``<|eot_id|>``) so
    the model generates its completion as a continuation.
    """
    context_blob = json.dumps(context, indent=2, sort_keys=True)
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Instruction: {instruction}\nContext:\n{context_blob}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )



# ── label normalisation ─────────────────────────────────────────────────

def canonicalize_label(label: str) -> str:
    """Convert a raw decision string into a clean UPPER_SNAKE token.

    Example: ``"  manual review! "`` -> ``"MANUAL_REVIEW"``.
    """
    normalized = normalize_text(label).upper()
    # Replace any non-alphanumeric run with a single underscore
    normalized = re.sub(r"[^A-Z0-9]+", "_", normalized).strip("_")
    return normalized or "UNKNOWN"


def normalize_decision_label(raw_label: str) -> str:
    """Map a raw decision label into one of the three canonical buckets.

    Lookup order:
    1. Exact membership in APPROVE_LABELS / REJECT_LABELS / REVIEW_LABELS.
    2. Prefix match (e.g. ``"APPROVE_WITH_CONDITIONS"`` starts with ``APPROVE``).
    3. Substring match for generic tokens like CLASS or CATEGORY.
    4. Default fallback: ``"review"`` (safest -- forces human inspection).
    """
    label = canonicalize_label(raw_label)
    if label in APPROVE_LABELS or label.startswith("APPROVE"):
        return "approve"
    if label in REJECT_LABELS or label.startswith("REJECT"):
        return "reject"
    if label in REVIEW_LABELS or label.startswith("REVIEW"):
        return "review"
    if any(token in label for token in ("CLASS", "CATEGORY", "ANSWER")):
        return "review"
    # Unknown labels default to "review" to avoid auto-approving/rejecting
    return "review"



# ── confidence estimation ───────────────────────────────────────────────

def estimate_confidence(raw_decision: str, explanation: str) -> float:
    """Estimate a [0.05, 0.99] confidence score from decision + explanation.

    This is a rule-based heuristic used when the model does not emit an
    explicit ``Confidence:`` line.  The score is built from a base value
    plus bonuses and penalties:

      base        0.56   -- a neutral starting point just above 50%
      +0.14       if the model emitted a concrete (non-UNKNOWN) decision
      +0.10       if the explanation has >= 10 words (more detail = more
                  confident)
      +0.06       if the explanation has >= 20 words (even more detail)
      -0.12       if the explanation contains hedging language ("maybe",
                  "unclear", etc.)
      +0.05       if the normalised decision is *not* "review" (a
                  definitive approve/reject signals higher confidence)

    The final value is clamped to [0.05, 0.99] so it is never exactly 0
    or 1, which would break log-odds based downstream systems.
    """
    confidence = 0.56  # neutral base
    if raw_decision and raw_decision != "UNKNOWN":
        confidence += 0.14  # bonus: model committed to a concrete label
    explanation_length = len(explanation.split())
    if explanation_length >= 10:
        confidence += 0.1  # bonus: reasonably detailed explanation
    if explanation_length >= 20:
        confidence += 0.06  # bonus: highly detailed explanation
    lowered = explanation.lower()
    if any(term in lowered for term in HEDGING_TERMS):
        confidence -= 0.12  # penalty: hedging language present
    if normalize_decision_label(raw_decision) != "review":
        confidence += 0.05  # bonus: definitive approve/reject
    return round(min(max(confidence, 0.05), 0.99), 4)



# ── response parsing ────────────────────────────────────────────────────

def parse_response(response_text: str) -> dict[str, Any]:
    """Parse the model's raw text output into a structured result dict.

    Expected format from the model::

        Decision: APPROVE
        Reason: The expense is within the $500 policy limit.

    If the model also emits ``Confidence: 0.85``, that value is used
    directly; otherwise ``estimate_confidence`` synthesises one.

    When the ``Reason:`` line is missing, the entire response text is
    used as the explanation (graceful degradation for malformed output).
    """
    decision_match = re.search(r"Decision:\s*(.+)", response_text)
    reason_match = re.search(r"Reason:\s*(.+)", response_text)
    confidence_match = re.search(r"Confidence:\s*([01](?:\.\d+)?)", response_text)
    raw_decision = (
        canonicalize_label(decision_match.group(1)) if decision_match else "UNKNOWN"
    )
    explanation = (
        normalize_text(reason_match.group(1))
        if reason_match
        else normalize_text(response_text)  # fallback: use entire output
    )
    confidence = (
        round(float(confidence_match.group(1)), 4)
        if confidence_match
        else estimate_confidence(raw_decision, explanation)  # heuristic fallback
    )
    return {
        "raw_decision": raw_decision,
        "decision": normalize_decision_label(raw_decision),
        "reason": explanation,
        "explanation": explanation,
        "confidence": confidence,
    }
