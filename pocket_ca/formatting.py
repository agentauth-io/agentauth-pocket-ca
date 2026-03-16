from __future__ import annotations

import json
import re
from typing import Any

from pocket_ca.data_utils import normalize_text


SYSTEM_PROMPT = (
    "You are Pocket CA, AgentAuth's financial reasoning model. "
    "Follow policy, compare financial numbers carefully, and return exactly two "
    "lines: 'Decision: ...' and 'Reason: ...'."
)

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
    "ANSWER",
    "UNKNOWN",
}
HEDGING_TERMS = {
    "maybe",
    "unclear",
    "likely",
    "possibly",
    "appears",
    "suggests",
    "estimate",
}


def build_prompt(instruction: str, context: dict[str, Any]) -> str:
    context_blob = json.dumps(context, indent=2, sort_keys=True)
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Instruction: {instruction}\nContext:\n{context_blob}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def canonicalize_label(label: str) -> str:
    normalized = normalize_text(label).upper()
    normalized = re.sub(r"[^A-Z0-9]+", "_", normalized).strip("_")
    return normalized or "UNKNOWN"


def normalize_decision_label(raw_label: str) -> str:
    label = canonicalize_label(raw_label)
    if label in APPROVE_LABELS or label.startswith("APPROVE"):
        return "approve"
    if label in REJECT_LABELS or label.startswith("REJECT"):
        return "reject"
    if label in REVIEW_LABELS or label.startswith("REVIEW"):
        return "review"
    if any(token in label for token in ("CLASS", "CATEGORY", "ANSWER")):
        return "review"
    return "review"


def estimate_confidence(raw_decision: str, explanation: str) -> float:
    confidence = 0.56
    if raw_decision and raw_decision != "UNKNOWN":
        confidence += 0.14
    explanation_length = len(explanation.split())
    if explanation_length >= 10:
        confidence += 0.1
    if explanation_length >= 20:
        confidence += 0.06
    lowered = explanation.lower()
    if any(term in lowered for term in HEDGING_TERMS):
        confidence -= 0.12
    if normalize_decision_label(raw_decision) != "review":
        confidence += 0.05
    return round(min(max(confidence, 0.05), 0.99), 4)


def parse_response(response_text: str) -> dict[str, Any]:
    decision_match = re.search(r"Decision:\s*(.+)", response_text)
    reason_match = re.search(r"Reason:\s*(.+)", response_text)
    confidence_match = re.search(r"Confidence:\s*([01](?:\.\d+)?)", response_text)
    raw_decision = (
        canonicalize_label(decision_match.group(1)) if decision_match else "UNKNOWN"
    )
    explanation = (
        normalize_text(reason_match.group(1))
        if reason_match
        else normalize_text(response_text)
    )
    confidence = (
        round(float(confidence_match.group(1)), 4)
        if confidence_match
        else estimate_confidence(raw_decision, explanation)
    )
    return {
        "raw_decision": raw_decision,
        "decision": normalize_decision_label(raw_decision),
        "reason": explanation,
        "explanation": explanation,
        "confidence": confidence,
    }
