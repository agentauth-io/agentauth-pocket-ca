"""Data loading, normalisation, deduplication, and stratified splitting for
the Pocket CA training pipeline.

Every raw record produced by synthetic data generators passes through this
module before it reaches the formatter or the trainer.  The pipeline is:

  load  ->  normalize fields  ->  deduplicate  ->  stratified split

Key design decisions:
* String-encoded numbers (``"$1,234.56"``) are automatically cast to floats
  when the field name matches a financial keyword heuristic, so the model
  sees consistent numeric types in its context dictionaries.
* Deduplication is content-based (JSON hash of the four semantic fields),
  not id-based, because multiple generators can independently emit the same
  scenario.
* Stratified splitting guarantees that every *category* is proportionally
  represented in train / validation / test sets, even when category sizes
  are highly uneven.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ── schema constants ────────────────────────────────────────────────────
# Every instruction record must contain at least these four keys.
REQUIRED_RECORD_KEYS = {"instruction", "context", "output", "category"}
# Heuristic token list for auto-converting string-encoded numbers to floats.
# If a *field name* (case-insensitive) contains any of these tokens AND the
# value matches NUMERIC_PATTERN, the value is parsed into a Python float.
# This avoids blindly converting identifiers like "PO-1234" that happen to
# contain digits.
NUMERIC_FIELD_TOKENS = (
    "amount",
    "budget",
    "remaining",
    "spend",
    "price",
    "cost",
    "revenue",
    "expense",
    "profit",
    "margin",
    "cash",
    "ebitda",
    "income",
    "assets",
    "liabilities",
    "equity",
    "debt",
    "ratio",
    "loan",
    "payment",
    "apr",
    "rate",
    "interest",
    "tax",
    "principal",
    "forecast",
    "variance",
    "valuation",
    "multiple",
    "balance",
    "salary",
)

# Matches strings that look like numbers in financial documents:
# optional parentheses for negatives, optional $, comma-grouped digits,
# optional decimal portion, optional trailing %.  Examples:
#   "$1,234.56"   "(500)"   "12.5%"   "-$3,000"
NUMERIC_PATTERN = re.compile(
    r"^\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?$"
)



# ── text helpers ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Collapse all whitespace runs into a single space and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_output_text(text: str) -> str:
    """Normalize each non-empty line independently, preserving line breaks.

    This keeps the ``Decision:`` / ``Reason:`` two-line format intact while
    cleaning up stray whitespace within each line.
    """
    lines = [normalize_text(line) for line in str(text).splitlines() if line.strip()]
    return "\n".join(lines)


def money(value: float) -> str:
    """Format a float as a US-dollar string, e.g. ``$1,234.56``."""
    return f"${value:,.2f}"


def make_output(decision: str, reason: str) -> str:
    """Build the canonical two-line output expected by the model."""
    return f"Decision: {normalize_text(decision)}\nReason: {normalize_text(reason)}"


# ── I/O helpers ─────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Read a JSON-Lines file; blank lines are silently skipped."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as sorted-key JSON-Lines, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a single JSON object with human-readable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)



# ── numeric parsing & value normalisation ───────────────────────────────

def parse_number(value: str) -> float | None:
    """Try to parse a financial-style string into a float.

    Handles dollar signs, commas, percentages, and accounting-style
    parenthesised negatives, e.g. ``"($1,200.50)"`` -> ``-1200.5``.
    Returns ``None`` when the string is not a recognisable number.
    """
    candidate = normalize_text(value)
    if not candidate or not NUMERIC_PATTERN.match(candidate):
        return None
    # Accounting convention: parentheses denote negative values
    negative = candidate.startswith("(") and candidate.endswith(")")
    cleaned = candidate.strip("()").replace("$", "").replace(",", "").replace("%", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if negative:
        number *= -1
    return round(number, 6)


def looks_numeric_field(key: str) -> bool:
    """Return True if the field name suggests a numeric/financial value.

    This is the first half of the two-gate heuristic.  A string is only
    auto-converted to float when *both* the field name passes this check
    AND the value matches NUMERIC_PATTERN.
    """
    lowered = key.lower()
    return any(token in lowered for token in NUMERIC_FIELD_TOKENS)


def normalize_value(value: Any, *, key: str = "") -> Any:
    """Recursively normalise a value from raw record data.

    * Floats are rounded to 6 decimal places for consistency.
    * Dicts are recursively normalised; ``None`` values are dropped.
    * Strings are whitespace-normalised; those that look like numbers in
      a financial field (two-gate heuristic: field-name + value pattern)
      are auto-converted to floats so the model sees uniform types.
    """
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [normalize_value(item, key=key) for item in value]
    if isinstance(value, dict):
        return {
            normalize_text(str(child_key)): normalize_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
            if child_value is not None  # drop None entries to keep context clean
        }
    if isinstance(value, str):
        candidate = normalize_text(value)
        parsed = parse_number(candidate)
        # Two-gate heuristic: convert only when the field name hints at a
        # numeric domain OR the value itself contains financial markers.
        should_parse = looks_numeric_field(key) or any(
            marker in candidate for marker in ("$", "%", ",", ".", "(", ")")
        )
        if parsed is not None and should_parse:
            return parsed
        return candidate
    return value



# ── record validation & normalisation ───────────────────────────────────

def ensure_instruction_record(record: dict, *, source: str | None = None) -> dict:
    """Validate and normalise a raw record into the canonical schema.

    Steps performed:
    1. Check that all REQUIRED_RECORD_KEYS are present.
    2. Normalise every field (whitespace, numeric conversion, etc.).
    3. Auto-generate a content-hash id if one is not provided.
    4. Wrap bare outputs in the ``Decision: / Reason:`` template.
    5. Attach an optional ``source`` tag for provenance tracking.
    6. Run ``validate_record`` as a final guard.
    """
    missing = REQUIRED_RECORD_KEYS.difference(record)
    if missing:
        raise ValueError(f"Record missing required keys: {sorted(missing)}")
    normalized = {
        "id": normalize_text(str(record.get("id") or "")),
        "instruction": normalize_text(str(record["instruction"])),
        "context": normalize_value(record["context"]),
        "output": normalize_output_text(record["output"]),
        "category": normalize_text(str(record["category"])),
    }
    if not normalized["id"]:
        # Derive a deterministic id from the content hash so identical
        # records always get the same id, enabling deduplication.
        digest = hashlib.md5(build_dedup_key(normalized).encode("utf-8")).hexdigest()[:16]
        normalized["id"] = f"{normalized['category']}-{digest}"
    if not normalized["output"].startswith("Decision:"):
        # Records without a structured decision are wrapped as a generic
        # ANSWER so downstream code can always assume the two-line format.
        normalized["output"] = make_output("ANSWER", normalized["output"])
    if source:
        normalized["source"] = source
    elif "source" in record and record["source"]:
        normalized["source"] = normalize_text(str(record["source"]))
    validate_record(normalized)
    return normalized


def validate_record(record: dict) -> None:
    """Raise ``ValueError`` if the record does not meet structural requirements."""
    if not isinstance(record["context"], dict):
        raise ValueError("Record context must be a dictionary.")
    if not record["instruction"]:
        raise ValueError("Instruction must be non-empty.")
    if not record["output"] or "Reason:" not in record["output"]:
        raise ValueError("Output must contain both Decision and Reason lines.")



# ── deduplication ───────────────────────────────────────────────────────

def build_dedup_key(record: dict) -> str:
    """Create a canonical JSON string from the four semantic fields.

    The result is used both as a dedup fingerprint and (via MD5) as a
    content-based id.  ``sort_keys=True`` guarantees that key order
    never causes false mismatches.
    """
    return json.dumps(
        {
            "instruction": record["instruction"],
            "context": record["context"],
            "output": record["output"],
            "category": record["category"],
        },
        sort_keys=True,
    )


def deduplicate_records(records: list[dict]) -> list[dict]:
    """Remove exact content duplicates while preserving insertion order.

    Two records are considered duplicates if their ``build_dedup_key``
    outputs are identical (same instruction, context, output, and category).
    """
    unique_records = []
    seen: set[str] = set()
    for record in records:
        key = build_dedup_key(record)
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(record)
    return unique_records


def allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw_counts = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remainder = total - sum(counts)
    ranked = sorted(
        range(3),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in ranked[:remainder]:
        counts[index] += 1

    non_zero_targets = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if total >= len(non_zero_targets):
        for index in non_zero_targets:
            if counts[index] == 0:
                donor = max(range(3), key=lambda position: counts[position])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    return counts[0], counts[1], counts[2]


def stratified_split(
    records: list[dict],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    ratio_total = round(train_ratio + validation_ratio + test_ratio, 6)
    if ratio_total != 1.0:
        raise ValueError("Train/validation/test ratios must sum to 1.0.")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    rng = random.Random(seed)
    train_records: list[dict] = []
    validation_records: list[dict] = []
    test_records: list[dict] = []
    for group_records in grouped.values():
        rng.shuffle(group_records)
        train_count, validation_count, test_count = allocate_counts(
            len(group_records),
            (train_ratio, validation_ratio, test_ratio),
        )
        train_records.extend(group_records[:train_count])
        validation_records.extend(
            group_records[train_count : train_count + validation_count]
        )
        test_records.extend(
            group_records[
                train_count + validation_count : train_count + validation_count + test_count
            ]
        )
    rng.shuffle(train_records)
    rng.shuffle(validation_records)
    rng.shuffle(test_records)
    return train_records, validation_records, test_records


def summarize_records(records: list[dict]) -> dict[str, Any]:
    category_counts = Counter(record["category"] for record in records)
    source_counts = Counter(record.get("source", "unknown") for record in records)
    return {
        "samples": len(records),
        "categories": dict(sorted(category_counts.items())),
        "sources": dict(sorted(source_counts.items())),
    }
