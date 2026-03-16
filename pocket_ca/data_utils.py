from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_RECORD_KEYS = {"instruction", "context", "output", "category"}
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
NUMERIC_PATTERN = re.compile(
    r"^\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?$"
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_output_text(text: str) -> str:
    lines = [normalize_text(line) for line in str(text).splitlines() if line.strip()]
    return "\n".join(lines)


def money(value: float) -> str:
    return f"${value:,.2f}"


def make_output(decision: str, reason: str) -> str:
    return f"Decision: {normalize_text(decision)}\nReason: {normalize_text(reason)}"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_number(value: str) -> float | None:
    candidate = normalize_text(value)
    if not candidate or not NUMERIC_PATTERN.match(candidate):
        return None
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
    lowered = key.lower()
    return any(token in lowered for token in NUMERIC_FIELD_TOKENS)


def normalize_value(value: Any, *, key: str = "") -> Any:
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
            if child_value is not None
        }
    if isinstance(value, str):
        candidate = normalize_text(value)
        parsed = parse_number(candidate)
        should_parse = looks_numeric_field(key) or any(
            marker in candidate for marker in ("$", "%", ",", ".", "(", ")")
        )
        if parsed is not None and should_parse:
            return parsed
        return candidate
    return value


def ensure_instruction_record(record: dict, *, source: str | None = None) -> dict:
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
        digest = hashlib.md5(build_dedup_key(normalized).encode("utf-8")).hexdigest()[:16]
        normalized["id"] = f"{normalized['category']}-{digest}"
    if not normalized["output"].startswith("Decision:"):
        normalized["output"] = make_output("ANSWER", normalized["output"])
    if source:
        normalized["source"] = source
    elif "source" in record and record["source"]:
        normalized["source"] = normalize_text(str(record["source"]))
    validate_record(normalized)
    return normalized


def validate_record(record: dict) -> None:
    if not isinstance(record["context"], dict):
        raise ValueError("Record context must be a dictionary.")
    if not record["instruction"]:
        raise ValueError("Instruction must be non-empty.")
    if not record["output"] or "Reason:" not in record["output"]:
        raise ValueError("Output must contain both Decision and Reason lines.")


def build_dedup_key(record: dict) -> str:
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
    unique_records = []
    seen = set()
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
