"""Preprocess the unified Pocket CA dataset for model training.

This script takes the merged raw JSONL file (synthetic + imported records),
enriches each record with a formatted prompt and optional token count,
deduplicates, and then splits the result into train / validation / test
JSONL files using a stratified split that preserves category proportions.

Pipeline:
    1. Load the unified raw dataset.
    2. For each record:
       a. Normalise to the canonical instruction schema.
       b. Validate that all context fields are JSON-safe primitives.
       c. Build the Llama-3 chat-formatted prompt string.
       d. Concatenate prompt + output into a ``text`` field for causal LM
          training.
       e. Optionally tokenise to compute ``token_count``.
    3. Deduplicate on record ``id``.
    4. Stratified split by ``category``.
    5. Write processed JSONL, split files, and a summary JSON.

Usage:
    python datasets/preprocess.py \
        --input data/raw/unified_financial_dataset.jsonl \
        --processed-output data/processed/financial_instruction_dataset.jsonl \
        --training-dir data/training \
        --stats-output data/processed/preprocess_summary.json

Inputs:  data/raw/unified_financial_dataset.jsonl
Outputs: data/processed/financial_instruction_dataset.jsonl
         data/training/{train,validation,test}.jsonl
         data/processed/preprocess_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# --- Project path setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pocket_ca.data_utils import (
    deduplicate_records,
    ensure_instruction_record,
    load_jsonl,
    stratified_split,
    summarize_records,
    write_json,
    write_jsonl,
)
from pocket_ca.formatting import build_prompt


# --- Context field validation ---

def validate_financial_fields(value: Any, *, path: str = "context") -> None:
    """Recursively verify that every leaf value in a context dict is a
    JSON-serialisable primitive (str, int, float, bool).

    Nulls are rejected because downstream prompt formatting would render
    them as the literal string ``"None"``, which confuses the model.
    The ``path`` parameter builds a dotted/bracketed trail for clear
    error messages (e.g. ``context.transactions[2].amount``).
    """
    if isinstance(value, dict):
        for key, child_value in value.items():
            validate_financial_fields(child_value, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_financial_fields(item, path=f"{path}[{index}]")
        return
    if value is None:
        raise ValueError(f"Null value found in {path}")
    if isinstance(value, (str, int, float, bool)):
        return
    raise ValueError(f"Unsupported value type at {path}: {type(value)!r}")


# --- Record enrichment ---

def enrich_record(record: dict, tokenizer=None) -> dict:
    """Transform a raw record into a training-ready record.

    Steps:
        1. Normalise schema via ``ensure_instruction_record``.
        2. Validate context fields (no nulls, no exotic types).
        3. Build the Llama-3 chat prompt from instruction + context.
        4. Concatenate prompt + output + end-of-turn token into a
           ``text`` field that can be fed directly to a causal LM trainer.
        5. If a tokenizer is provided, count tokens so the summary can
           report min/max/avg sequence lengths.
    """
    normalized = ensure_instruction_record(record)
    validate_financial_fields(normalized["context"])
    prompt = build_prompt(normalized["instruction"], normalized["context"])
    normalized["prompt"] = prompt
    # The <|eot_id|> sentinel tells the Llama-3 tokenizer where the
    # assistant turn ends; the trainer masks loss on everything before
    # the output portion.
    normalized["text"] = prompt + normalized["output"] + "\n<|eot_id|>"
    if tokenizer is not None:
        normalized["token_count"] = len(
            tokenizer(
                normalized["text"],
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
        )
    return normalized


# --- CLI argument parsing ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Pocket CA datasets.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/unified_financial_dataset.jsonl"),
        help="Unified raw dataset JSONL path.",
    )
    parser.add_argument(
        "--processed-output",
        type=Path,
        default=Path("data/processed/financial_instruction_dataset.jsonl"),
        help="Processed dataset JSONL path.",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("data/training"),
        help="Directory for train/validation/test splits.",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=Path("data/processed/preprocess_summary.json"),
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--tokenizer-id",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Tokenizer identifier used for token counting.",
    )
    parser.add_argument(
        "--skip-tokenizer-validation",
        action="store_true",
        help="Skip tokenizer loading and token counts.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


# --- Entrypoint ---

def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")

    # --- Optional tokenizer loading ---
    # The tokenizer is only needed for sequence-length statistics; in CI
    # or lightweight runs it can be skipped with --skip-tokenizer-validation.
    tokenizer = None
    if not args.skip_tokenizer_validation:
        from models.tokenizer import load_tokenizer

        tokenizer = load_tokenizer(args.tokenizer_id)

    # --- Enrich and deduplicate ---
    raw_records = load_jsonl(args.input)
    processed_records = [enrich_record(record, tokenizer=tokenizer) for record in raw_records]
    deduped_records = deduplicate_records(processed_records)

    # --- Stratified train / validation / test split ---
    # Stratification is by ``category`` so each split has the same
    # category distribution as the full dataset.
    train_records, validation_records, test_records = stratified_split(
        deduped_records,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # --- Write output files ---
    write_jsonl(args.processed_output, deduped_records)
    write_jsonl(args.training_dir / "train.jsonl", train_records)
    write_jsonl(args.training_dir / "validation.jsonl", validation_records)
    write_jsonl(args.training_dir / "test.jsonl", test_records)

    # --- Build summary statistics ---
    # Token counts are only available when a tokenizer was loaded.
    token_counts = [record.get("token_count", 0) for record in deduped_records if "token_count" in record]
    summary = {
        "input_path": str(args.input),
        "processed_summary": summarize_records(deduped_records),
        "split_sizes": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "tokenizer_id": None if tokenizer is None else args.tokenizer_id,
        "token_stats": {
            "min": min(token_counts) if token_counts else None,
            "max": max(token_counts) if token_counts else None,
            "avg": round(sum(token_counts) / len(token_counts), 2) if token_counts else None,
        },
    }
    write_json(args.stats_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
