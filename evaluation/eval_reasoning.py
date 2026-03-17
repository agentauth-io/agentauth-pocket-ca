"""
Pocket CA -- General Reasoning Evaluation Script
=================================================

Measures how well a fine-tuned Pocket CA checkpoint reproduces the expected
financial decisions **across all task categories**.  (For budget-specific
metrics, see ``eval_budget.py``.)

Two complementary accuracy metrics are reported:

  * **reasoning_accuracy** (``raw_decision_matches / N``)
    Compares only the *canonicalised decision label* (e.g. ``APPROVE``,
    ``REJECT``) between the prediction and the gold reference.  This metric
    is lenient: the model may phrase its explanation differently and still
    score a hit as long as the decision label matches.

  * **exact_match_rate** (``exact_matches / N``)
    Compares the *entire* model output against the gold reference after
    whitespace-normalisation.  This is a much stricter metric because
    *both* the decision label and the explanation text must match.

Evaluation flow
---------------
1. Load the checkpoint (adapter or merged) and tokenizer via ``load_model_bundle``.
2. For each record in the test split (up to ``--limit``):
   a. Build a prompt from the record's instruction + context.
   b. Generate a response with greedy decoding.
   c. Parse both the predicted and expected outputs to extract ``raw_decision``.
   d. Record whether the decision label matched (``raw_match``) and whether
      the full normalised text matched (``exact_match``).
3. Write a JSON report to ``experiments/metrics/<checkpoint>/reasoning_report.json``.

Usage
-----
    python evaluation/eval_reasoning.py --checkpoint experiments/checkpoints/pocket-ca-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root bootstrapping
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pocket_ca.eval_utils import (
    default_report_path,
    load_jsonl,
    load_model_bundle,
    normalize_text,
    predict_record,
)
from pocket_ca.formatting import parse_response


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate overall Pocket CA reasoning.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Adapter checkpoint directory.")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs/model.yaml",
        help="Model config YAML path.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data/training/test.jsonl",
        help="Evaluation split JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=250, help="Max records to score.")
    parser.add_argument("--output", type=Path, default=None, help="Optional report path.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_path = args.output or default_report_path(args.checkpoint, "reasoning_report")

    # Load model weights + tokenizer (adapter or merged) onto GPU.
    model, tokenizer, model_config = load_model_bundle(args.checkpoint, args.model_config)

    records = load_jsonl(args.split)[: args.limit]
    results = []
    raw_decision_matches = 0  # count of records where the decision label matched
    exact_matches = 0         # count of records where the full output text matched

    for record in records:
        # Generate a prediction and parse both predicted and expected outputs.
        raw_response, predicted = predict_record(model, tokenizer, model_config, record)
        expected = parse_response(record["output"])

        # ``raw_decision`` is the canonicalised label (e.g. "APPROVE", "REJECT")
        # *before* it is lowered to the API-level label.  Comparing at this
        # level is intentional: it catches cases where the model outputs the
        # right semantic decision but uses a synonym (e.g. "ALLOW" vs "APPROVE").
        # Both sides go through ``canonicalize_label``, so they compare apples
        # to apples.
        raw_match = predicted["raw_decision"] == expected["raw_decision"]

        # Exact match is much stricter -- the entire output must be identical
        # after lowering and collapsing whitespace.
        exact_match = normalize_text(raw_response) == normalize_text(record["output"])

        raw_decision_matches += int(raw_match)
        exact_matches += int(exact_match)

        results.append(
            {
                "id": record["id"],
                "category": record["category"],
                "expected_decision": expected["raw_decision"],
                "predicted_decision": predicted["raw_decision"],
                "expected_explanation": expected["explanation"],
                "predicted_explanation": predicted["explanation"],
                "reasoning_accuracy": raw_match,
                "exact_match": exact_match,
            }
        )

    # ---------------------------------------------------------------------------
    # Aggregate metrics and persist the report
    # ---------------------------------------------------------------------------
    # ``max(len(results), 1)`` prevents division-by-zero when the test set is empty.
    report = {
        "samples": len(results),
        "reasoning_accuracy": raw_decision_matches / max(len(results), 1),
        "exact_match_rate": exact_matches / max(len(results), 1),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    # Print a compact summary to stdout for CI / quick inspection.
    print(json.dumps({k: report[k] for k in ("samples", "reasoning_accuracy", "exact_match_rate")}, indent=2))


if __name__ == "__main__":
    main()
