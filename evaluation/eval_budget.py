"""
Pocket CA -- Budget-Specific Evaluation Script
===============================================

Evaluates the model's accuracy on **budget-related tasks only**, filtering
the test split down to records whose ``category`` belongs to
``BUDGET_CATEGORIES``.

Why filter by category?
~~~~~~~~~~~~~~~~~~~~~~~
Budget reasoning is a critical safety surface for AgentAuth: an incorrect
approval on a transaction that exceeds the user's budget could lead to
real financial harm.  By evaluating budget categories in isolation we get a
sharper signal on this high-stakes slice without it being diluted by easier
(or harder) categories like fraud detection or tax classification.

Metrics
~~~~~~~
  * **budget_decision_accuracy** -- fraction of budget records where the
    predicted decision label matches the expected one (same semantics as
    ``reasoning_accuracy`` in ``eval_reasoning.py``).

  * **reject_recall** -- of the records whose gold label is ``REJECT``,
    how many did the model also predict as ``REJECT``?  This metric is
    tracked separately because *missing a reject is the costliest error*:
    it means the system would approve a transaction that should have been
    blocked.  High reject recall is therefore more important than high
    overall accuracy for budget decisions.

Usage
-----
    python evaluation/eval_budget.py --checkpoint experiments/checkpoints/pocket-ca-v1
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

from pocket_ca.eval_utils import default_report_path, load_jsonl, load_model_bundle, predict_record
from pocket_ca.formatting import parse_response

# Only these two categories are considered budget-related.  Other categories
# (e.g. fraud_detection, tax_classification) are intentionally excluded so
# that this evaluation focuses on spending-limit and multi-txn budget logic.
BUDGET_CATEGORIES = {"budget_reasoning", "multi_transaction_budgeting"}


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pocket CA budget decisions.")
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
    output_path = args.output or default_report_path(args.checkpoint, "budget_report")

    # Load model weights + tokenizer (adapter or merged) onto GPU.
    model, tokenizer, model_config = load_model_bundle(args.checkpoint, args.model_config)

    # Filter the full test split to budget-only records *before* applying the
    # limit, so ``--limit`` caps the number of budget records evaluated rather
    # than the total records read from disk.
    records = [
        record for record in load_jsonl(args.split) if record["category"] in BUDGET_CATEGORIES
    ][: args.limit]

    results = []
    matches = 0        # total correct decisions (any label)
    reject_total = 0   # number of gold REJECT records (denominator for recall)
    reject_hits = 0    # number of those the model also predicted as REJECT

    for record in records:
        # We discard the raw response text (first return value) because budget
        # evaluation only cares about the decision label, not exact wording.
        _, predicted = predict_record(model, tokenizer, model_config, record)
        expected = parse_response(record["output"])

        match = predicted["raw_decision"] == expected["raw_decision"]
        matches += int(match)

        # Track reject recall separately -- missing a REJECT is the most
        # dangerous failure mode for budget enforcement.
        if expected["raw_decision"] == "REJECT":
            reject_total += 1
            reject_hits += int(predicted["raw_decision"] == "REJECT")

        results.append(
            {
                "id": record["id"],
                "category": record["category"],
                "expected_decision": expected["raw_decision"],
                "predicted_decision": predicted["raw_decision"],
                "decision_match": match,
            }
        )

    # ---------------------------------------------------------------------------
    # Aggregate metrics and persist the report
    # ---------------------------------------------------------------------------
    # ``max(..., 1)`` prevents division-by-zero when no records match the
    # filter or when there are no gold REJECT labels.
    report = {
        "samples": len(results),
        "budget_decision_accuracy": matches / max(len(results), 1),
        "reject_recall": reject_hits / max(reject_total, 1),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    # Print a compact summary to stdout for CI / quick inspection.
    print(json.dumps({k: report[k] for k in ("samples", "budget_decision_accuracy", "reject_recall")}, indent=2))


if __name__ == "__main__":
    main()
