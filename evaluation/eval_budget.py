from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pocket_ca.eval_utils import default_report_path, load_jsonl, load_model_bundle, predict_record
from pocket_ca.formatting import parse_response


BUDGET_CATEGORIES = {"budget_reasoning", "multi_transaction_budgeting"}


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


def main() -> None:
    args = parse_args()
    output_path = args.output or default_report_path(args.checkpoint, "budget_report")
    model, tokenizer, model_config = load_model_bundle(args.checkpoint, args.model_config)

    records = [
        record for record in load_jsonl(args.split) if record["category"] in BUDGET_CATEGORIES
    ][: args.limit]
    results = []
    matches = 0
    reject_total = 0
    reject_hits = 0
    for record in records:
        _, predicted = predict_record(model, tokenizer, model_config, record)
        expected = parse_response(record["output"])
        match = predicted["raw_decision"] == expected["raw_decision"]
        matches += int(match)
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

    report = {
        "samples": len(results),
        "budget_decision_accuracy": matches / max(len(results), 1),
        "reject_recall": reject_hits / max(reject_total, 1),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps({k: report[k] for k in ("samples", "budget_decision_accuracy", "reject_recall")}, indent=2))


if __name__ == "__main__":
    main()
