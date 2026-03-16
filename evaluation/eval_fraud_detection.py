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


POSITIVE_LABEL = "MANUAL_REVIEW"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Pocket CA fraud detection.")
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
    output_path = args.output or default_report_path(args.checkpoint, "fraud_detection_report")
    model, tokenizer, model_config = load_model_bundle(args.checkpoint, args.model_config)

    records = [
        record
        for record in load_jsonl(args.split)
        if record["category"] == "fraud_detection"
    ][: args.limit]
    results = []
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for record in records:
        _, predicted = predict_record(model, tokenizer, model_config, record)
        expected = parse_response(record["output"])
        predicted_positive = predicted["raw_decision"] == POSITIVE_LABEL
        expected_positive = expected["raw_decision"] == POSITIVE_LABEL
        true_positive += int(predicted_positive and expected_positive)
        false_positive += int(predicted_positive and not expected_positive)
        false_negative += int((not predicted_positive) and expected_positive)
        results.append(
            {
                "id": record["id"],
                "expected_decision": expected["raw_decision"],
                "predicted_decision": predicted["raw_decision"],
                "is_true_positive": predicted_positive and expected_positive,
            }
        )

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    report = {
        "samples": len(results),
        "fraud_detection_precision": precision,
        "fraud_detection_recall": recall,
        "fraud_detection_f1": (
            2 * precision * recall / max(precision + recall, 1e-9)
            if precision + recall
            else 0.0
        ),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "samples",
                    "fraud_detection_precision",
                    "fraud_detection_recall",
                    "fraud_detection_f1",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
