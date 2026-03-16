from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pocket_ca.data_utils import (
    deduplicate_records,
    ensure_instruction_record,
    load_jsonl,
    summarize_records,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge synthetic and imported datasets.")
    parser.add_argument(
        "--synthetic",
        type=Path,
        required=True,
        help="Synthetic dataset JSONL path.",
    )
    parser.add_argument(
        "--imported",
        type=Path,
        nargs="*",
        default=[],
        help="Imported dataset JSONL paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/unified_financial_dataset.jsonl"),
        help="Merged dataset JSONL path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/processed/merge_summary.json"),
        help="Merge summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.synthetic.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {args.synthetic}")

    merged_records = [ensure_instruction_record(record) for record in load_jsonl(args.synthetic)]
    for imported_path in args.imported:
        if not imported_path.exists():
            raise FileNotFoundError(f"Imported dataset not found: {imported_path}")
        merged_records.extend(
            ensure_instruction_record(record) for record in load_jsonl(imported_path)
        )

    before_dedup = len(merged_records)
    merged_records = deduplicate_records(merged_records)
    summary = {
        "before_deduplication": before_dedup,
        "after_deduplication": len(merged_records),
        "target_met": len(merged_records) >= 100_000,
        "dataset_summary": summarize_records(merged_records),
        "sources": [str(args.synthetic)] + [str(path) for path in args.imported],
    }

    write_jsonl(args.output, merged_records)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
