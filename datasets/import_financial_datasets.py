from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pocket_ca.data_utils import ensure_instruction_record, make_output, write_jsonl


SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv"}


def resolve_input_files(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                    resolved.append(candidate)
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"Dataset path not found: {path}")
    return resolved


def flatten_json_payload(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "examples", "records", "items"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
        flattened: list[dict] = []
        for value in payload.values():
            if isinstance(value, list):
                flattened.extend(item for item in value if isinstance(item, dict))
        if flattened:
            return flattened
        return [payload]
    return []


def load_file_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return flatten_json_payload(json.load(handle))
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported dataset file type: {path}")


def joined_text(parts: Iterable[Any]) -> str:
    cleaned = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            cleaned.extend(str(item) for item in part if item)
        elif isinstance(part, dict):
            cleaned.append(json.dumps(part, sort_keys=True))
        else:
            cleaned.append(str(part))
    return "\n".join(segment.strip() for segment in cleaned if str(segment).strip())


def convert_finqa_record(record: dict, record_id: str) -> dict | None:
    qa = record.get("qa", {}) if isinstance(record.get("qa"), dict) else {}
    question = record.get("question") or qa.get("question")
    answer = record.get("answer") or qa.get("answer") or qa.get("exe_ans")
    reasoning = (
        record.get("reasoning")
        or qa.get("program")
        or qa.get("program_re")
        or record.get("gold_inds")
    )
    if not question or answer is None:
        return None
    context = {
        "dataset": "FinQA",
        "company": record.get("company") or record.get("company_name"),
        "table": record.get("table") or record.get("table_ori"),
        "report_text": joined_text(
            [record.get("pre_text"), record.get("post_text"), record.get("text")]
        ),
        "reasoning_steps": reasoning,
    }
    reason = str(answer)
    if reasoning:
        reason += f" Reasoning trace: {reasoning}"
    return ensure_instruction_record(
        {
            "id": record_id,
            "instruction": question,
            "context": context,
            "output": make_output("ANSWER", reason),
            "category": "finqa_reasoning",
            "source": "FinQA",
        }
    )


def convert_convfinqa_record(record: dict, record_id_prefix: str) -> list[dict]:
    base_context = {
        "dataset": "ConvFinQA",
        "company": record.get("company") or record.get("company_name"),
        "table": record.get("table") or record.get("table_ori"),
        "report_text": joined_text(
            [record.get("pre_text"), record.get("post_text"), record.get("text")]
        ),
    }
    turns = None
    for key in ("qa", "qas", "questions", "conversation", "dialogue", "turns"):
        if isinstance(record.get(key), list):
            turns = record[key]
            break
    if not turns:
        generic = convert_finqa_record(record, f"{record_id_prefix}-000")
        return [generic] if generic is not None else []

    converted = []
    history: list[dict[str, str]] = []
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        question = turn.get("question") or turn.get("query")
        answer = turn.get("answer") or turn.get("exe_ans") or turn.get("final_answer")
        reasoning = turn.get("program") or turn.get("reasoning")
        if not question or answer is None:
            continue
        context = dict(base_context)
        context["conversation_history"] = list(history)
        context["reasoning_steps"] = reasoning
        converted.append(
            ensure_instruction_record(
                {
                    "id": f"{record_id_prefix}-{turn_index:03d}",
                    "instruction": question,
                    "context": context,
                    "output": make_output(
                        "ANSWER",
                        f"{answer}"
                        + (f" Reasoning trace: {reasoning}" if reasoning else ""),
                    ),
                    "category": "convfinqa_reasoning",
                    "source": "ConvFinQA",
                }
            )
        )
        history.append({"question": str(question), "answer": str(answer)})
    return converted


def convert_reasoning_record(record: dict, dataset_name: str, category: str, record_id: str) -> dict | None:
    question = (
        record.get("question")
        or record.get("instruction")
        or record.get("query")
        or record.get("prompt")
    )
    answer = (
        record.get("answer")
        or record.get("final_answer")
        or record.get("response")
        or record.get("gold_answer")
    )
    reasoning = (
        record.get("reasoning")
        or record.get("rationale")
        or record.get("analysis")
        or record.get("cot")
        or record.get("explanation")
    )
    if not question or answer is None:
        return None
    context = {
        "dataset": dataset_name,
        "company": record.get("company") or record.get("ticker"),
        "period": record.get("period") or record.get("fiscal_period"),
        "report_text": joined_text(
            [
                record.get("context"),
                record.get("documents"),
                record.get("evidence"),
                record.get("supporting_passages"),
            ]
        ),
        "reasoning_steps": reasoning,
    }
    return ensure_instruction_record(
        {
            "id": record_id,
            "instruction": question,
            "context": context,
            "output": make_output(
                "ANSWER",
                f"{answer}" + (f" Reasoning trace: {reasoning}" if reasoning else ""),
            ),
            "category": category,
            "source": dataset_name,
        }
    )


def import_source(paths: list[Path], dataset_name: str, max_records_per_source: int | None) -> list[dict]:
    if not paths:
        return []
    imported: list[dict] = []
    for file_index, path in enumerate(resolve_input_files(paths)):
        records = load_file_records(path)
        for record_index, record in enumerate(records):
            if max_records_per_source is not None and len(imported) >= max_records_per_source:
                return imported
            record_id_prefix = f"{dataset_name.lower()}-{file_index:03d}-{record_index:06d}"
            if dataset_name == "FinQA":
                converted = convert_finqa_record(record, record_id_prefix)
                if converted is not None:
                    imported.append(converted)
            elif dataset_name == "ConvFinQA":
                imported.extend(convert_convfinqa_record(record, record_id_prefix))
            elif dataset_name == "FinanceBench":
                converted = convert_reasoning_record(
                    record,
                    dataset_name,
                    "financebench_reasoning",
                    record_id_prefix,
                )
                if converted is not None:
                    imported.append(converted)
            elif dataset_name == "FinR1":
                converted = convert_reasoning_record(
                    record,
                    dataset_name,
                    "finr1_reasoning",
                    record_id_prefix,
                )
                if converted is not None:
                    imported.append(converted)
    return imported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import external financial reasoning datasets into Pocket CA format."
    )
    parser.add_argument("--finqa", type=Path, nargs="*", default=[], help="FinQA files or directories.")
    parser.add_argument(
        "--convfinqa",
        type=Path,
        nargs="*",
        default=[],
        help="ConvFinQA files or directories.",
    )
    parser.add_argument(
        "--financebench",
        type=Path,
        nargs="*",
        default=[],
        help="FinanceBench style files or directories.",
    )
    parser.add_argument("--finr1", type=Path, nargs="*", default=[], help="FinR1 style files or directories.")
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="Optional cap per named source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/imported_financial_reasoning.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imported_records = []
    imported_records.extend(import_source(args.finqa, "FinQA", args.max_records_per_source))
    imported_records.extend(
        import_source(args.convfinqa, "ConvFinQA", args.max_records_per_source)
    )
    imported_records.extend(
        import_source(args.financebench, "FinanceBench", args.max_records_per_source)
    )
    imported_records.extend(import_source(args.finr1, "FinR1", args.max_records_per_source))
    write_jsonl(args.output, imported_records)
    print(f"Wrote {len(imported_records)} imported records to {args.output}")


if __name__ == "__main__":
    main()
