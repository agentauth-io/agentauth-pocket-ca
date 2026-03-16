#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_SIZE="${1:-100000}"
IMPORTED_OUTPUT="data/raw/imported_financial_reasoning.jsonl"
UNIFIED_OUTPUT="data/raw/unified_financial_dataset.jsonl"

cd "${PROJECT_ROOT}"

python datasets/generate_dataset.py \
  --output data/raw/financial_scenarios.jsonl \
  --size "${DATASET_SIZE}" \
  --seed 42

IMPORT_ARGS=()
if [[ -n "${FINQA_PATH:-}" ]]; then
  IMPORT_ARGS+=(--finqa "${FINQA_PATH}")
fi
if [[ -n "${CONVFINQA_PATH:-}" ]]; then
  IMPORT_ARGS+=(--convfinqa "${CONVFINQA_PATH}")
fi
if [[ -n "${FINANCEBENCH_PATH:-}" ]]; then
  IMPORT_ARGS+=(--financebench "${FINANCEBENCH_PATH}")
fi
if [[ -n "${FINR1_PATH:-}" ]]; then
  IMPORT_ARGS+=(--finr1 "${FINR1_PATH}")
fi

if [[ ${#IMPORT_ARGS[@]} -gt 0 ]]; then
  python datasets/import_financial_datasets.py \
    "${IMPORT_ARGS[@]}" \
    --output "${IMPORTED_OUTPUT}"
  python datasets/merge_datasets.py \
    --synthetic data/raw/financial_scenarios.jsonl \
    --imported "${IMPORTED_OUTPUT}" \
    --output "${UNIFIED_OUTPUT}"
else
  python datasets/merge_datasets.py \
    --synthetic data/raw/financial_scenarios.jsonl \
    --output "${UNIFIED_OUTPUT}"
fi

PREPROCESS_ARGS=()
if [[ "${SKIP_TOKENIZER_VALIDATION:-0}" == "1" ]]; then
  PREPROCESS_ARGS+=(--skip-tokenizer-validation)
fi

python datasets/preprocess.py \
  --input "${UNIFIED_OUTPUT}" \
  --processed-output data/processed/financial_instruction_dataset.jsonl \
  --training-dir data/training \
  --train-ratio 0.8 \
  --validation-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42 \
  "${PREPROCESS_ARGS[@]}"
