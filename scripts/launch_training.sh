#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${1:-}"
DATASET_SIZE="${2:-100000}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model.yaml}"

cd "${PROJECT_ROOT}"

# Rebuild dataset if missing, forced, or suspiciously small (< 1000 lines)
TRAIN_LINES=0
if [[ -f data/training/train.jsonl ]]; then
  TRAIN_LINES=$(wc -l < data/training/train.jsonl)
fi

if [[ "${FORCE_REBUILD_DATASET:-0}" == "1" || ! -f data/training/train.jsonl || "${TRAIN_LINES}" -lt 1000 ]]; then
  echo "Building dataset (current train lines: ${TRAIN_LINES}, target: ${DATASET_SIZE})..."
  bash scripts/build_dataset.sh "${DATASET_SIZE}"
fi

TRAIN_ARGS=(
  --training-config "${TRAINING_CONFIG}"
  --model-config "${MODEL_CONFIG}"
)

if [[ -n "${RUN_NAME}" ]]; then
  TRAIN_ARGS+=(--run-name "${RUN_NAME}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

python training/train_lora.py "${TRAIN_ARGS[@]}"
