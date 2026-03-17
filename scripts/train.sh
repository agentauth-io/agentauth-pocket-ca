#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train.sh — Launch Pocket CA QLoRA training
# ---------------------------------------------------------------------------
# Usage:
#   bash scripts/train.sh                          # default run
#   bash scripts/train.sh --run-name my-run        # custom run name
#   bash scripts/train.sh --resume-from-checkpoint experiments/checkpoints/pocket-ca-v1/checkpoint-500
#
# Environment variables:
#   TRAINING_CONFIG  — path to training YAML (default: configs/training.yaml)
#   MODEL_CONFIG     — path to model YAML    (default: configs/model.yaml)
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve project root relative to this script's location
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Config file paths (override via environment if needed)
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model.yaml}"

# Ensure dataset exists before training
if [[ ! -f data/training/train.jsonl ]]; then
    echo "Dataset not found. Building dataset first..."
    bash scripts/build_dataset.sh
fi

# Launch training with any extra CLI arguments passed through
echo "Starting QLoRA training..."
python training/train_lora.py \
    --training-config "${TRAINING_CONFIG}" \
    --model-config "${MODEL_CONFIG}" \
    "$@"
