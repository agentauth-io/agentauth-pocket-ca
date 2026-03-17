"""
Pocket CA QLoRA fine-tuning entry point.

This script orchestrates the full training pipeline:
  1. Parse CLI args and load YAML configs for training hyperparams and model setup.
  2. Validate that pre-built dataset splits exist on disk.
  3. Load and tokenize the dataset.
  4. Load the base model with optional 4-bit quantization, then attach LoRA adapters.
  5. Build a HuggingFace Trainer, optionally resuming from a prior checkpoint.
  6. Train, evaluate, and persist the final checkpoint + metrics summary.

Usage:
    python -m training.train_lora [--training-config PATH] [--model-config PATH]
                                  [--resume-from-checkpoint PATH] [--run-name NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

# ---------------------------------------------------------------------------
# Project root setup — ensures sibling packages (models/, training/) are
# importable when running this script directly (e.g. `python training/train_lora.py`).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model_loader import apply_lora, load_base_model
from models.tokenizer import load_tokenizer
from training.dataloader import load_training_splits, tokenize_splits
from training.trainer import build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pocket CA with production QLoRA.")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=PROJECT_ROOT / "configs/training.yaml",
        help="Training config YAML path.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs/model.yaml",
        help="Model config YAML path.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint path.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name override for this training invocation.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_resume_checkpoint(output_dir: Path, training_config: dict, cli_resume: str | None) -> str | None:
    """Determine the checkpoint directory to resume from, if any.

    Resolution order (first non-None wins):
      1. Explicit CLI flag  --resume-from-checkpoint
      2. ``resume_from_checkpoint`` key in the training YAML
      3. Auto-detect: when ``resume_if_available`` is True, scan *output_dir*
         for the most recent ``checkpoint-*`` subfolder written by the Trainer.
    Returning None means training starts from scratch.
    """
    if cli_resume:
        return cli_resume
    config_resume = training_config.get("resume_from_checkpoint")
    if config_resume:
        return str(config_resume)
    if training_config.get("resume_if_available", False):
        # get_last_checkpoint inspects output_dir for dirs matching "checkpoint-<step>"
        # and returns the one with the highest step number.
        checkpoint = get_last_checkpoint(str(output_dir))
        if checkpoint:
            return checkpoint
    return None


def configure_tracking(training_config: dict) -> None:
    """Set W&B environment variables if wandb tracking is requested.

    Uses ``setdefault`` so that env vars already set by the caller (e.g.
    a CI job) take precedence over config values.
    """
    if training_config.get("report_to") != "wandb":
        return
    os.environ.setdefault("WANDB_PROJECT", training_config["wandb_project"])
    if training_config.get("wandb_entity"):
        os.environ.setdefault("WANDB_ENTITY", training_config["wandb_entity"])


def apply_runtime_overrides(training_config: dict, args: argparse.Namespace) -> dict:
    """Merge CLI-level overrides into the YAML-loaded training config.

    Returns a *new* dict so the original YAML config stays untouched,
    which is useful for logging the original vs. effective config later.
    """
    resolved_config = dict(training_config)
    if args.run_name:
        resolved_config["run_name"] = args.run_name
    return resolved_config


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Configuration: load YAML files, apply CLI overrides, seed RNG
    # -----------------------------------------------------------------------
    args = parse_args()
    training_config = apply_runtime_overrides(load_yaml(args.training_config), args)
    model_config = load_yaml(args.model_config)
    set_seed(training_config["seed"])  # reproducibility across torch, numpy, random
    configure_tracking(training_config)

    # Enable TF32 on Ampere+ GPUs for ~3x faster matmul at near-FP32 accuracy.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = training_config["tf32"]

    # -----------------------------------------------------------------------
    # 2. Validate that pre-built dataset splits exist.
    #    We expect the user to have run `scripts/build_dataset.sh` beforehand,
    #    which generates JSONL files from the raw conversation data.
    # -----------------------------------------------------------------------
    train_path = PROJECT_ROOT / "data/training/train.jsonl"
    validation_path = PROJECT_ROOT / "data/training/validation.jsonl"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            "Dataset splits are missing. Run bash scripts/build_dataset.sh first."
        )

    # -----------------------------------------------------------------------
    # 3. Set up output directories (checkpoints, TensorBoard/wandb logs, metrics).
    #    Each run gets its own subdirectory keyed by run_name so multiple
    #    experiments can coexist without overwriting each other.
    # -----------------------------------------------------------------------
    checkpoints_root = PROJECT_ROOT / training_config["checkpoints_dir"]
    logs_root = PROJECT_ROOT / training_config["logs_dir"]
    metrics_root = PROJECT_ROOT / training_config["metrics_dir"]
    output_dir = checkpoints_root / training_config["run_name"]
    logging_dir = logs_root / training_config["run_name"]
    metrics_dir = metrics_root / training_config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Inject the resolved logging_dir back into the config dict so the
    # Trainer receives it when constructing TrainingArguments.
    training_config["logging_dir"] = str(logging_dir)

    # -----------------------------------------------------------------------
    # 4. Tokenize the dataset.
    #    Raw JSONL rows contain a "text" field; tokenize_splits converts them
    #    into input_ids / attention_mask / labels for causal-LM training.
    # -----------------------------------------------------------------------
    tokenizer = load_tokenizer(model_config["tokenizer_id"])
    raw_dataset = load_training_splits(train_path, validation_path)
    tokenized_dataset = tokenize_splits(
        raw_dataset,
        tokenizer=tokenizer,
        max_seq_length=training_config["max_seq_length"],
        num_proc=training_config.get("tokenizer_num_proc"),
    )

    # -----------------------------------------------------------------------
    # 5. Load the base model (optionally quantized to 4-bit via bitsandbytes)
    #    and attach LoRA adapters.  Only the adapter weights are trainable;
    #    the frozen base weights stay in low-precision to save VRAM.
    # -----------------------------------------------------------------------
    model = load_base_model(
        model_config["base_model_id"],
        torch_dtype=model_config["torch_dtype"],
        use_4bit=model_config["use_4bit"],
    )
    lora_config = model_config["lora"]
    model = apply_lora(
        model,
        r=lora_config["r"],
        alpha=lora_config["alpha"],
        dropout=lora_config["dropout"],
        bias=lora_config["bias"],
        target_modules=lora_config["target_modules"],
    )
    # Log trainable vs. total parameter counts to verify LoRA is wired up.
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # 6. Build the HuggingFace Trainer and optionally resume from a checkpoint.
    # -----------------------------------------------------------------------
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        training_config=training_config,
        output_dir=output_dir,
    )
    resume_checkpoint = resolve_resume_checkpoint(
        output_dir,
        training_config,
        args.resume_from_checkpoint,
    )

    # -----------------------------------------------------------------------
    # 7. Train, save, and evaluate.
    # -----------------------------------------------------------------------
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Persist the final LoRA adapter weights, trainer state (optimizer,
    # scheduler, RNG states), and tokenizer so inference can reload them.
    trainer.save_model(output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(output_dir)

    # Run a final evaluation pass on the validation set.
    evaluation_metrics = trainer.evaluate()

    # -----------------------------------------------------------------------
    # 8. Write a comprehensive training summary JSON.
    #    Saved in two locations: next to the checkpoint (for portability) and
    #    in the dedicated metrics directory (for easy comparison across runs).
    # -----------------------------------------------------------------------
    training_summary = {
        "run_name": training_config["run_name"],
        "base_model_id": model_config["base_model_id"],
        "tokenizer_id": model_config["tokenizer_id"],
        "training_config": training_config,
        "model_config": model_config,
        "resume_from_checkpoint": resume_checkpoint,
        "train_examples": len(raw_dataset["train"]),
        "validation_examples": len(raw_dataset["validation"]),
        "train_metrics": train_result.metrics,
        "eval_metrics": evaluation_metrics,
        # Full per-step log history lets us recreate loss curves without W&B.
        "log_history": trainer.state.log_history,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2, sort_keys=True)
    with (metrics_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2, sort_keys=True)

    print(f"Saved checkpoint to {output_dir}")
    print(f"Wrote metrics to {metrics_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
