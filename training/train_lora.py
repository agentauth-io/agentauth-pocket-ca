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
    if cli_resume:
        return cli_resume
    config_resume = training_config.get("resume_from_checkpoint")
    if config_resume:
        return str(config_resume)
    if training_config.get("resume_if_available", False):
        checkpoint = get_last_checkpoint(str(output_dir))
        if checkpoint:
            return checkpoint
    return None


def configure_tracking(training_config: dict) -> None:
    if training_config.get("report_to") != "wandb":
        return
    os.environ.setdefault("WANDB_PROJECT", training_config["wandb_project"])
    if training_config.get("wandb_entity"):
        os.environ.setdefault("WANDB_ENTITY", training_config["wandb_entity"])


def apply_runtime_overrides(training_config: dict, args: argparse.Namespace) -> dict:
    resolved_config = dict(training_config)
    if args.run_name:
        resolved_config["run_name"] = args.run_name
    return resolved_config


def main() -> None:
    args = parse_args()
    training_config = apply_runtime_overrides(load_yaml(args.training_config), args)
    model_config = load_yaml(args.model_config)
    set_seed(training_config["seed"])
    configure_tracking(training_config)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = training_config["tf32"]

    train_path = PROJECT_ROOT / "data/training/train.jsonl"
    validation_path = PROJECT_ROOT / "data/training/validation.jsonl"
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            "Dataset splits are missing. Run bash scripts/build_dataset.sh first."
        )

    checkpoints_root = PROJECT_ROOT / training_config["checkpoints_dir"]
    logs_root = PROJECT_ROOT / training_config["logs_dir"]
    metrics_root = PROJECT_ROOT / training_config["metrics_dir"]
    output_dir = checkpoints_root / training_config["run_name"]
    logging_dir = logs_root / training_config["run_name"]
    metrics_dir = metrics_root / training_config["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    training_config["logging_dir"] = str(logging_dir)
    tokenizer = load_tokenizer(model_config["tokenizer_id"])
    raw_dataset = load_training_splits(train_path, validation_path)
    tokenized_dataset = tokenize_splits(
        raw_dataset,
        tokenizer=tokenizer,
        max_seq_length=training_config["max_seq_length"],
        num_proc=training_config.get("tokenizer_num_proc"),
    )

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
    model.print_trainable_parameters()

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
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output_dir)
    trainer.save_state()
    tokenizer.save_pretrained(output_dir)
    evaluation_metrics = trainer.evaluate()

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
