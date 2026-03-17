"""
Trainer construction and custom data collation for causal-LM fine-tuning.

Key responsibilities:
  - SupervisedDataCollator: dynamically pads variable-length tokenized samples
    within each batch, using -100 for label padding so that PyTorch's
    CrossEntropyLoss ignores pad positions.
  - create_training_args: translates our flat YAML config dict into a
    HuggingFace TrainingArguments object, with runtime introspection to stay
    compatible across different ``transformers`` versions.
  - build_trainer: wires model, tokenizer, datasets, and config into a
    ready-to-use HuggingFace Trainer instance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments


@dataclass
class SupervisedDataCollator:
    """Batch collator that pads input_ids, attention_mask, and labels on the fly.

    Why a custom collator?  The default HuggingFace collator either doesn't
    handle ``labels`` at all, or pads them with 0 — which would count as a
    real token ID and corrupt the loss.  We pad labels with **-100** instead,
    because PyTorch's ``CrossEntropyLoss`` treats -100 as an *ignore_index*
    and excludes those positions from the loss computation.
    """

    tokenizer: object

    def __call__(self, features: list[dict]) -> dict:
        # Separate labels before padding because the tokenizer's .pad() method
        # doesn't know how to handle our label tensors.
        labels = [feature["labels"] for feature in features]

        # Pad input_ids and attention_mask to the longest sequence in this batch.
        batch = self.tokenizer.pad(
            [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ],
            padding=True,
            return_tensors="pt",
        )

        # Manually pad labels to match the padded input_ids length.
        # -100 tells CrossEntropyLoss to ignore these padding positions,
        # so the model is only penalised for predicting real tokens.
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded_labels.append(label + [-100] * (max_length - len(label)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def create_training_args(config: dict, output_dir: Path) -> TrainingArguments:
    """Build a TrainingArguments instance from the flat training config dict.

    We use ``inspect.signature`` to introspect the installed version of
    TrainingArguments and silently drop any kwargs it doesn't recognise.
    This keeps us compatible across transformers 4.36+ without hard-coding
    version checks — if a parameter was added in a newer release (e.g.
    ``gradient_checkpointing_kwargs`` in 4.36) it simply gets dropped on
    older installs rather than crashing.
    """
    import inspect

    # Normalise report_to into the list format TrainingArguments expects.
    report_target = config.get("report_to", "none")
    report_to = [] if report_target in {"none", "", None} else [report_target]

    kwargs = dict(
        output_dir=str(output_dir),
        logging_dir=str(config["logging_dir"]),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        num_train_epochs=config["num_train_epochs"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        save_total_limit=config["save_total_limit"],
        lr_scheduler_type=config["lr_scheduler_type"],
        optim=config["optim"],
        max_grad_norm=config["max_grad_norm"],
        bf16=config["bf16"],
        tf32=config["tf32"],
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        report_to=report_to,
        gradient_checkpointing=config["gradient_checkpointing"],
        # use_reentrant=False is required for gradient checkpointing with LoRA.
        # The reentrant variant doesn't correctly track gradients through
        # adapter layers, leading to silent correctness bugs.  The non-
        # reentrant implementation (torch.utils.checkpoint with
        # use_reentrant=False) handles this safely at a small overhead cost.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=config["dataloader_num_workers"],
        save_safetensors=config["save_safetensors"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        # We're tracking eval *loss*, where lower is better.
        greater_is_better=False,
        # Disable Trainer's default column removal — our custom collator
        # handles the exact set of keys it needs, and the dataset has already
        # been stripped of raw columns by dataloader.tokenize_splits.
        remove_unused_columns=False,
        run_name=config["run_name"],
    )

    # --- Backwards-compatibility guard ---
    # Introspect the __init__ signature of the installed TrainingArguments
    # class to discover which parameters it actually accepts.  Any kwargs
    # not in the signature are silently dropped.  This lets us target the
    # latest API without crashing on older transformers releases.
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return TrainingArguments(**kwargs)


def build_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_config: dict,
    output_dir: Path,
) -> Trainer:
    """Assemble a HuggingFace Trainer ready for ``.train()``."""
    training_args = create_training_args(training_config, output_dir)
    data_collator = SupervisedDataCollator(tokenizer=tokenizer)
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # processing_class provides the tokenizer reference Trainer needs for
        # saving / logging without relying on the deprecated `tokenizer=` arg.
        processing_class=tokenizer,
        data_collator=data_collator,
    )
