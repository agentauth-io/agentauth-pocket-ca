from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments


@dataclass
class SupervisedDataCollator:
    tokenizer: object

    def __call__(self, features: list[dict]) -> dict:
        labels = [feature["labels"] for feature in features]
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
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded_labels.append(label + [-100] * (max_length - len(label)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def create_training_args(config: dict, output_dir: Path) -> TrainingArguments:
    report_target = config.get("report_to", "none")
    report_to = [] if report_target in {"none", "", None} else [report_target]
    return TrainingArguments(
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
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        report_to=report_to,
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=config["dataloader_num_workers"],
        save_safetensors=config["save_safetensors"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=False,
        remove_unused_columns=False,
        run_name=config["run_name"],
    )


def build_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_config: dict,
    output_dir: Path,
) -> Trainer:
    training_args = create_training_args(training_config, output_dir)
    data_collator = SupervisedDataCollator(tokenizer=tokenizer)
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
