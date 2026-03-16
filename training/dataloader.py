from __future__ import annotations

import sys
from pathlib import Path


def load_hf_datasets_module():
    project_root = Path(__file__).resolve().parents[1]
    original_sys_path = list(sys.path)
    sys.modules.pop("datasets", None)
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != project_root
    ]
    try:
        import datasets as hf_datasets
    finally:
        sys.path = original_sys_path
    return hf_datasets


hf_datasets = load_hf_datasets_module()
load_dataset = hf_datasets.load_dataset


def load_training_splits(train_path: Path, validation_path: Path):
    return load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(validation_path),
        },
    )


def tokenize_splits(dataset_dict, tokenizer, max_seq_length: int, num_proc: int | None = None):
    def tokenize(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized["labels"] = [list(input_ids) for input_ids in tokenized["input_ids"]]
        return tokenized

    return dataset_dict.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing dataset splits",
    )
