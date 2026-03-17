"""
Dataset loading and tokenization for causal-LM fine-tuning.

This module does two important things:

1. **Import disambiguation** — The project has a local ``datasets/`` directory
   (containing raw conversation data), which shadows the third-party
   ``datasets`` package from HuggingFace when the project root is on
   ``sys.path``.  ``load_hf_datasets_module()`` works around this by
   temporarily removing the project root from ``sys.path`` so that
   ``import datasets`` resolves to the HuggingFace library, not our local
   folder.

2. **Tokenization** — Converts raw text rows from JSONL files into
   ``input_ids``, ``attention_mask``, and ``labels`` suitable for causal
   language model training (where labels == input_ids, shifted internally
   by the model).
"""
from __future__ import annotations

import sys
from pathlib import Path


def load_hf_datasets_module():
    """Import the HuggingFace ``datasets`` library without colliding with our
    local ``datasets/`` directory.

    The problem: Python's import system checks ``sys.path`` entries in order.
    Because the project root is on ``sys.path`` (so sibling packages like
    ``models/`` and ``training/`` are importable), ``import datasets`` finds
    the local ``datasets/`` *directory* before the installed HuggingFace
    ``datasets`` *package*.

    The fix:
      1. Evict any already-cached ``datasets`` module from ``sys.modules``.
      2. Temporarily strip the project root from ``sys.path``.
      3. Import ``datasets`` — now Python finds the HuggingFace package.
      4. Restore ``sys.path`` to its original state (in a ``finally`` block
         to be safe even if the import fails).
    """
    project_root = Path(__file__).resolve().parents[1]
    original_sys_path = list(sys.path)

    # Clear any cached reference to the wrong (local) datasets module.
    sys.modules.pop("datasets", None)

    # Temporarily remove the project root so Python won't find our local
    # datasets/ directory when resolving the import.
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != project_root
    ]
    try:
        import datasets as hf_datasets
    finally:
        # Always restore the original sys.path, even if the import fails.
        sys.path = original_sys_path
    return hf_datasets


# Run the disambiguation at module load time so that downstream code can
# simply call load_dataset() without worrying about the import hack.
hf_datasets = load_hf_datasets_module()
load_dataset = hf_datasets.load_dataset


def load_training_splits(train_path: Path, validation_path: Path):
    """Load pre-split JSONL files into a HuggingFace DatasetDict.

    Each JSONL row is expected to have at least a ``text`` field containing the
    fully-formatted conversation string (system + user + assistant turns).
    """
    return load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(validation_path),
        },
    )


def tokenize_splits(dataset_dict, tokenizer, max_seq_length: int, num_proc: int | None = None):
    """Tokenize all splits in *dataset_dict* for causal language model training.

    For causal LM fine-tuning the labels are identical to the input_ids: the
    model internally shifts them by one position so that each token predicts
    the *next* token.  We set ``padding=False`` here because padding is
    handled later by the ``SupervisedDataCollator`` on a per-batch basis
    (dynamic padding), which is more memory-efficient than padding every
    sample to ``max_seq_length`` up front.
    """

    def tokenize(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            # No padding here — the data collator pads each batch dynamically
            # to the longest sequence in that batch, saving memory.
            padding=False,
        )
        # For causal LM training, labels == input_ids.  The model's forward
        # pass shifts labels right by one so position i predicts token i+1.
        tokenized["labels"] = [list(input_ids) for input_ids in tokenized["input_ids"]]
        return tokenized

    return dataset_dict.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        # remove_columns drops the original raw columns (e.g. "text") that are
        # no longer needed after tokenization.  Without this, the Dataset would
        # carry the raw strings alongside the tokenized tensors, wasting memory
        # and potentially confusing the Trainer's column-detection heuristics.
        remove_columns=dataset_dict["train"].column_names,
        desc="Tokenizing dataset splits",
    )
