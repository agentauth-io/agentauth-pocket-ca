"""Tokenizer loading with causal-LM-safe defaults.

This thin wrapper ensures two invariants that are easy to forget when
using HuggingFace tokenizers for causal language model fine-tuning:

1. **``padding_side="right"``** -- Causal (auto-regressive) models
   attend to tokens left-to-right.  If padding were on the left, the
   positional embeddings for real tokens would shift between padded and
   unpadded sequences, degrading quality.  Right-padding keeps the
   prompt tokens at consistent positions and places pads where the
   model's causal mask already blocks attention.

2. **``pad_token`` fallback** -- Many decoder-only tokenizers (GPT-2,
   Llama, etc.) ship without a dedicated pad token.  Setting it to
   ``eos_token`` is the standard workaround; the training loss mask
   already ignores pad positions, so reusing EOS does not interfere
   with learning.
"""

from __future__ import annotations

from transformers import AutoTokenizer


def load_tokenizer(tokenizer_id: str):
    """Load a fast tokenizer with causal-LM-safe padding defaults."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        use_fast=True,
        # Right-padding is critical for causal LMs -- see module docstring.
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        # Fall back to eos_token so the Trainer can batch sequences of
        # different lengths.  Loss masking ensures pad positions do not
        # contribute to the training objective.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
