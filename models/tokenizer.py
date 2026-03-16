from __future__ import annotations

from transformers import AutoTokenizer


def load_tokenizer(tokenizer_id: str):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        use_fast=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
