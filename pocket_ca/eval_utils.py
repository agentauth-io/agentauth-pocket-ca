"""Evaluation utilities for running inference with a fine-tuned Pocket CA
checkpoint and parsing the results.

The primary workflow is:

1. Load the quantised base model (e.g. Llama 3 8B in NF4).
2. Overlay the LoRA adapter weights from a training checkpoint using
   ``PeftModel.from_pretrained``.
3. Run greedy decoding (``do_sample=False``) on each test record to
   produce deterministic, reproducible outputs.
4. Parse each raw response into a structured dict (decision, reason,
   confidence) via ``parse_response``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel


# Ensure the project root is on sys.path so that sibling packages
# (``models``, ``pocket_ca``) can be imported regardless of how this
# module is invoked (notebook, script, pytest, etc.).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model_loader import load_base_model
from models.tokenizer import load_tokenizer
from pocket_ca.formatting import build_prompt, parse_response



# ── I/O helpers ─────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file (model config, training config, etc.)."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSON-Lines file; blank lines are silently skipped."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def default_report_path(checkpoint: Path, report_name: str) -> Path:
    """Build the conventional path for saving evaluation metric reports."""
    return PROJECT_ROOT / "experiments/metrics" / checkpoint.name / f"{report_name}.json"


# ── model loading ───────────────────────────────────────────────────────

def load_model_bundle(checkpoint: Path, model_config_path: Path) -> tuple[Any, Any, dict]:
    """Load tokenizer + base model + LoRA adapter in one call.

    The procedure:
    1. Parse the YAML model config for base_model_id, dtype, etc.
    2. Load the tokenizer -- preferring the checkpoint's own copy if it
       saved one, otherwise falling back to the config's ``tokenizer_id``.
    3. Load the base model with optional NF4 quantisation.
    4. Overlay the LoRA adapter weights from the checkpoint directory
       using ``PeftModel.from_pretrained``, which merges the low-rank
       delta matrices on top of the frozen base weights.
    5. Switch to eval mode (disables dropout in LoRA layers).
    """
    model_config = load_yaml(model_config_path)
    # Prefer checkpoint-local tokenizer if available (it may have been
    # extended with new special tokens during training).
    tokenizer_source = (
        str(checkpoint) if (checkpoint / "tokenizer.json").exists() else model_config["tokenizer_id"]
    )
    tokenizer = load_tokenizer(tokenizer_source)
    base_model = load_base_model(
        model_config["base_model_id"],
        torch_dtype=model_config["torch_dtype"],
        use_4bit=model_config["use_4bit"],
    )
    # Apply the LoRA adapter on top of the quantised base model
    model = PeftModel.from_pretrained(base_model, str(checkpoint))
    model.eval()  # disable dropout for deterministic inference
    return model, tokenizer, model_config


# ── inference ───────────────────────────────────────────────────────────

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Run greedy decoding to generate the model's response.

    ``do_sample=False`` ensures deterministic output -- the same prompt
    always produces the same response, which is essential for
    reproducible evaluation metrics.  No temperature or top-p sampling
    is applied.

    Only the *newly generated* tokens are decoded; the input prompt
    tokens are sliced off so the caller gets just the model's answer.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding for reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )
    # Slice off the input prompt tokens to isolate the generated response
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def predict_record(model, tokenizer, model_config: dict, record: dict) -> tuple[str, dict]:
    """End-to-end prediction: build prompt -> generate -> parse.

    Returns both the raw text (for logging / debugging) and the parsed
    structured dict (for metric computation).
    """
    raw_response = generate_response(
        model,
        tokenizer,
        prompt=build_prompt(record["instruction"], record["context"]),
        max_new_tokens=model_config["generation"]["max_new_tokens"],
    )
    return raw_response, parse_response(raw_response)


# ── text helpers ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace.  Used for fuzzy text comparison
    in evaluation scripts (e.g. checking if predicted reasons overlap
    with ground-truth reasons).
    """
    return " ".join(text.lower().split())
