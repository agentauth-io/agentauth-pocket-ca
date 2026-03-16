from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import PeftModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model_loader import load_base_model
from models.tokenizer import load_tokenizer
from pocket_ca.formatting import build_prompt, parse_response


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def default_report_path(checkpoint: Path, report_name: str) -> Path:
    return PROJECT_ROOT / "experiments/metrics" / checkpoint.name / f"{report_name}.json"


def load_model_bundle(checkpoint: Path, model_config_path: Path) -> tuple[Any, Any, dict]:
    model_config = load_yaml(model_config_path)
    tokenizer_source = (
        str(checkpoint) if (checkpoint / "tokenizer.json").exists() else model_config["tokenizer_id"]
    )
    tokenizer = load_tokenizer(tokenizer_source)
    base_model = load_base_model(
        model_config["base_model_id"],
        torch_dtype=model_config["torch_dtype"],
        use_4bit=model_config["use_4bit"],
    )
    model = PeftModel.from_pretrained(base_model, str(checkpoint))
    model.eval()
    return model, tokenizer, model_config


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def predict_record(model, tokenizer, model_config: dict, record: dict) -> tuple[str, dict]:
    raw_response = generate_response(
        model,
        tokenizer,
        prompt=build_prompt(record["instruction"], record["context"]),
        max_new_tokens=model_config["generation"]["max_new_tokens"],
    )
    return raw_response, parse_response(raw_response)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())
