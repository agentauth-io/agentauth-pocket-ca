from __future__ import annotations

from typing import Iterable

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def resolve_torch_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def load_base_model(
    model_id: str,
    *,
    torch_dtype: str = "bfloat16",
    use_4bit: bool = True,
):
    dtype = resolve_torch_dtype(torch_dtype)
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    return model


def apply_lora(
    model,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
):
    if getattr(model, "is_loaded_in_4bit", False) or getattr(
        model, "is_loaded_in_8bit", False
    ):
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=list(target_modules),
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)
