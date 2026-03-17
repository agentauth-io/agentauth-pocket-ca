"""Base model loading with optional NF4 quantisation and LoRA adapter
injection.

This module encapsulates two concerns:

1. **Quantised loading** -- Uses ``BitsAndBytesConfig`` to load a large
   causal LM (e.g. Llama 3 8B) in 4-bit NormalFloat (NF4) precision,
   reducing VRAM from ~16 GB (fp16) to ~5 GB while retaining most of
   the model's representational capacity.
2. **LoRA application** -- Wraps the frozen base model with low-rank
   adapter layers via PEFT, so only a small fraction of parameters
   (~1-3%) are trainable during fine-tuning.
"""

from __future__ import annotations

from typing import Iterable

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


# LoRA target modules for the Llama architecture.  These cover every
# linear projection in both the self-attention block (q/k/v/o) and the
# feed-forward network (gate/up/down).  Targeting all linear layers
# gives the adapter maximum expressivity without touching embeddings or
# layer norms.
DEFAULT_TARGET_MODULES = (
    "q_proj",   # query projection in self-attention
    "k_proj",   # key projection in self-attention
    "v_proj",   # value projection in self-attention
    "o_proj",   # output projection in self-attention
    "gate_proj",  # gating projection in SwiGLU FFN
    "up_proj",    # up-projection in SwiGLU FFN
    "down_proj",  # down-projection in SwiGLU FFN
)



# ── dtype resolution ────────────────────────────────────────────────────

def resolve_torch_dtype(dtype_name: str):
    """Map a human-friendly dtype string to a ``torch.dtype`` constant.

    Accepts both short (``"bf16"``) and long (``"bfloat16"``) forms.
    """
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


# ── base model loading ──────────────────────────────────────────────────

def load_base_model(
    model_id: str,
    *,
    torch_dtype: str = "bfloat16",
    use_4bit: bool = True,
):
    """Load a causal LM with optional NF4 4-bit quantisation.

    When ``use_4bit=True``, the model weights are stored in 4-bit
    NormalFloat format (NF4), which is information-theoretically optimal
    for normally distributed weights.  Key config choices:

    * ``bnb_4bit_use_double_quant=True`` -- applies a second round of
      quantisation to the quantisation constants themselves, saving an
      additional ~0.4 bits per parameter.
    * ``bnb_4bit_quant_type="nf4"`` -- NormalFloat4, better than FP4 for
      pre-trained LLM weights which follow a roughly normal distribution.
    * ``bnb_4bit_compute_dtype`` -- the dtype used for *computation*
      (matrix multiplies) even though *storage* is 4-bit.  bfloat16 is
      the default because it matches Llama 3's pre-training dtype.

    ``device_map="auto"`` lets Accelerate shard the model across
    available GPUs (and CPU/disk if needed).
    """
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
    # Disable KV-cache because it is incompatible with gradient
    # checkpointing used during training.  For inference-only paths
    # (eval_utils.py) the cache is re-enabled implicitly by model.eval().
    model.config.use_cache = False
    return model


# ── LoRA adapter ────────────────────────────────────────────────────────

def apply_lora(
    model,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
):
    """Wrap the base model with LoRA (Low-Rank Adaptation) layers.

    Parameters
    ----------
    r : int
        Rank of the low-rank decomposition.  Higher = more expressive
        but more trainable params.  16 is a good default for 8B models.
    alpha : int
        LoRA scaling factor.  The effective learning rate for LoRA
        layers is scaled by ``alpha / r``, so alpha=32 with r=16 gives
        a 2x multiplier.
    dropout : float
        Dropout applied to LoRA layers during training for regularisation.
    bias : str
        Whether to train bias terms.  ``"none"`` keeps them frozen.
    target_modules : Iterable[str]
        Which linear layers to inject LoRA adapters into.

    ``prepare_model_for_kbit_training`` is called first when the model
    is quantised.  This freezes the base weights, casts layer norms to
    float32 for stability, and enables gradient checkpointing to reduce
    VRAM usage during backpropagation.
    """
    if getattr(model, "is_loaded_in_4bit", False) or getattr(
        model, "is_loaded_in_8bit", False
    ):
        # Freeze base weights, cast norms to fp32, enable grad checkpointing
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
