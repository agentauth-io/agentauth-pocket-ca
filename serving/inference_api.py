"""
AgentAuth Pocket CA -- FastAPI Inference Server
================================================

Exposes a REST API that wraps the Pocket CA financial-reasoning model.
The server supports two inference backends:

  1. **Transformers** (default) -- works with both LoRA adapter checkpoints
     and fully-merged checkpoints.  Adapter checkpoints are detected
     automatically by the presence of ``adapter_config.json`` in the
     checkpoint directory.
  2. **vLLM** (opt-in) -- activated by setting the environment variable
     ``POCKET_CA_USE_VLLM=1``.  vLLM cannot load LoRA adapters on the fly,
     so this backend is only available for merged checkpoints.

Endpoints
---------
POST /evaluate_transaction  -- evaluate a single financial transaction
POST /financial_reasoning   -- alias for evaluate_transaction
POST /decision              -- generic decision endpoint (same logic)
GET  /health                -- liveness probe; returns engine type and path

Usage
-----
    POCKET_CA_MODEL_PATH=/path/to/checkpoint uvicorn serving.inference_api:app
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Project-root bootstrapping -- ensures sibling packages (models/, pocket_ca/)
# are importable regardless of where the server is launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model_loader import load_base_model, resolve_torch_dtype
from models.tokenizer import load_tokenizer
from serving.agentauth_adapter import DecisionRequest, DecisionResponse, build_prompt, parse_response

# vLLM is an optional high-throughput backend.  If it is not installed the
# server falls back to vanilla transformers without error.
try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover
    LLM = None
    SamplingParams = None

# ---------------------------------------------------------------------------
# Default paths -- overridable via environment variables at startup.
# ---------------------------------------------------------------------------
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs/model.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments/checkpoints/pocket-ca-v1"


def load_yaml(path: Path) -> dict:
    """Read a YAML config file and return its contents as a dict."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class InferenceEngine:
    """Unified inference wrapper that abstracts over transformers and vLLM.

    Checkpoint type detection
    ~~~~~~~~~~~~~~~~~~~~~~~~
    A checkpoint is treated as a **LoRA adapter** when the directory contains
    ``adapter_config.json``.  In that case the base model is loaded first and
    the adapter weights are applied on top via ``PeftModel``.  Otherwise the
    checkpoint is assumed to be a fully-merged model and is loaded directly
    with ``AutoModelForCausalLM``.

    vLLM selection
    ~~~~~~~~~~~~~~
    vLLM is used only when *all three* conditions hold:
      1. ``POCKET_CA_USE_VLLM=1`` is set in the environment,
      2. the ``vllm`` package is installed, and
      3. the checkpoint is **not** a LoRA adapter (vLLM cannot load adapters
         at runtime without a separate adapter-serving workflow).
    """

    def __init__(self, model_path: Path, model_config: dict):
        self.model_path = model_path
        self.model_config = model_config

        # Cache generation hyper-parameters from config for fast access.
        self.max_new_tokens = model_config["generation"]["max_new_tokens"]
        self.temperature = model_config["generation"]["temperature"]
        self.top_p = model_config["generation"]["top_p"]

        self.kind = "transformers"  # default backend; may be overridden below

        # Detect checkpoint type by checking for the PEFT adapter marker file.
        self.adapter_checkpoint = (model_path / "adapter_config.json").exists()

        # --- vLLM backend (early return) ---
        use_vllm = (
            os.getenv("POCKET_CA_USE_VLLM", "0") == "1"
            and LLM is not None
            and not self.adapter_checkpoint  # vLLM can't load raw adapters
        )
        if use_vllm:
            self.kind = "vllm"
            self.llm = LLM(model=str(model_path), dtype=model_config["torch_dtype"])
            # tokenizer / model are managed internally by vLLM; set to None
            # so the rest of the code can safely check ``self.kind``.
            self.tokenizer = None
            self.model = None
            return

        # --- Transformers backend ---
        # Prefer the tokenizer bundled with the checkpoint; fall back to the
        # HuggingFace Hub tokenizer specified in model.yaml.
        tokenizer_source = (
            str(model_path) if (model_path / "tokenizer.json").exists() else model_config["tokenizer_id"]
        )
        self.tokenizer = load_tokenizer(tokenizer_source)

        if self.adapter_checkpoint:
            # Load the base foundation model first, then overlay LoRA weights.
            base_model = load_base_model(
                model_config["base_model_id"],
                torch_dtype=model_config["torch_dtype"],
                use_4bit=model_config["use_4bit"],
            )
            self.model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            # Merged checkpoint -- load everything from a single directory.
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                torch_dtype=resolve_torch_dtype(model_config["torch_dtype"]),
            )

        # Freeze all parameters and disable dropout for deterministic inference.
        self.model.eval()

    def generate(self, prompt: str) -> str:
        """Run a single-sequence generation and return the decoded text.

        The method dispatches to the active backend (vLLM or transformers).
        """

        # ---- vLLM path ----
        if self.kind == "vllm":
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
            )
            outputs = self.llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()

        # ---- Transformers path ----
        # Identify the device the model lives on so we can move input tensors
        # to the same device (important for multi-GPU / CPU setups).
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                # Use greedy decoding when temperature is zero; otherwise
                # sample with the configured temperature and top-p.
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Slice off the prompt tokens so we only decode newly generated text.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# FastAPI application and global engine singleton
# ---------------------------------------------------------------------------
app = FastAPI(title="AgentAuth Pocket CA Inference API")

# ``engine`` is initialised lazily during the startup event so that model
# loading happens *after* the ASGI server has bound the socket, preventing
# long import-time delays.
engine: InferenceEngine | None = None


@app.on_event("startup")
def startup_event() -> None:
    """Load model weights into the global InferenceEngine on server boot.

    The checkpoint path can be overridden with the ``POCKET_CA_MODEL_PATH``
    environment variable; otherwise it defaults to the v1 checkpoint under
    ``experiments/checkpoints/``.
    """
    global engine
    model_config = load_yaml(MODEL_CONFIG_PATH)
    checkpoint_path = Path(os.getenv("POCKET_CA_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    engine = InferenceEngine(checkpoint_path, model_config)


# ---------------------------------------------------------------------------
# Health / liveness probe
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Return server status and active backend information."""
    if engine is None:
        return {"status": "starting"}
    return {
        "status": "ok",
        "engine": engine.kind,
        "model_path": str(engine.model_path),
    }


# ---------------------------------------------------------------------------
# Core reasoning helper -- shared by all POST endpoints
# ---------------------------------------------------------------------------
def run_reasoning(request: DecisionRequest) -> DecisionResponse:
    """Build a prompt from the request, run inference, and parse the output.

    The model's raw text is parsed by ``parse_response`` which extracts the
    decision label, explanation, and confidence.  If the engine has not
    finished loading, a 503 is returned.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized.")
    prompt = build_prompt(request.instruction, request.context)
    parsed = parse_response(engine.generate(prompt))
    return DecisionResponse(
        decision=parsed["decision"],
        explanation=parsed["explanation"],
        confidence=parsed["confidence"],
    )


# ---------------------------------------------------------------------------
# Public POST endpoints -- all route through ``run_reasoning`` so that
# callers can use whichever URL best fits their integration.
# ---------------------------------------------------------------------------
@app.post("/evaluate_transaction", response_model=DecisionResponse)
def evaluate_transaction(request: DecisionRequest) -> DecisionResponse:
    """Evaluate a financial transaction and return approve/reject/review."""
    return run_reasoning(request)


@app.post("/financial_reasoning", response_model=DecisionResponse)
def financial_reasoning(request: DecisionRequest) -> DecisionResponse:
    """Alias for /evaluate_transaction -- kept for backward compatibility."""
    return run_reasoning(request)


@app.post("/decision", response_model=DecisionResponse)
def decision(request: DecisionRequest) -> DecisionResponse:
    """Generic decision endpoint -- same logic as /evaluate_transaction."""
    return run_reasoning(request)
