from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from transformers import AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.base_model_loader import load_base_model, resolve_torch_dtype
from models.tokenizer import load_tokenizer
from serving.agentauth_adapter import DecisionRequest, DecisionResponse, build_prompt, parse_response

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover
    LLM = None
    SamplingParams = None


MODEL_CONFIG_PATH = PROJECT_ROOT / "configs/model.yaml"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "experiments/checkpoints/pocket-ca-v1"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class InferenceEngine:
    def __init__(self, model_path: Path, model_config: dict):
        self.model_path = model_path
        self.model_config = model_config
        self.max_new_tokens = model_config["generation"]["max_new_tokens"]
        self.temperature = model_config["generation"]["temperature"]
        self.top_p = model_config["generation"]["top_p"]
        self.kind = "transformers"
        self.adapter_checkpoint = (model_path / "adapter_config.json").exists()

        use_vllm = (
            os.getenv("POCKET_CA_USE_VLLM", "0") == "1"
            and LLM is not None
            and not self.adapter_checkpoint
        )
        if use_vllm:
            self.kind = "vllm"
            self.llm = LLM(model=str(model_path), dtype=model_config["torch_dtype"])
            self.tokenizer = None
            self.model = None
            return

        tokenizer_source = (
            str(model_path) if (model_path / "tokenizer.json").exists() else model_config["tokenizer_id"]
        )
        self.tokenizer = load_tokenizer(tokenizer_source)
        if self.adapter_checkpoint:
            base_model = load_base_model(
                model_config["base_model_id"],
                torch_dtype=model_config["torch_dtype"],
                use_4bit=model_config["use_4bit"],
            )
            self.model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                torch_dtype=resolve_torch_dtype(model_config["torch_dtype"]),
            )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        if self.kind == "vllm":
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
            )
            outputs = self.llm.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


app = FastAPI(title="AgentAuth Pocket CA Inference API")
engine: InferenceEngine | None = None


@app.on_event("startup")
def startup_event() -> None:
    global engine
    model_config = load_yaml(MODEL_CONFIG_PATH)
    checkpoint_path = Path(os.getenv("POCKET_CA_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    engine = InferenceEngine(checkpoint_path, model_config)


@app.get("/health")
def health() -> dict:
    if engine is None:
        return {"status": "starting"}
    return {
        "status": "ok",
        "engine": engine.kind,
        "model_path": str(engine.model_path),
    }


def run_reasoning(request: DecisionRequest) -> DecisionResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine is not initialized.")
    prompt = build_prompt(request.instruction, request.context)
    parsed = parse_response(engine.generate(prompt))
    return DecisionResponse(
        decision=parsed["decision"],
        explanation=parsed["explanation"],
        confidence=parsed["confidence"],
    )


@app.post("/evaluate_transaction", response_model=DecisionResponse)
def evaluate_transaction(request: DecisionRequest) -> DecisionResponse:
    return run_reasoning(request)


@app.post("/financial_reasoning", response_model=DecisionResponse)
def financial_reasoning(request: DecisionRequest) -> DecisionResponse:
    return run_reasoning(request)


@app.post("/decision", response_model=DecisionResponse)
def decision(request: DecisionRequest) -> DecisionResponse:
    return run_reasoning(request)
