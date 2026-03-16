# AgentAuth Pocket CA

Notebook-first fine-tuning pipeline for a domain-specific financial reasoning
model built on top of `meta-llama/Meta-Llama-3-8B-Instruct`.

The primary entry point is
`agentauth_pocket_ca_pipeline.ipynb`, which walks through:

- synthetic financial dataset generation
- real dataset import for FinQA, ConvFinQA, FinanceBench, and FinR1 style data
- dataset merging and normalization into a unified 100k+ corpus
- preprocessing and train/validation/test splits
- QLoRA fine-tuning with Hugging Face + PEFT
- evaluation for reasoning and budget decisions
- model registry export into `experiments/checkpoints/`
- FastAPI inference serving with optional vLLM support

## Layout

```text
agentauth-pocket-ca/
├── agentauth_pocket_ca_pipeline.ipynb
├── configs/
├── data/
├── datasets/
├── evaluation/
├── experiments/
├── models/
├── scripts/
├── serving/
└── training/
```

## Quick Start

Run these commands on your Lambda GPU box from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
bash scripts/build_dataset.sh 100000
bash scripts/launch_training.sh pocket-ca-v1
python evaluation/eval_reasoning.py \
  --checkpoint experiments/checkpoints/pocket-ca-v1
uvicorn serving.inference_api:app --host 0.0.0.0 --port 8000
```

## Notes

- Accept the Llama 3 model license in Hugging Face before training.
- The training script saves PEFT adapter weights by default.
- `scripts/launch_training.sh <run_name> [dataset_size]` lets you launch
  `pocket-ca-v2`, `pocket-ca-v3`, and similar runs without editing YAML.
- Set `FORCE_REBUILD_DATASET=1` if you want the launcher to rebuild dataset
  splits before training.
- `serving/inference_api.py` can load adapter checkpoints through Transformers.
- vLLM is supported for merged full-model checkpoints. If the checkpoint is a
  PEFT adapter, the server falls back to Transformers automatically.
- Real dataset import is controlled with environment variables:
  `FINQA_PATH`, `CONVFINQA_PATH`, `FINANCEBENCH_PATH`, and `FINR1_PATH`.
- If you need an offline preprocessing smoke test before the tokenizer is
  available locally, run `SKIP_TOKENIZER_VALIDATION=1 bash scripts/build_dataset.sh`.
