# AgentAuth Pocket CA Checkpoint

Date: 2026-03-16

## Current Goal

Production-ready financial reasoning fine-tuning pipeline for Lambda GPU training
using QLoRA on `meta-llama/Meta-Llama-3-8B-Instruct`.

## Current State

The subproject at `agentauth-pocket-ca/` now contains:

- synthetic dataset generation at 100k default scale
- real dataset import adapters for FinQA, ConvFinQA, FinanceBench-style, and FinR1-style files
- dataset merging and deduplication
- preprocessing into Llama chat-format instruction data
- `80/10/10` train/validation/test split generation
- resumable QLoRA training with checkpointing and W&B config
- evaluation scripts for reasoning, budget, tax, fraud, and expense classification
- FastAPI inference endpoints for transaction evaluation and financial reasoning

## Base Model

- Base model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Training method: QLoRA / PEFT adapters
- Quantization: 4-bit
- Compute dtype: bf16

## Dataset Scope

Synthetic categories currently include:

- budget reasoning
- expense classification
- fraud detection
- policy compliance
- tax deduction advice
- spending alerts
- investment analysis
- loan evaluation
- financial ratios
- multi-transaction budgeting
- anomaly detection
- corporate finance reasoning

Real dataset ingestion supports local files/directories for:

- FinQA
- ConvFinQA
- FinanceBench-style datasets
- FinR1-style datasets

## Important Files

- `configs/training.yaml`
- `configs/model.yaml`
- `datasets/generate_dataset.py`
- `datasets/import_financial_datasets.py`
- `datasets/merge_datasets.py`
- `datasets/preprocess.py`
- `training/train_lora.py`
- `training/trainer.py`
- `evaluation/eval_reasoning.py`
- `evaluation/eval_budget.py`
- `evaluation/eval_tax_reasoning.py`
- `evaluation/eval_fraud_detection.py`
- `evaluation/eval_expense_classification.py`
- `serving/inference_api.py`
- `scripts/build_dataset.sh`
- `scripts/launch_training.sh`
- `pocket_ca/data_utils.py`
- `pocket_ca/formatting.py`
- `pocket_ca/eval_utils.py`

## Run Flow

### Build dataset only

```bash
cd /home/seyominaoto/Videos/AgentAuth/agentauth-pocket-ca
bash scripts/build_dataset.sh 100000
```

### Build with real datasets

```bash
cd /home/seyominaoto/Videos/AgentAuth/agentauth-pocket-ca
export FINQA_PATH=/path/to/finqa
export CONVFINQA_PATH=/path/to/convfinqa
export FINANCEBENCH_PATH=/path/to/financebench
export FINR1_PATH=/path/to/finr1
bash scripts/build_dataset.sh 100000
```

### Offline preprocessing smoke test

```bash
cd /home/seyominaoto/Videos/AgentAuth/agentauth-pocket-ca
SKIP_TOKENIZER_VALIDATION=1 bash scripts/build_dataset.sh 120
```

### Train

```bash
cd /home/seyominaoto/Videos/AgentAuth/agentauth-pocket-ca
bash scripts/launch_training.sh pocket-ca-v1
```

### Evaluate

```bash
python evaluation/eval_reasoning.py --checkpoint experiments/checkpoints/pocket-ca-v1
python evaluation/eval_budget.py --checkpoint experiments/checkpoints/pocket-ca-v1
python evaluation/eval_tax_reasoning.py --checkpoint experiments/checkpoints/pocket-ca-v1
python evaluation/eval_fraud_detection.py --checkpoint experiments/checkpoints/pocket-ca-v1
python evaluation/eval_expense_classification.py --checkpoint experiments/checkpoints/pocket-ca-v1
```

### Serve

```bash
uvicorn serving.inference_api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `POST /evaluate_transaction`
- `POST /financial_reasoning`
- `POST /decision`

Response schema:

```json
{
  "decision": "approve | reject | review",
  "explanation": "...",
  "confidence": 0.0
}
```

## Validation Already Completed

- shell syntax check passed for `scripts/build_dataset.sh`
- shell syntax check passed for `scripts/launch_training.sh`
- Python bytecode compilation passed for the updated modules
- offline dataset build smoke test passed with `SKIP_TOKENIZER_VALIDATION=1`

## Known Assumptions

- Real dataset import expects local files already downloaded onto the machine.
- FinQA / ConvFinQA / FinanceBench / FinR1 import is schema-tolerant, but if your
  local copy uses a custom field layout, adjust the importer mappings.
- Full training was not run in this environment because the required training
  dependencies and Hugging Face model access are expected on the Lambda GPU box.

## Next Recommended Step

On the Lambda A100 instance:

1. install dependencies from `requirements.txt`
2. run `huggingface-cli login`
3. accept the Llama 3 license if needed
4. place real finance datasets locally
5. run `bash scripts/build_dataset.sh 100000`
6. run `bash scripts/launch_training.sh pocket-ca-v1`
7. launch follow-up runs with `bash scripts/launch_training.sh pocket-ca-v2` and `bash scripts/launch_training.sh pocket-ca-v3`
