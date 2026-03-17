"""
test_model.py — Interactive evaluation of a trained Pocket CA checkpoint.

Loads the fine-tuned LoRA adapter, runs a set of representative financial
reasoning prompts, and prints the model's Decision/Reason output.

Usage:
    python evaluation/test_model.py --checkpoint experiments/checkpoints/pocket-ca-v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Project path setup — ensure imports work regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pocket_ca.eval_utils import generate_response, load_model_bundle
from pocket_ca.formatting import build_prompt


# ---------------------------------------------------------------------------
# Test scenarios — each is a (instruction, context, expected_decision) tuple
# covering the key financial reasoning categories.
# ---------------------------------------------------------------------------
TEST_SCENARIOS = [
    {
        "name": "Budget Reasoning — Over Budget",
        "instruction": "Evaluate the transaction against the remaining budget.",
        "context": {
            "budget": 500,
            "remaining": 80,
            "purchase": "AI SaaS subscription",
            "amount": 120,
            "department": "engineering",
        },
        "expected": "REJECT",
    },
    {
        "name": "Budget Reasoning — Within Budget",
        "instruction": "Evaluate the transaction against the remaining budget.",
        "context": {
            "budget": 5000,
            "remaining": 3200,
            "purchase": "contractor invoice",
            "amount": 1500,
            "department": "operations",
        },
        "expected": "APPROVE",
    },
    {
        "name": "Fraud Detection — High Risk",
        "instruction": "Assess the transaction for fraud risk.",
        "context": {
            "amount": 6500.00,
            "velocity_24h": 7,
            "geo_mismatch": True,
            "new_device": True,
            "after_hours": True,
            "card_present": False,
        },
        "expected": "MANUAL_REVIEW",
    },
    {
        "name": "Fraud Detection — Low Risk",
        "instruction": "Assess the transaction for fraud risk.",
        "context": {
            "amount": 45.00,
            "velocity_24h": 1,
            "geo_mismatch": False,
            "new_device": False,
            "after_hours": False,
            "card_present": True,
        },
        "expected": "ALLOW",
    },
    {
        "name": "Expense Classification",
        "instruction": "Classify the expense into the correct accounting category.",
        "context": {
            "vendor": "Delta flight",
            "amount": 890.00,
            "team": "sales",
            "memo": "Delta flight for client visit",
        },
        "expected": "TRAVEL",
    },
    {
        "name": "Policy Compliance — Missing Receipt",
        "instruction": "Check whether the expense complies with company policy.",
        "context": {
            "amount": 250.00,
            "policy_limit": 500.00,
            "vendor_approved": True,
            "receipt_attached": False,
            "manager_approval_required": False,
            "manager_approved": True,
            "expense_type": "office supplies",
        },
        "expected": "REJECT",
    },
    {
        "name": "Tax Deduction — Business Expense",
        "instruction": "Give a preliminary tax deduction recommendation.",
        "context": {
            "expense_type": "developer laptop",
            "amount": 2400.00,
            "business_use_percent": 95,
            "receipt_attached": True,
            "jurisdiction": "US",
        },
        "expected": "DEDUCTIBLE",
    },
    {
        "name": "Loan Evaluation — Strong Application",
        "instruction": "Evaluate whether the company loan should be approved.",
        "context": {
            "loan_amount": 500000.00,
            "dscr": 1.8,
            "collateral_coverage": 1.5,
            "fico_score": 750,
            "recent_delinquencies": 0,
            "leverage_ratio": 2.1,
        },
        "expected": "APPROVE",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a trained Pocket CA checkpoint with example prompts."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained adapter checkpoint directory.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs/model.yaml",
        help="Model config YAML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load model + tokenizer from checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, model_config = load_model_bundle(
        args.checkpoint, args.model_config
    )
    max_new_tokens = model_config["generation"]["max_new_tokens"]
    print("Model loaded.\n")

    # Run each test scenario
    correct = 0
    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        print(f"{'=' * 60}")
        print(f"Test {i}: {scenario['name']}")
        print(f"{'=' * 60}")

        # Build prompt in Llama 3 chat format and generate
        prompt = build_prompt(scenario["instruction"], scenario["context"])
        response = generate_response(model, tokenizer, prompt, max_new_tokens)

        print(f"Response:\n  {response}")
        print(f"Expected decision: {scenario['expected']}")

        # Check if the expected decision appears in the response
        if scenario["expected"] in response.upper():
            print("Result: PASS")
            correct += 1
        else:
            print("Result: FAIL")
        print()

    # Summary
    total = len(TEST_SCENARIOS)
    print(f"{'=' * 60}")
    print(f"Results: {correct}/{total} passed ({correct/total:.0%})")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
