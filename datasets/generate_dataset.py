"""Generate synthetic financial reasoning training data for Pocket CA.

Produces JSONL records across 12 financial categories covering budget
approval, expense classification, fraud detection, policy compliance,
tax deduction advice, spending alerts, investment analysis, loan
evaluation, financial ratio health checks, multi-transaction batching,
anomaly detection, and corporate M&A accretion/dilution analysis.

Each generator function creates one randomised scenario with a structured
instruction, context dict, and a deterministic decision + reasoning
output that the model learns to reproduce.

Usage:
    python datasets/generate_dataset.py --size 100000 --seed 42 \
        --output data/raw/financial_scenarios.jsonl

Inputs:  None (all data is synthetically generated from parameterised
         distributions).
Outputs: A single JSONL file where each line is one training record with
         fields: id, instruction, context, output, category, source.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


# --- Project path setup ---
# Ensure the project root is on sys.path so that `pocket_ca` package
# imports work regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pocket_ca.data_utils import make_output, money, write_jsonl


# --- Reference catalogs ---
# These catalogs define the universe of vendors, departments, purchases,
# and industries that generators sample from. Keeping them at the module
# level lets every generator share a consistent vocabulary and makes it
# easy to extend the domain coverage.
EXPENSE_CATALOG = [
    ("OpenAI API", "software_saas"),
    ("Slack Enterprise Grid", "software_saas"),
    ("AWS GPU reservation", "cloud_infrastructure"),
    ("Datadog annual contract", "monitoring"),
    ("Delta flight", "travel"),
    ("Hilton stay", "travel"),
    ("Google Ads", "marketing"),
    ("LinkedIn recruiting", "recruiting"),
    ("Payroll transfer", "payroll"),
    ("Office Depot", "office_supplies"),
]
# Departments that can be assigned as owners/requestors of an expense.
DEPARTMENTS = ["finance", "engineering", "growth", "operations", "sales"]
# Common high-value purchase types in a tech/startup finance context.
PURCHASES = [
    "AI SaaS subscription",
    "GPU training credits",
    "contractor invoice",
    "travel reimbursement",
    "security audit",
    "annual data vendor renewal",
]
INDUSTRIES = ["fintech", "enterprise software", "healthcare", "ecommerce", "logistics"]


# --- Record construction helper ---

def build_record(
    *,
    record_id: str,
    instruction: str,
    context: dict,
    output: str,
    category: str,
) -> dict:
    """Assemble a single training record in the canonical Pocket CA schema.

    Every record carries a ``source`` tag of ``"synthetic"`` so downstream
    pipelines can distinguish generated data from imported external datasets.
    """
    return {
        "id": record_id,
        "instruction": instruction,
        "context": context,
        "output": output,
        "category": category,
        "source": "synthetic",
    }


# --- Generator: Budget Reasoning ---
# Teaches the model to evaluate a single proposed purchase against a
# department's remaining budget and decide APPROVE or REJECT.

def budget_reasoning(rng: random.Random, index: int) -> dict:
    budget = round(rng.uniform(500, 12_000), 2)
    remaining = round(rng.uniform(0, budget), 2)
    # Cap the purchase amount at 35% of total budget so most scenarios
    # are realistic (departments rarely try to spend the entire budget
    # in one transaction), while still generating some over-budget cases.
    amount = round(rng.uniform(25, max(50, budget * 0.35)), 2)
    purchase = rng.choice(PURCHASES)
    difference = round(remaining - amount, 2)
    if amount <= remaining:
        decision = "APPROVE"
        reason = (
            f"Approve because {money(amount)} fits within the remaining budget of "
            f"{money(remaining)}, leaving {money(difference)} available."
        )
    else:
        decision = "REJECT"
        reason = (
            f"Reject because {money(amount)} exceeds the remaining budget of "
            f"{money(remaining)} by {money(abs(difference))}."
        )
    return build_record(
        record_id=f"budget_reasoning-{index:06d}",
        instruction="Evaluate the transaction against the remaining budget.",
        context={
            "budget": budget,
            "remaining": remaining,
            "purchase": purchase,
            "amount": amount,
            "department": rng.choice(DEPARTMENTS),
        },
        output=make_output(decision, reason),
        category="budget_reasoning",
    )


# --- Generator: Expense Classification ---
# Maps a vendor charge to the correct accounting category (e.g.
# software_saas, travel, marketing). The model learns vendor-to-bucket
# associations used in chart-of-accounts tagging.

def expense_classification(rng: random.Random, index: int) -> dict:
    vendor, category = rng.choice(EXPENSE_CATALOG)
    amount = round(rng.uniform(40, 5_000), 2)
    team = rng.choice(DEPARTMENTS)
    reason = (
        f"Classify the {money(amount)} charge from {vendor} as {category} because "
        f"the vendor and purchase intent match that expense bucket for {team}."
    )
    return build_record(
        record_id=f"expense_classification-{index:06d}",
        instruction="Classify the expense into the correct accounting category.",
        context={
            "vendor": vendor,
            "amount": amount,
            "team": team,
            "memo": f"{vendor} renewal or payment",
        },
        output=make_output(category.upper(), reason),
        category="expense_classification",
    )


# --- Generator: Fraud Detection ---
# Simulates a real-time fraud scoring system. Each boolean signal
# (geo mismatch, new device, etc.) contributes 1 point to a composite
# risk score. The threshold of 4 out of 6 possible signals triggers
# MANUAL_REVIEW; below that the transaction is ALLOWed. This mirrors
# rule-based fraud engines that escalate on signal accumulation.

def fraud_detection(rng: random.Random, index: int) -> dict:
    amount = round(rng.uniform(20, 8_000), 2)
    velocity_24h = rng.randint(1, 9)  # number of transactions in last 24h
    geo_mismatch = rng.choice([True, False])
    new_device = rng.choice([True, False])
    after_hours = rng.choice([True, False])
    card_present = rng.choice([True, False])
    # Composite risk score: each True condition adds 1 point.
    # Six signals total: high velocity (>=5 txns/24h), geo mismatch,
    # new/unrecognised device, transaction outside business hours,
    # high dollar amount (>=$5k), and card-not-present (CNP).
    risk_score = sum(
        [
            velocity_24h >= 5,
            geo_mismatch,
            new_device,
            after_hours,
            amount >= 5_000,
            not card_present,
        ]
    )
    # Threshold of 4 balances false-positive rate against fraud loss;
    # in practice this would be tuned on historical chargeback data.
    if risk_score >= 4:
        decision = "MANUAL_REVIEW"
        reason = (
            f"Escalate for review because the transaction shows {risk_score} "
            "material fraud signals across velocity, geolocation, device trust, or amount."
        )
    else:
        decision = "ALLOW"
        reason = (
            f"Allow because only {risk_score} material fraud signals are present and "
            "the transaction remains below the review threshold."
        )
    return build_record(
        record_id=f"fraud_detection-{index:06d}",
        instruction="Assess the transaction for fraud risk.",
        context={
            "amount": amount,
            "velocity_24h": velocity_24h,
            "geo_mismatch": geo_mismatch,
            "new_device": new_device,
            "after_hours": after_hours,
            "card_present": card_present,
        },
        output=make_output(decision, reason),
        category="fraud_detection",
    )


# --- Generator: Policy Compliance ---
# Checks an expense against a multi-rule company policy: approved vendor
# list, spending limit, manager sign-off for amounts >$1k, and receipt
# requirement for amounts >$75. Any single violation triggers REJECT,
# and all failing reasons are aggregated so the model learns to cite
# every policy breach, not just the first one.

def policy_compliance(rng: random.Random, index: int) -> dict:
    amount = round(rng.uniform(20, 7_500), 2)
    policy_limit = round(rng.uniform(500, 5_000), 2)
    vendor_approved = rng.choice([True, False])
    receipt_attached = rng.choice([True, False])
    # $1,000 threshold for manager approval mirrors common corporate
    # delegation-of-authority matrices.
    manager_approval_required = amount > 1_000
    manager_approved = rng.choice([True, False]) if manager_approval_required else True
    # Accumulate all policy violations so the output enumerates every issue.
    reasons = []
    if not vendor_approved:
        reasons.append("the vendor is not on the approved list")
    if amount > policy_limit:
        reasons.append(
            f"the amount exceeds the policy limit by {money(amount - policy_limit)}"
        )
    if manager_approval_required and not manager_approved:
        reasons.append("manager approval is missing for a high-value purchase")
    # $75 receipt threshold follows IRS accountable-plan guidance for
    # business expenses.
    if amount > 75 and not receipt_attached:
        reasons.append("the receipt is missing")
    if reasons:
        decision = "REJECT"
        reason = "Reject because " + "; ".join(reasons) + "."
    else:
        decision = "APPROVE"
        reason = (
            "Approve because the vendor, limit, receipt, and approval requirements "
            "all satisfy policy."
        )
    return build_record(
        record_id=f"policy_compliance-{index:06d}",
        instruction="Check whether the expense complies with company policy.",
        context={
            "amount": amount,
            "policy_limit": policy_limit,
            "vendor_approved": vendor_approved,
            "receipt_attached": receipt_attached,
            "manager_approval_required": manager_approval_required,
            "manager_approved": manager_approved,
            "expense_type": rng.choice(
                ["software", "travel", "consulting", "office supplies"]
            ),
        },
        output=make_output(decision, reason),
        category="policy_compliance",
    )


# --- Generator: Tax Deduction Advice ---
# Gives a preliminary deductibility recommendation for a US-context
# expense. An expense is considered DEDUCTIBLE only when (a) business-use
# percentage is at least 50%, (b) the expense category is not inherently
# personal (e.g. "family vacation"), and (c) a receipt is available.
# The 50% threshold reflects IRS mixed-use property rules.

def tax_deduction_advice(rng: random.Random, index: int) -> dict:
    expense_type = rng.choice(
        [
            "home office desk",
            "client dinner",
            "business mileage",
            "developer laptop",
            "family vacation flight",  # intentionally personal to generate NON_DEDUCTIBLE cases
        ]
    )
    amount = round(rng.uniform(30, 3_500), 2)
    business_use_percent = rng.randint(0, 100)
    receipt_attached = rng.choice([True, False])
    # "family vacation" is always non-deductible regardless of reported
    # business-use percentage, teaching the model to override the number
    # when the category itself is disqualifying.
    clearly_business = business_use_percent >= 50 and "family vacation" not in expense_type
    if clearly_business and receipt_attached:
        decision = "DEDUCTIBLE"
        reason = (
            f"Treat the {expense_type} expense as deductible because business use is "
            f"{business_use_percent}% and documentation is available."
        )
    else:
        decision = "NON_DEDUCTIBLE"
        reason = (
            f"Do not treat the {expense_type} expense as deductible because business "
            "use is mixed, insufficient, or undocumented."
        )
    return build_record(
        record_id=f"tax_deduction_advice-{index:06d}",
        instruction="Give a preliminary tax deduction recommendation.",
        context={
            "expense_type": expense_type,
            "amount": amount,
            "business_use_percent": business_use_percent,
            "receipt_attached": receipt_attached,
            "jurisdiction": "US",
        },
        output=make_output(decision, reason),
        category="tax_deduction_advice",
    )


# --- Generator: Spending Alert ---
# Determines whether a cost center needs an overspend alert based on
# actual spend, forecasted spend, and budget variance. An alert fires
# when any of three conditions holds: actual > budget, forecast > 105%
# of budget, or variance exceeds 10%. The 5% forecast buffer prevents
# alerts on trivially small projected overruns.

def spending_alert(rng: random.Random, index: int) -> dict:
    budget = round(rng.uniform(2_000, 50_000), 2)
    # actual_spend can exceed budget to generate already-over scenarios.
    actual_spend = round(rng.uniform(1_000, 60_000), 2)
    forecast_spend = round(actual_spend + rng.uniform(-2_500, 8_000), 2)
    variance_pct = round(((forecast_spend - budget) / budget) * 100, 2)
    # Three independent alert triggers — any one is sufficient.
    if actual_spend > budget or forecast_spend > budget * 1.05 or variance_pct > 10:
        decision = "ALERT"
        reason = (
            f"Raise an alert because forecast spend of {money(forecast_spend)} is "
            f"{variance_pct}% versus the budget of {money(budget)}."
        )
    else:
        decision = "NO_ALERT"
        reason = (
            f"No alert is required because forecast spend of {money(forecast_spend)} "
            f"stays close to the budget of {money(budget)}."
        )
    return build_record(
        record_id=f"spending_alerts-{index:06d}",
        instruction="Check whether the account needs a spending alert.",
        context={
            "budget": budget,
            "actual_spend": actual_spend,
            "forecast_spend": forecast_spend,
            "variance_pct": variance_pct,
            "period": rng.choice(["monthly", "quarterly"]),
            "cost_center": rng.choice(["infrastructure", "sales", "g&a"]),
        },
        output=make_output(decision, reason),
        category="spending_alerts",
    )


# --- Generator: Investment Analysis ---
# Scores a company on six fundamental quality factors and maps the
# composite score to BUY / HOLD / AVOID. Thresholds are chosen to
# reflect typical growth-equity screening criteria:
#   - Revenue growth >= 15%  (healthy topline expansion)
#   - Gross margin >= 50%    (asset-light / SaaS-like economics)
#   - Net margin >= 10%      (profitable at scale)
#   - Debt/EBITDA <= 2.5x    (conservative leverage)
#   - EV/Revenue <= peer     (not over-valued vs. comps)
#   - Free cash flow positive (self-funding operations)

def investment_analysis(rng: random.Random, index: int) -> dict:
    revenue_growth = round(rng.uniform(-15, 45), 2)
    gross_margin = round(rng.uniform(20, 85), 2)
    net_margin = round(rng.uniform(-25, 30), 2)
    debt_to_ebitda = round(rng.uniform(0.0, 6.0), 2)
    valuation_multiple = round(rng.uniform(3.0, 20.0), 2)
    peer_multiple = round(rng.uniform(4.0, 16.0), 2)
    free_cash_flow_positive = rng.choice([True, False])
    # Each criterion contributes 1 point; max score is 6.
    score = sum(
        [
            revenue_growth >= 15,
            gross_margin >= 50,
            net_margin >= 10,
            debt_to_ebitda <= 2.5,
            valuation_multiple <= peer_multiple,
            free_cash_flow_positive,
        ]
    )
    # 5+ signals = high-conviction BUY; 3-4 = HOLD; 0-2 = AVOID.
    if score >= 5:
        decision = "BUY"
        reason = (
            "Recommend buy because growth, margins, leverage, cash generation, and "
            "valuation versus peers are all attractive."
        )
    elif score >= 3:
        decision = "HOLD"
        reason = (
            "Recommend hold because the business has mixed fundamentals and does not "
            "clear a high-conviction buy threshold."
        )
    else:
        decision = "AVOID"
        reason = (
            "Avoid because growth, margins, leverage, or valuation are too weak for "
            "a favorable investment case."
        )
    return build_record(
        record_id=f"investment_analysis-{index:06d}",
        instruction="Analyze whether the investment looks attractive.",
        context={
            "industry": rng.choice(INDUSTRIES),
            "revenue_growth_pct": revenue_growth,
            "gross_margin_pct": gross_margin,
            "net_margin_pct": net_margin,
            "debt_to_ebitda": debt_to_ebitda,
            "valuation_multiple_ev_revenue": valuation_multiple,
            "peer_multiple_ev_revenue": peer_multiple,
            "free_cash_flow_positive": free_cash_flow_positive,
        },
        output=make_output(decision, reason),
        category="investment_analysis",
    )


# --- Generator: Loan Evaluation ---
# Simulates a commercial lending underwriting decision. Key thresholds:
#   APPROVE: DSCR >= 1.3x (borrower earns 30% more than debt service),
#            collateral >= 1.2x (sufficient asset backing), FICO >= 680
#            (prime credit), and zero recent delinquencies.
#   REJECT:  DSCR < 1.0x (cannot cover debt payments), FICO < 620
#            (subprime), 2+ delinquencies, or leverage > 4.5x.
#   REVIEW:  Everything else — borderline credits requiring human judgment.

def loan_evaluation(rng: random.Random, index: int) -> dict:
    loan_amount = round(rng.uniform(25_000, 1_500_000), 2)
    dscr = round(rng.uniform(0.7, 2.4), 2)            # Debt Service Coverage Ratio
    collateral_coverage = round(rng.uniform(0.5, 2.2), 2)  # loan-to-value inverse
    fico = rng.randint(540, 820)
    delinquencies = rng.randint(0, 4)
    leverage_ratio = round(rng.uniform(1.0, 6.0), 2)
    # Strict approval gate: all four criteria must pass simultaneously.
    if dscr >= 1.3 and collateral_coverage >= 1.2 and fico >= 680 and delinquencies == 0:
        decision = "APPROVE"
        reason = (
            "Approve because debt service coverage, collateral support, and credit "
            "quality meet lending thresholds."
        )
    # Hard reject gate: any single disqualifying signal is enough.
    elif dscr < 1.0 or fico < 620 or delinquencies >= 2 or leverage_ratio > 4.5:
        decision = "REJECT"
        reason = (
            "Reject because repayment coverage, credit history, or leverage falls "
            "outside acceptable lending policy."
        )
    else:
        decision = "REVIEW"
        reason = (
            "Send for review because the loan shows mixed underwriting signals and "
            "needs analyst judgment."
        )
    return build_record(
        record_id=f"loan_evaluation-{index:06d}",
        instruction="Evaluate whether the company loan should be approved.",
        context={
            "loan_amount": loan_amount,
            "dscr": dscr,
            "collateral_coverage": collateral_coverage,
            "fico_score": fico,
            "recent_delinquencies": delinquencies,
            "leverage_ratio": leverage_ratio,
        },
        output=make_output(decision, reason),
        category="loan_evaluation",
    )


# --- Generator: Financial Ratios ---
# Evaluates a company's financial health by checking five standard
# accounting ratios against conservative "healthy" thresholds:
#   - Current ratio >= 1.2  (can cover short-term liabilities)
#   - Quick ratio >= 1.0    (liquid assets alone cover current liabilities)
#   - Debt/Equity <= 1.5    (not over-leveraged)
#   - Interest coverage >= 3x (earnings comfortably service debt)
#   - Operating margin >= 8% (sustainably profitable from operations)
# All five must pass for HEALTHY; failure on any one yields DISTRESSED.

def financial_ratios(rng: random.Random, index: int) -> dict:
    current_ratio = round(rng.uniform(0.6, 3.0), 2)
    quick_ratio = round(rng.uniform(0.4, 2.6), 2)
    debt_to_equity = round(rng.uniform(0.1, 3.5), 2)
    interest_coverage = round(rng.uniform(0.3, 10.0), 2)
    operating_margin = round(rng.uniform(-12, 35), 2)
    # Conjunctive test — every ratio must clear its threshold.
    if (
        current_ratio >= 1.2
        and quick_ratio >= 1.0
        and debt_to_equity <= 1.5
        and interest_coverage >= 3.0
        and operating_margin >= 8
    ):
        decision = "HEALTHY"
        reason = (
            "The ratios indicate strong liquidity, manageable leverage, adequate "
            "interest coverage, and healthy profitability."
        )
    else:
        decision = "DISTRESSED"
        reason = (
            "The ratio profile indicates liquidity or leverage stress that merits "
            "caution."
        )
    return build_record(
        record_id=f"financial_ratios-{index:06d}",
        instruction="Assess the company's financial ratio health.",
        context={
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "debt_to_equity": debt_to_equity,
            "interest_coverage": interest_coverage,
            "operating_margin_pct": operating_margin,
        },
        output=make_output(decision, reason),
        category="financial_ratios",
    )


def multi_transaction_budgeting(rng: random.Random, index: int) -> dict:
    budget = round(rng.uniform(5_000, 60_000), 2)
    committed_spend = round(rng.uniform(0, budget * 0.8), 2)
    transaction_count = rng.randint(2, 5)
    transactions = []
    proposed_total = 0.0
    for _ in range(transaction_count):
        amount = round(rng.uniform(150, budget * 0.18), 2)
        proposed_total += amount
        transactions.append(
            {
                "vendor": rng.choice([vendor for vendor, _ in EXPENSE_CATALOG]),
                "amount": amount,
                "type": rng.choice(["software", "travel", "consulting", "marketing"]),
            }
        )
    remaining = round(budget - committed_spend, 2)
    if proposed_total <= remaining:
        decision = "APPROVE"
        reason = (
            f"Approve because the proposed batch total of {money(proposed_total)} fits "
            f"inside the remaining budget of {money(remaining)}."
        )
    else:
        decision = "REJECT"
        reason = (
            f"Reject because the transaction batch totals {money(proposed_total)}, "
            f"which exceeds the remaining budget of {money(remaining)}."
        )
    return build_record(
        record_id=f"multi_transaction_budgeting-{index:06d}",
        instruction="Decide whether the batch of transactions can be approved together.",
        context={
            "budget": budget,
            "committed_spend": committed_spend,
            "remaining_budget": remaining,
            "transactions": transactions,
            "proposed_total": round(proposed_total, 2),
        },
        output=make_output(decision, reason),
        category="multi_transaction_budgeting",
    )


def anomaly_detection(rng: random.Random, index: int) -> dict:
    baseline_mean = round(rng.uniform(200, 5_000), 2)
    baseline_std = round(rng.uniform(20, max(50, baseline_mean * 0.35)), 2)
    observed_amount = round(rng.uniform(20, baseline_mean * 4.0), 2)
    z_score = round((observed_amount - baseline_mean) / max(baseline_std, 1.0), 2)
    recurring_vendor = rng.choice([True, False])
    if abs(z_score) >= 2.5 and not recurring_vendor:
        decision = "OUTLIER"
        reason = (
            f"Flag as an outlier because the transaction is {z_score} standard "
            "deviations from the historical average and does not match a recurring pattern."
        )
    else:
        decision = "NORMAL"
        reason = (
            f"Treat as normal because the observed amount sits within an acceptable "
            f"range of the baseline distribution at a z-score of {z_score}."
        )
    return build_record(
        record_id=f"anomaly_detection-{index:06d}",
        instruction="Determine whether the transaction is anomalous.",
        context={
            "baseline_mean_amount": baseline_mean,
            "baseline_std_amount": baseline_std,
            "observed_amount": observed_amount,
            "z_score": z_score,
            "recurring_vendor": recurring_vendor,
        },
        output=make_output(decision, reason),
        category="anomaly_detection",
    )


def corporate_finance_reasoning(rng: random.Random, index: int) -> dict:
    purchase_price = round(rng.uniform(10_000_000, 250_000_000), 2)
    expected_synergies = round(rng.uniform(500_000, 20_000_000), 2)
    target_ebitda = round(rng.uniform(1_000_000, 30_000_000), 2)
    financing_cost = round(rng.uniform(3.0, 12.0), 2)
    dilution_pct = round(rng.uniform(0.0, 18.0), 2)
    leverage_post_close = round(rng.uniform(1.2, 6.0), 2)
    accretion_proxy = expected_synergies + target_ebitda * 0.2 - purchase_price * (
        financing_cost / 100
    )
    if accretion_proxy > 0 and dilution_pct <= 8 and leverage_post_close <= 4.0:
        decision = "ACCRETIVE"
        reason = (
            "The transaction looks accretive because expected synergies and EBITDA "
            "support outweigh financing drag without excessive dilution or leverage."
        )
    elif accretion_proxy < 0 or leverage_post_close > 5.0 or dilution_pct > 12:
        decision = "DILUTIVE"
        reason = (
            "The transaction looks dilutive because financing burden, dilution, or "
            "post-close leverage overwhelms the operating benefits."
        )
    else:
        decision = "REVIEW"
        reason = (
            "The transaction needs review because the accretion case is borderline "
            "and depends on execution assumptions."
        )
    return build_record(
        record_id=f"corporate_finance_reasoning-{index:06d}",
        instruction="Assess whether the acquisition is financially attractive.",
        context={
            "purchase_price": purchase_price,
            "expected_synergies": expected_synergies,
            "target_ebitda": target_ebitda,
            "financing_cost_pct": financing_cost,
            "equity_dilution_pct": dilution_pct,
            "post_close_leverage": leverage_post_close,
        },
        output=make_output(decision, reason),
        category="corporate_finance_reasoning",
    )


GENERATORS = [
    budget_reasoning,
    expense_classification,
    fraud_detection,
    policy_compliance,
    tax_deduction_advice,
    spending_alert,
    investment_analysis,
    loan_evaluation,
    financial_ratios,
    multi_transaction_budgeting,
    anomaly_detection,
    corporate_finance_reasoning,
]


def generate_samples(size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for index in range(size):
        generator = GENERATORS[index % len(GENERATORS)]
        samples.append(generator(rng, index))
    rng.shuffle(samples)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Pocket CA synthetic data.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/financial_scenarios.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100_000,
        help="Number of synthetic instruction samples to generate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = generate_samples(size=args.size, seed=args.seed)
    write_jsonl(args.output, samples)
    print(f"Wrote {len(samples)} records to {args.output}")


if __name__ == "__main__":
    main()
