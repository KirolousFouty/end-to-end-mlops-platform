from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_SEED

REQUIRED_COLUMNS = {
    "customer_id",
    "event_date",
    "tenure_months",
    "plan_type",
    "monthly_usage_gb",
    "monthly_bill",
    "late_payments_6m",
    "complaints_6m",
    "service_outages_6m",
    "support_tickets_6m",
    "autopay",
    "contract_months",
    "churn",
}


def generate_customers(
    n_customers: int = 24_000, seed: int = RANDOM_SEED, drift: bool = False
) -> pd.DataFrame:
    """Create a deterministic, explicitly synthetic telecom churn dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2025-06-30", periods=n_customers)
    plan = rng.choice(["basic", "standard", "premium"], n_customers, p=[0.38, 0.44, 0.18])
    tenure = rng.integers(0, 96, n_customers)
    usage = np.clip(rng.gamma(4, 5, n_customers) + (plan == "premium") * 15, 0, 150)
    bill_base = np.select([plan == "basic", plan == "standard", plan == "premium"], [29, 55, 85])
    bill = np.round(bill_base + usage * 0.42 + rng.normal(0, 7, n_customers), 2)
    complaints = rng.poisson(0.45 + (tenure < 6) * 0.3, n_customers)
    outages = rng.poisson(0.35, n_customers)
    tickets = rng.poisson(0.7 + complaints * 0.35, n_customers)
    late = rng.poisson(0.4, n_customers)
    autopay = rng.binomial(1, 0.62, n_customers)
    contract = np.where(
        plan == "premium",
        rng.choice([12, 24], n_customers),
        rng.choice([1, 12], n_customers, p=[0.6, 0.4]),
    )
    if drift:
        # A later population: higher usage/bills and more support friction.
        usage = np.clip(usage * 1.35 + 8, 0, 180)
        bill = np.round(bill * 1.18, 2)
        complaints = complaints + rng.binomial(1, 0.28, n_customers)
        outages = outages + rng.binomial(1, 0.20, n_customers)
    logit = (
        -1.9
        + 0.75 * (tenure < 6)
        + 0.38 * late
        + 0.52 * complaints
        + 0.33 * outages
        + 0.18 * tickets
        + 0.42 * (contract == 1)
        - 0.62 * autopay
        + 0.20 * (plan == "basic")
        + (0.25 if drift else 0)
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n_customers)],
            "event_date": dates,
            "tenure_months": tenure,
            "plan_type": plan,
            "monthly_usage_gb": usage.round(2),
            "monthly_bill": bill,
            "late_payments_6m": late,
            "complaints_6m": complaints,
            "service_outages_6m": outages,
            "support_tickets_6m": tickets,
            "autopay": autopay,
            "contract_months": contract,
            "churn": churn,
        }
    )
    # Realistic quality issue for handling, never target leakage.
    frame.loc[rng.choice(n_customers, n_customers // 100, replace=False), "monthly_usage_gb"] = (
        np.nan
    )
    return frame


def validate_data(df: pd.DataFrame, require_target: bool = True) -> list[str]:
    needed = REQUIRED_COLUMNS if require_target else REQUIRED_COLUMNS - {"churn"}
    errors = sorted(needed - set(df.columns))
    if errors:
        return [f"missing required columns: {', '.join(errors)}"]
    issues: list[str] = []
    if df["customer_id"].duplicated().any():
        issues.append("customer_id must be unique")
    if not pd.api.types.is_datetime64_any_dtype(df["event_date"]):
        issues.append("event_date must be datetime")
    if (df["tenure_months"] < 0).any():
        issues.append("tenure_months must be non-negative")
    if (df["monthly_bill"] < 0).any():
        issues.append("monthly_bill must be non-negative")
    if require_target and not set(df["churn"].dropna().unique()).issubset({0, 1}):
        issues.append("churn must be binary")
    return issues
