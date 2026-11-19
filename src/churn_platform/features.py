from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "tenure_months",
    "monthly_usage_gb",
    "monthly_bill",
    "late_payments_6m",
    "complaints_6m",
    "service_outages_6m",
    "support_tickets_6m",
    "autopay",
    "contract_months",
    "usage_per_dollar",
    "complaint_rate",
]
CATEGORICAL = ["plan_type"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["usage_per_dollar"] = x["monthly_usage_gb"] / x["monthly_bill"].clip(lower=1)
    x["complaint_rate"] = x["complaints_6m"] / (x["tenure_months"] + 1)
    return x[NUMERIC + CATEGORICAL]


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
