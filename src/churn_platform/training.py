from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .config import METADATA_PATH, MODEL_PATH, RANDOM_SEED
from .features import engineer_features, make_preprocessor


@dataclass
class Evaluation:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]
    training_seconds: float


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("event_date").reset_index(drop=True)
    a, b = int(len(ordered) * 0.7), int(len(ordered) * 0.85)
    return ordered.iloc[:a], ordered.iloc[a:b], ordered.iloc[b:]


def evaluate(model: Pipeline, test: pd.DataFrame, seconds: float = 0.0) -> Evaluation:
    x, y = engineer_features(test), test["churn"]
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return Evaluation(
        float(precision_score(y, pred, zero_division=0)),
        float(recall_score(y, pred, zero_division=0)),
        float(f1_score(y, pred, zero_division=0)),
        float(roc_auc_score(y, proba)),
        float(average_precision_score(y, proba)),
        confusion_matrix(y, pred).tolist(),
        seconds,
    )


def fit_candidates(
    train: pd.DataFrame, val: pd.DataFrame
) -> tuple[Pipeline, str, dict[str, Evaluation]]:
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }
    scores: dict[str, Evaluation] = {}
    fitted: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        model = Pipeline(
            [
                ("preprocess", make_preprocessor(scale=name == "logistic_regression")),
                ("model", estimator),
            ]
        )
        started = time.perf_counter()
        model.fit(engineer_features(train), train["churn"])
        scores[name] = evaluate(model, val, time.perf_counter() - started)
        fitted[name] = model
    best = max(scores, key=lambda n: scores[n].pr_auc)
    return fitted[best], best, scores


def train_and_register(
    df: pd.DataFrame,
    dataset_version: str = "synthetic-v1",
    model_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[Pipeline, dict]:
    model_path = model_path or MODEL_PATH
    metadata_path = metadata_path or METADATA_PATH
    train, val, test = chronological_split(df)
    model, name, validation = fit_candidates(train, val)
    started = time.perf_counter()
    model.fit(engineer_features(pd.concat([train, val])), pd.concat([train, val])["churn"])
    metrics = asdict(evaluate(model, test, time.perf_counter() - started))
    metadata = {
        "name": name,
        "version": "local-approved-v1",
        "dataset_version": dataset_version,
        "split": "chronological 70/15/15",
        "metrics": metrics,
        "validation": {k: asdict(v) for k, v in validation.items()},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    # Tracking stays optional for fast local use; if configured it records the selected run.
    try:
        import mlflow

        mlflow.set_experiment("customer-churn")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {"model": name, "dataset_version": dataset_version, "split": metadata["split"]}
            )
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            mlflow.sklearn.log_model(model, "model")
    except Exception as exc:  # Tracking server must never prevent local reproducibility.
        metadata["mlflow_warning"] = str(exc)
        metadata_path.write_text(json.dumps(metadata, indent=2))
    return model, metadata


def load_approved_model(path: Path = MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError("No approved model. Run `make train` first.")
    return joblib.load(path)
