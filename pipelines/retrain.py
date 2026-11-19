import pandas as pd
from prefect import flow, task
from churn_platform.config import ARTIFACTS, METADATA_PATH, MODEL_PATH, RAW_DATA, REFERENCE_DATA
from churn_platform.data import validate_data
from churn_platform.monitoring import drift_report
from churn_platform.training import train_and_register


@task
def validate(df: pd.DataFrame) -> None:
    errors = validate_data(df)
    if errors:
        raise ValueError(errors)


def quality_gate(metrics: dict[str, float]) -> bool:
    """Minimum candidate quality required before a human can promote it."""
    return metrics["pr_auc"] >= 0.35 and metrics["roc_auc"] >= 0.60


@flow(name="guarded-churn-retraining")
def guarded_retrain(data_path: str = str(RAW_DATA), approved: bool = False) -> dict:
    """The serving artifact is replaced only with explicit approval."""
    df = pd.read_csv(data_path, parse_dates=["event_date"])
    validate(df)
    ref = pd.read_csv(REFERENCE_DATA) if REFERENCE_DATA.exists() else df
    drift = drift_report(ref, df, ARTIFACTS / "retrain_drift_report.json")
    if not drift["drift_detected"]:
        return {"decision": "no_retrain", "drift": drift}
    candidate_model = ARTIFACTS / "candidate_model.joblib"
    candidate_metadata = ARTIFACTS / "candidate_metadata.json"
    _, candidate = train_and_register(
        df,
        dataset_version="retraining-candidate",
        model_path=candidate_model,
        metadata_path=candidate_metadata,
    )
    passes = quality_gate(candidate["metrics"])
    if passes and approved:
        MODEL_PATH.write_bytes(candidate_model.read_bytes())
        METADATA_PATH.write_bytes(candidate_metadata.read_bytes())
    return {
        "decision": "approved_for_serving"
        if (passes and approved)
        else "candidate_pending_approval"
        if passes
        else "rejected_quality_gate",
        "candidate": candidate,
        "drift": drift,
    }


if __name__ == "__main__":
    print(guarded_retrain())
