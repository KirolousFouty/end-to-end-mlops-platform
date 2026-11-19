from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DRIFT_THRESHOLD


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref, cur = reference.dropna(), current.dropna()
    if not len(ref) or not len(cur):
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    rp = np.histogram(ref, bins=edges)[0] / len(ref)
    cp = np.histogram(cur, bins=edges)[0] / len(cur)
    rp, cp = np.clip(rp, 1e-6, None), np.clip(cp, 1e-6, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, output: Path | None = None
) -> dict:
    numerical = [
        c
        for c in reference.columns
        if c in current and pd.api.types.is_numeric_dtype(reference[c]) and c != "churn"
    ]
    features = {c: population_stability_index(reference[c], current[c]) for c in numerical}
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": DRIFT_THRESHOLD,
        "feature_psi": features,
        "drifted_features": [k for k, v in features.items() if v > DRIFT_THRESHOLD],
        "drift_detected": any(v > DRIFT_THRESHOLD for v in features.values()),
        "missing_rates": {c: float(current[c].isna().mean()) for c in current.columns},
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
    return result


def save_evidently_report(reference: pd.DataFrame, current: pd.DataFrame, output: Path) -> None:
    """Persist an Evidently interactive report beside the JSON quality-gate artifact."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    output.parent.mkdir(parents=True, exist_ok=True)
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    snapshot.save_html(str(output))
