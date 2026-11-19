from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data/raw/customers.csv"
REFERENCE_DATA = ROOT / "data/reference/training_reference.csv"
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models/approved_model.joblib"))
METADATA_PATH = ROOT / "models/model_metadata.json"
ARTIFACTS = ROOT / "artifacts"
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
RANDOM_SEED = 42
