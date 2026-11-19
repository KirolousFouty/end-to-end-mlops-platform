# End-to-End MLOps & Model Monitoring Platform

A local-first, open-source customer-churn platform. It uses a **synthetic telecom dataset** (not real customer data), generated reproducibly with fixed seeds and a temporal signal. The project covers data creation/validation, feature engineering, chronological evaluation, model selection, MLflow tracking, FastAPI serving, PostgreSQL prediction persistence, Prometheus/Grafana operations, drift checks, and approval-gated retraining.

## Quick start

```bash
make setup
source .venv/bin/activate
make data
make train
make test
make serve
```

In a separate terminal, start the service stack (after `make train`, so the image has an approved artifact mounted/built locally):

```bash
docker compose up --build
```

Endpoints: `POST /predict`, `POST /predict/batch`, `GET /health`, `GET /model-info`, and `GET /metrics`.

```bash
curl -X POST http://localhost:8000/predict -H 'content-type: application/json' -d '{"customer_id":"demo","tenure_months":2,"plan_type":"basic","monthly_usage_gb":10,"monthly_bill":40,"late_payments_6m":1,"complaints_6m":2,"service_outages_6m":1,"support_tickets_6m":3,"autopay":0,"contract_months":1}'
```

## Commands

| Command | Outcome |
|---|---|
| `make setup` | Creates local virtualenv and installs dependencies |
| `make data` | Generates 24,000 synthetic, timestamped customers |
| `make train` | Validates, evaluates two models, tracks MLflow run, saves approved artifact |
| `make serve` | Runs FastAPI locally |
| `make monitor` | Produces a deterministic drift report |
| `make simulate-drift` | Creates shifted customer behavior and detects PSI drift |
| `make test` | Runs unit/API/integration-style tests |
| `make down` | Stops the Compose stack and removes volumes |

## Evaluation and governance

The pipeline uses a chronological 70/15/15 train/validation/test split. Logistic regression and a class-balanced random forest are compared using validation PR-AUC; the selected model is re-fit on train+validation and measured on the untouched test period. Reported metrics include precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix and train duration in `models/model_metadata.json`.

`pipelines.retrain.guarded_retrain` first validates new data and requires drift. Its quality gates are PR-AUC ≥ 0.35 and ROC-AUC ≥ 0.60. A passing candidate is still **pending explicit approval** unless called with `approved=True`; it will never automatically replace serving with a worse model.

## Versioning, licenses, and observability

`dvc.yaml` describes data/training stages; initialize with `dvc init` before using `dvc repro`. All dependencies are permissively available open-source projects; see `pyproject.toml` for the exact direct dependency set and `LICENSE` for this repository. MLflow is optional at local train time so a down tracking server does not prevent reproducible training. Prometheus scrapes API request count and latency; Grafana supplies an operational dashboard. Prediction records go to PostgreSQL under Compose (SQLite is used for local development). `make simulate-drift` writes a JSON quality gate and interactive Evidently HTML report in `artifacts/`.

The architecture is documented in [docs/architecture.md](docs/architecture.md).
