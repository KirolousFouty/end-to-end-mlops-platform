# Architecture

```mermaid
flowchart LR
  A[Synthetic temporal data/DVC] --> B[Validation + features]
  B --> C[Prefect training]
  C --> D[MLflow experiments]
  C --> E[Approved model]
  E --> F[FastAPI]
  F --> G[(PostgreSQL)]
  F --> H[Prometheus/Grafana]
  A --> I[Drift PSI report]
  I --> J[Guarded retraining]
  J -->|explicit approval only| E
```

The synthetic generator has a fixed seed and includes 24,000 temporal customer observations. `event_date` determines the 70/15/15 chronological split. Features never use future outcomes. PSI is used for deterministic lightweight drift gates; Evidently can be added as an HTML reporting backend without changing this contract.
