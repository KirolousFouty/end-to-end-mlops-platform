from __future__ import annotations
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from churn_platform.config import METADATA_PATH
from churn_platform.features import engineer_features
from churn_platform.training import load_approved_model

REQUESTS = Counter("churn_prediction_requests_total", "Prediction requests", ["endpoint"])
LATENCY = Histogram("churn_prediction_latency_seconds", "Prediction latency", ["endpoint"])
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///./predictions.db"))
Session = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64))
    prediction: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class Customer(BaseModel):
    customer_id: str = "ad-hoc"
    tenure_months: int = Field(ge=0)
    plan_type: str
    monthly_usage_gb: float | None = Field(default=None, ge=0)
    monthly_bill: float = Field(ge=0)
    late_payments_6m: int = Field(ge=0)
    complaints_6m: int = Field(ge=0)
    service_outages_6m: int = Field(ge=0)
    support_tickets_6m: int = Field(ge=0)
    autopay: int = Field(ge=0, le=1)
    contract_months: int = Field(ge=1)


class Prediction(BaseModel):
    customer_id: str
    churn_prediction: int
    churn_probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    app.state.model = load_approved_model()
    yield


app = FastAPI(title="Churn Model Serving", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


def predict_rows(rows: list[Customer], endpoint: str) -> list[Prediction]:
    started = time.perf_counter()
    REQUESTS.labels(endpoint).inc(len(rows))
    try:
        frame = pd.DataFrame([r.model_dump() for r in rows])
        probabilities = app.state.model.predict_proba(engineer_features(frame))[:, 1]
        output = [
            Prediction(
                customer_id=r.customer_id,
                churn_prediction=int(p >= 0.5),
                churn_probability=float(p),
            )
            for r, p in zip(rows, probabilities)
        ]
        with Session.begin() as session:
            session.add_all(
                [
                    PredictionRecord(
                        customer_id=x.customer_id,
                        prediction=x.churn_prediction,
                        probability=x.churn_probability,
                    )
                    for x in output
                ]
            )
        return output
    finally:
        LATENCY.labels(endpoint).observe(time.perf_counter() - started)


@app.post("/predict", response_model=Prediction)
def predict(row: Customer) -> Prediction:
    return predict_rows([row], "predict")[0]


@app.post("/predict/batch", response_model=list[Prediction])
def batch_predict(rows: list[Customer]) -> list[Prediction]:
    if not rows:
        raise HTTPException(422, "batch must not be empty")
    return predict_rows(rows, "batch")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    if not METADATA_PATH.exists():
        raise HTTPException(503, "model metadata unavailable")
    return json.loads(METADATA_PATH.read_text())
