from fastapi.testclient import TestClient
from churn_platform.data import generate_customers
from churn_platform.training import train_and_register


def test_api_prediction(tmp_path, monkeypatch):
    import churn_platform.training as training
    import churn_platform.config as config

    model_path = tmp_path / "model.joblib"
    metadata = tmp_path / "metadata.json"
    monkeypatch.setattr(training, "MODEL_PATH", model_path)
    monkeypatch.setattr(training, "METADATA_PATH", metadata)
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    monkeypatch.setattr(config, "METADATA_PATH", metadata)
    train_and_register(generate_customers(900, seed=5))
    import api.main as main

    monkeypatch.setattr(
        main, "load_approved_model", lambda: training.load_approved_model(model_path)
    )
    with TestClient(main.app) as client:
        payload = (
            generate_customers(1, seed=6)
            .drop(columns=["event_date", "churn"])
            .iloc[0]
            .where(lambda x: x.notna(), None)
            .to_dict()
        )
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert 0 <= response.json()["churn_probability"] <= 1
