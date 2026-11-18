"""API tests. Trains a small throwaway model if no trained model is present (e.g. in CI)."""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

MODEL = Path("models/churn_model.joblib")

EXAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
    "TotalCharges": 171.0,
}
LOYAL = {**EXAMPLE, "tenure": 70, "Contract": "Two year", "InternetService": "DSL",
         "OnlineSecurity": "Yes", "TechSupport": "Yes", "PaymentMethod": "Credit card (automatic)",
         "MonthlyCharges": 60.0, "TotalCharges": 4200.0, "PaperlessBilling": "No"}


@pytest.fixture(scope="module")
def client():
    if not MODEL.exists():
        from churn import train
        train.main()
    os.environ["MODEL_PATH"] = str(MODEL)
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_high_risk_customer(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["recommend_retention_offer"] is True


def test_loyal_customer_scores_lower(client):
    risky = client.post("/predict", json=EXAMPLE).json()["churn_probability"]
    loyal = client.post("/predict", json=LOYAL).json()["churn_probability"]
    assert loyal < risky
    assert client.post("/predict", json=LOYAL).json()["risk_band"] == "low"


def test_invalid_payload_rejected(client):
    bad = {**EXAMPLE, "Contract": "Lifetime"}
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "tenure": -1}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"customers": [EXAMPLE, LOYAL]})
    assert r.status_code == 200
    assert r.json()["n"] == 2
