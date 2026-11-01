import pytest
from fastapi.testclient import TestClient

from churnlens.api import app
from churnlens.train import train

VALID = {
    "tenure": 12, "MonthlyCharges": 70.5, "gender": "Female",
    "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("model")
    model_path = tmp / "churn.joblib"
    train(out_path=model_path, scored_path=tmp / "scored.csv")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CHURN_MODEL_PATH", str(model_path))
        with TestClient(app) as c:
            yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_score_valid(client):
    r = client.post("/score", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert 1 <= body["decile"] <= 10


def test_score_negative_tenure_returns_422(client):
    r = client.post("/score", json={**VALID, "tenure": -1})
    assert r.status_code == 422


def test_score_unknown_contract_returns_422(client):
    r = client.post("/score", json={**VALID, "Contract": "Weekly"})
    assert r.status_code == 422
