import importlib

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

FEATURES = ["V1", "V2", "Amount"]


@pytest.fixture(scope="module")
def client():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=FEATURES)
    y = (X["V1"] + X["V2"] > 1).astype(int)
    model = xgb.XGBClassifier(n_estimators=20, max_depth=2).fit(X, y)

    patcher = pytest.MonkeyPatch()
    patcher.setattr("src.tracking.load_serving_model", lambda: (model, "test-run"))
    patcher.setattr("src.tracking.load_threshold", lambda run_id=None, default=0.8: 0.5)

    module = importlib.import_module("api.main")
    module = importlib.reload(module)
    yield TestClient(module.app)
    patcher.undo()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["n_features"] == 3


def test_predict_valid_transaction(client):
    response = client.post("/predict", json={"features": {"V1": 2.0, "V2": 2.0, "Amount": 10.0}})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"is_fraud", "fraud_probability", "threshold_applied"}
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_is_independent_of_key_order(client):
    first = client.post("/predict", json={"features": {"V1": 1.5, "V2": -0.3, "Amount": 5.0}})
    second = client.post("/predict", json={"features": {"Amount": 5.0, "V2": -0.3, "V1": 1.5}})
    assert first.json() == second.json()


def test_predict_missing_features_returns_422(client):
    response = client.post("/predict", json={"features": {"V1": 1.0}})
    assert response.status_code == 422
    assert sorted(response.json()["detail"]["missing_features"]) == ["Amount", "V2"]


def test_predict_unexpected_features_returns_422(client):
    payload = {"features": {"V1": 1.0, "V2": 1.0, "Amount": 1.0, "Extra": 3.0}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]["unexpected_features"] == ["Extra"]
