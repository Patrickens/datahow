"""Tests for the inference microservice.

The endpoint tests use a **mocked** predictor (via FastAPI ``dependency_overrides``
and ``app.state``), so they exercise the API/validation/conversion layers without
running real — and expensive — model inference. Payloads are built from the real
column schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from titer_prediction import schema
from titer_prediction.service.app import app, get_predictor
from titer_prediction.service.dto import PredictRequest
from titer_prediction.service.predictor import payload_to_frame


class MockPredictor:
    """A Predictor that returns a constant, so tests need no real inference."""

    model_type = "mock"
    target = schema.TARGET_COL
    expected_static = schema.EXPECTED_STATIC_COLS
    expected_control = schema.EXPECTED_CONTROL_COLS
    expected_state = schema.EXPECTED_STATE_COLS

    def predict_frame(self, df: pd.DataFrame) -> pd.Series:
        exps = df[schema.EXP_COL].unique()
        return pd.Series([1234.5] * len(exps), index=exps, name=schema.TARGET_COL)


@pytest.fixture
def mock_predictor():
    return MockPredictor()


@pytest.fixture
def client(mock_predictor):
    # No `with` block -> lifespan (which would load the real model) does not run;
    # we inject the mock directly.
    app.state.predictor = mock_predictor
    app.dependency_overrides[get_predictor] = lambda: mock_predictor
    yield TestClient(app)
    app.dependency_overrides.clear()
    app.state.predictor = None


@pytest.fixture
def valid_payload() -> dict:
    """A structurally valid, full-schema payload with 3 timepoints."""
    values: dict[str, list[float]] = {}
    for col in schema.EXPECTED_STATIC_COLS:
        values[col] = [1.0]  # single-element Z:
    for col in (*schema.EXPECTED_CONTROL_COLS, *schema.EXPECTED_STATE_COLS):
        values[col] = [1.0, 2.0, 3.0]
    return {"timestamps": [0.0, 1.0, 2.0], "values": values}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_health_reports_no_model():
    app.state.predictor = None
    app.dependency_overrides.clear()
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


# ---------------------------------------------------------------------------
# Predict — happy path + conversion
# ---------------------------------------------------------------------------
def test_predict_valid_payload(client, valid_payload):
    resp = client.post("/predict", json=valid_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == 1234.5
    assert body["target"] == "Y:Titer"
    assert body["model_type"] == "mock"
    assert body["n_timepoints"] == 3


def test_z_single_element_is_expanded(valid_payload):
    frame = payload_to_frame(PredictRequest(**valid_payload))
    assert len(frame) == 3
    # A single-element Z: value is expanded across every timestamp.
    assert (frame["Z:Stir"] == 1.0).all()
    assert frame[schema.TIME_COL].tolist() == [0.0, 1.0, 2.0]


def test_predict_returns_503_without_model(valid_payload):
    app.state.predictor = None
    app.dependency_overrides.clear()
    resp = TestClient(app).post("/predict", json=valid_payload)
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Predict — validation (422)
# ---------------------------------------------------------------------------
def test_non_increasing_timestamps_rejected(client, valid_payload):
    valid_payload["timestamps"] = [0.0, 0.0, 1.0]
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_wrong_length_channel_rejected(client, valid_payload):
    valid_payload["values"]["W:temp"] = [1.0, 2.0]  # should be length 3
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_missing_prefix_group_rejected(client, valid_payload):
    for col in list(valid_payload["values"]):
        if col.startswith("X:"):
            del valid_payload["values"][col]  # remove all X: variables
    assert client.post("/predict", json=valid_payload).status_code == 422


def test_unknown_prefix_rejected(client, valid_payload):
    valid_payload["values"]["Q:bogus"] = [1.0, 2.0, 3.0]
    assert client.post("/predict", json=valid_payload).status_code == 422


# ---------------------------------------------------------------------------
# Batch utility
# ---------------------------------------------------------------------------
def _synthetic_full_schema_csv(tmp_path, exp_ids=("E1", "E2"), n_days=3):
    rows = []
    for exp in exp_ids:
        for day in range(n_days):
            row = {schema.EXP_COL: exp, schema.TIME_COL: float(day)}
            for col in schema.EXPECTED_STATIC_COLS:
                row[col] = 5.0 if day == 0 else np.nan  # Z: only on day 0
            for col in (*schema.EXPECTED_CONTROL_COLS, *schema.EXPECTED_STATE_COLS):
                row[col] = float(day)
            rows.append(row)
    path = tmp_path / "inputs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_batch_predict_writes_template(tmp_path, monkeypatch, mock_predictor):
    from titer_prediction.service import batch_predict as bp

    monkeypatch.setattr(bp, "load_predictor", lambda _path: mock_predictor)
    csv = _synthetic_full_schema_csv(tmp_path)
    out = tmp_path / "preds.csv"
    bp.batch_predict(csv, "ignored.joblib", out)

    result = pd.read_csv(out)
    assert list(result.columns) == ["RowID", "Exp", "Time[day]", "Y:Titer"]
    assert result["Exp"].tolist() == ["E1", "E2"]
    assert (result["Y:Titer"] == 1234.5).all()
