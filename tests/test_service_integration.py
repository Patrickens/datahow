"""Integration tests that exercise the **real** model bundle (no mock).

Unlike ``test_service.py`` (which mocks the predictor to test the API layer in
isolation), these load the actual ``artifacts/xgb_baseline.joblib`` and run real
inference end to end. They skip cleanly when the servable artifact or the
confidential test CSV are absent (fresh clone / CI).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from titer_prediction import schema
from titer_prediction.service import batch_predict as bp
from titer_prediction.service.app import app, get_predictor
from titer_prediction.service.errors import ModelLoadError
from titer_prediction.service.model_loader import BundlePredictor, load_predictor

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "artifacts" / "xgb_baseline.joblib"
TEST_DATA = REPO / "data" / "datahow_interview_test_data.csv"

requires_model = pytest.mark.skipif(not MODEL.exists(), reason="xgb_baseline.joblib not present")
requires_test_data = pytest.mark.skipif(not TEST_DATA.exists(), reason="test data CSV not present")


def _payload_from_first_experiment(csv_path: Path) -> dict:
    """Build a /predict JSON payload from the first experiment in a CSV."""
    df = pd.read_csv(csv_path)
    exp = df[schema.EXP_COL].iloc[0]
    group = df[df[schema.EXP_COL] == exp]
    return bp.experiment_to_request(group).model_dump()


# ---------------------------------------------------------------------------
# load_predictor: error paths + real bundle
# ---------------------------------------------------------------------------
def test_load_predictor_missing_file(tmp_path):
    with pytest.raises(ModelLoadError):
        load_predictor(tmp_path / "does_not_exist.joblib")


def test_load_predictor_unknown_suffix(tmp_path):
    bogus = tmp_path / "model.txt"
    bogus.write_text("not a model")
    with pytest.raises(ModelLoadError):
        load_predictor(bogus)


@requires_model
def test_load_predictor_real_bundle():
    predictor = load_predictor(MODEL)
    assert isinstance(predictor, BundlePredictor)
    assert predictor.model_type == "xgboost"
    assert predictor.target == schema.TARGET_COL
    assert predictor.expected_static == schema.EXPECTED_STATIC_COLS
    assert predictor.expected_control == schema.EXPECTED_CONTROL_COLS
    assert predictor.expected_state == schema.EXPECTED_STATE_COLS


# ---------------------------------------------------------------------------
# Real end-to-end prediction (HTTP layer + real model)
# ---------------------------------------------------------------------------
@requires_model
@requires_test_data
def test_predict_real_model_over_http():
    predictor = load_predictor(MODEL)
    app.state.predictor = predictor
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        client = TestClient(app)
        resp = client.post("/predict", json=_payload_from_first_experiment(TEST_DATA))
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_type"] == "xgboost"
        assert body["target"] == schema.TARGET_COL
        assert math.isfinite(body["prediction"])
        assert body["prediction"] > 0  # titer is strictly positive
    finally:
        app.dependency_overrides.clear()
        app.state.predictor = None


@requires_model
@requires_test_data
def test_batch_predict_real_model(tmp_path):
    out = tmp_path / "preds.csv"
    bp.batch_predict(TEST_DATA, MODEL, out)

    written = pd.read_csv(out)
    assert list(written.columns) == [
        schema.ROWID_COL,
        schema.EXP_COL,
        schema.TIME_COL,
        schema.TARGET_COL,
    ]
    n_exp = pd.read_csv(TEST_DATA)[schema.EXP_COL].nunique()
    assert len(written) == n_exp
    preds = written[schema.TARGET_COL].to_numpy()
    assert np.isfinite(preds).all()
    assert (preds > 0).all()
