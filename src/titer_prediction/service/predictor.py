"""Turn an API request into model input and run the prediction.

This is the only place the API touches model data. It builds a one-experiment
DataFrame in exactly the shape the training pipeline expects (columns ``Exp``,
``Time[day]``, ``Z:``/``W:``/``X:``) and hands it to the loaded model's
``predict_frame`` — which flows through the same ``read_inputs`` preprocessing as
training. No model logic is re-implemented here.
"""

from __future__ import annotations

import pandas as pd

from .. import schema
from .dto import PredictRequest, PredictResponse
from .errors import PayloadError
from .model_loader import Predictor

DEFAULT_EXP_ID = "request_0"


def payload_to_frame(request: PredictRequest) -> pd.DataFrame:
    """Build a one-row-per-timestep DataFrame from a validated request.

    Single-element ``Z:`` arrays are expanded across all timestamps (matching the
    CSV convention, where design scalars are constant per experiment). Rows are
    sorted by time.
    """
    exp_id = request.experiment_id or DEFAULT_EXP_ID
    n = len(request.timestamps)

    columns: dict[str, list] = {
        schema.EXP_COL: [exp_id] * n,
        schema.TIME_COL: list(request.timestamps),
    }
    for name, arr in request.values.items():
        if name.startswith(schema.STATIC_PREFIX) and len(arr) == 1:
            columns[name] = list(arr) * n  # expand Z: scalar across timestamps
        else:
            columns[name] = list(arr)

    return pd.DataFrame(columns).sort_values(schema.TIME_COL).reset_index(drop=True)


def predict_one(predictor: Predictor, request: PredictRequest) -> PredictResponse:
    """Convert, validate against the model's schema, and predict a single titer."""
    frame = payload_to_frame(request)
    expected = (
        *predictor.expected_static,
        *predictor.expected_control,
        *predictor.expected_state,
    )
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise PayloadError(f"payload is missing variables the model expects: {missing}")

    # Feed the model exactly its trained schema (drops any extra variables), so
    # both the tabular baseline and the channel-shape-sensitive CDE work.
    frame = frame[[schema.EXP_COL, schema.TIME_COL, *expected]]
    preds = predictor.predict_frame(frame)

    return PredictResponse(
        prediction=float(preds.iloc[0]),
        target=predictor.target,
        model_type=predictor.model_type,
        n_timepoints=len(request.timestamps),
        experiment_id=request.experiment_id,
    )
