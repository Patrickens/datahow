"""Batch prediction utility for the interview / test-template workflow.

The OpenAPI schema is single-experiment; this CLI bridges the provided test CSV
to the target-template CSV by converting each experiment into the same
``/predict`` payload shape and running the local predictor (exercising the exact
API conversion path). Output columns match the template:
``RowID, Exp, Time[day], Y:Titer``.

    python -m titer_prediction.service.batch_predict \
        --data data/datahow_interview_test_data.csv \
        --model artifacts/xgb_baseline.joblib \
        --out artifacts/test_predictions.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .. import schema
from .config import DEFAULT_MODEL_PATH
from .dto import PredictRequest
from .model_loader import load_predictor
from .predictor import predict_one

logger = logging.getLogger(__name__)


def experiment_to_request(group: pd.DataFrame) -> PredictRequest:
    """Turn one experiment's rows into a /predict request payload."""
    group = group.sort_values(schema.TIME_COL)
    timestamps = group[schema.TIME_COL].astype(float).tolist()

    values: dict[str, list[float]] = {}
    for col in group.columns:
        if col.startswith(schema.STATIC_PREFIX):  # Z: -> single element (day-0 value)
            present = group[col].dropna()
            if not present.empty:
                values[col] = [float(present.iloc[0])]
        elif col.startswith((schema.CONTROL_PREFIX, schema.STATE_PREFIX)):  # W:/X:
            values[col] = group[col].astype(float).tolist()

    return PredictRequest(
        timestamps=timestamps,
        values=values,
        experiment_id=str(group[schema.EXP_COL].iloc[0]),
    )


def batch_predict(
    data_path: str | Path, model_path: str | Path, out_path: str | Path
) -> pd.DataFrame:
    """Predict every experiment in ``data_path`` and write a template CSV."""
    predictor = load_predictor(model_path)
    df = pd.read_csv(data_path)

    rows = []
    for exp, group in df.groupby(schema.EXP_COL, sort=False):
        response = predict_one(predictor, experiment_to_request(group))
        rows.append(
            {
                schema.EXP_COL: exp,
                schema.TIME_COL: float(group[schema.TIME_COL].max()),
                schema.TARGET_COL: response.prediction,
            }
        )

    out = pd.DataFrame(rows)
    out.insert(0, schema.ROWID_COL, range(len(out)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote %d predictions (%s) to %s", len(out), predictor.model_type, out_path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-predict a CSV into the target template.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    batch_predict(args.data, args.model, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
