"""Gradient-boosted (XGBoost) baseline for final-titer prediction.

Consumes the baseline feature matrix from :mod:`titer_prediction.features`, fits
an XGBoost regressor under honest repeated cross-validation, benchmarks it
against a mean-predictor, refits on all training data, and persists a
self-contained model bundle for the inference server.

CLI::

    # train + cross-validate + save a model bundle
    python -m titer_prediction.regression train \
        --data data/datahow_interview_train_data.csv \
        --targets data/datahow_interview_train_targets.csv \
        --model artifacts/xgb_baseline.joblib

    # predict on new inputs using a saved bundle
    python -m titer_prediction.regression predict \
        --data data/datahow_interview_test_data.csv \
        --model artifacts/xgb_baseline.joblib \
        --out artifacts/test_predictions.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate as sk_cross_validate
from xgboost import XGBRegressor

from . import features as feats
from . import schema

logger = logging.getLogger(__name__)

# Shallow, regularised defaults: with ~100 experiments and ~230 features the
# priority is variance control, not capacity.
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 3,
    "random_state": 0,
    "n_jobs": -1,
}

# Titer is strictly positive and right-skewed; train on log1p and invert for
# predictions/metrics so errors are reported in the original units.
_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mape": "neg_mean_absolute_percentage_error",
    "r2": "r2",
}


@dataclass
class ModelBundle:
    """A self-contained, serialisable prediction artifact.

    Bundles the fitted estimator with everything needed to reproduce the feature
    matrix at inference time, so the server has a single source of truth.
    """

    model: TransformedTargetRegressor
    feature_names: list[str]
    gompertz_channel: str
    catch22_channels: list[str] | None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_model(params: dict | None = None) -> TransformedTargetRegressor:
    """XGBoost regressor wrapped to train on log1p(titer)."""
    regressor = XGBRegressor(**(params or DEFAULT_XGB_PARAMS))
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1
    )


# ---------------------------------------------------------------------------
# Cross-validation / benchmarking
# ---------------------------------------------------------------------------
def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Repeated K-fold CV for the model and a mean-predictor baseline.

    Returns ``{estimator: {metric: mean, metric_std: std, ...}}`` with errors in
    the original titer units. Reporting a naive baseline alongside the model
    makes the benchmarking honest: it shows how much signal the features add.
    """
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    estimators = {
        "xgboost": build_model(params),
        "baseline_mean": DummyRegressor(strategy="mean"),
    }

    results: dict[str, dict[str, float]] = {}
    for name, estimator in estimators.items():
        scores = sk_cross_validate(estimator, X, y, cv=cv, scoring=_SCORING)
        summary: dict[str, float] = {}
        for metric in _SCORING:
            values = scores[f"test_{metric}"]
            # sklearn returns negative error scores; flip back to positive.
            if metric != "r2":
                values = -values
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
        results[name] = summary

    return results


def _log_cv_results(results: dict[str, dict[str, float]]) -> None:
    for name, metrics in results.items():
        logger.info(
            "[%-13s] RMSE=%.1f±%.1f  MAE=%.1f±%.1f  MAPE=%.1f%%  R2=%.3f",
            name,
            metrics["rmse"],
            metrics["rmse_std"],
            metrics["mae"],
            metrics["mae_std"],
            metrics["mape"] * 100,
            metrics["r2"],
        )


# ---------------------------------------------------------------------------
# Train / predict
# ---------------------------------------------------------------------------
def train(
    data_path: str | Path,
    targets_path: str | Path,
    params: dict | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
) -> tuple[ModelBundle, dict[str, dict[str, float]]]:
    """Build features, cross-validate, and refit on all data into a bundle."""
    dataset = feats.build_baseline_dataset(data_path, targets_path)
    X, y = dataset.features, dataset.targets
    logger.info("Training on %d experiments x %d features.", *X.shape)

    cv_results = cross_validate(X, y, params, n_splits, n_repeats, seed)
    _log_cv_results(cv_results)

    model = build_model(params)
    model.fit(X, y)
    bundle = ModelBundle(
        model=model,
        feature_names=list(X.columns),
        gompertz_channel=feats.GOMPERTZ_VCD_CHANNEL,
        catch22_channels=None,
        metadata={
            "n_train": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "cv": cv_results,
            "xgb_params": params or DEFAULT_XGB_PARAMS,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return bundle, cv_results


def predict(bundle: ModelBundle, data_path: str | Path) -> pd.Series:
    """Predict final titer for every experiment in ``data_path``."""
    parsed = feats.dp.read_inputs(data_path)
    features = feats.build_baseline_features(parsed, bundle.catch22_channels)
    # Align to the training feature order; unseen/absent columns become NaN,
    # which XGBoost handles natively.
    features = features.reindex(columns=bundle.feature_names)
    preds = bundle.model.predict(features)
    return pd.Series(preds, index=features.index, name=schema.TARGET_COL)


def _final_times(data_path: str | Path) -> pd.Series:
    """Final observed day per experiment (for the output table's Time column)."""
    parsed = feats.dp.read_inputs(data_path)
    return parsed.groupby(schema.EXP_COL, sort=False)[schema.TIME_COL].max()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Saved model bundle to %s", path)


def load_bundle(path: str | Path) -> ModelBundle:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XGBoost baseline for titer prediction.")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Cross-validate, fit, and save a model.")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--targets", required=True)
    p_train.add_argument("--model", required=True, help="Output path for the bundle.")
    p_train.add_argument("--n-splits", type=int, default=5)
    p_train.add_argument("--n-repeats", type=int, default=5)
    p_train.add_argument("--seed", type=int, default=0)

    p_pred = sub.add_parser("predict", help="Predict titer from a saved model.")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--out", required=True, help="Output CSV of predictions.")

    return parser


def _run_train(args: argparse.Namespace) -> int:
    bundle, _ = train(
        args.data, args.targets,
        n_splits=args.n_splits, n_repeats=args.n_repeats, seed=args.seed,
    )
    save_bundle(bundle, args.model)
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.model)
    preds = predict(bundle, args.data)
    times = _final_times(args.data).reindex(preds.index)
    out = pd.DataFrame(
        {
            schema.EXP_COL: preds.index,
            schema.TIME_COL: times.to_numpy(),
            schema.TARGET_COL: preds.to_numpy(),
        }
    ).reset_index(drop=True)
    out.insert(0, schema.ROWID_COL, range(len(out)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote %d predictions to %s", len(out), out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "train":
        return _run_train(args)
    if args.command == "predict":
        return _run_predict(args)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
