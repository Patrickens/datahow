"""Reproducible hyperparameter sweeps and final refits.

Sweeping is kept separate from the model modules: this file samples configs,
records every seed/config/metric, picks the best validation run, and refits a
deployable model on all training experiments.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import cde, regression
from . import features as feats

logger = logging.getLogger(__name__)

# One seed for everything: config sampling, the train/val split, model init, CV,
# and the final refit. Hardcoded default; the sweep functions/CLI take ``seed=``.
SEED = 0
CDE_N_CONFIGS = 12
XGB_N_CONFIGS = 10
# Each sampled CDE config is scored across this many seeds (different train/val
# splits + inits); we select on the *mean* val R2, not a single lucky holdout.
CDE_N_SEEDS = 3

# CDE hyperparameter grid. Only the genuinely-swept dimensions live here; the rest
# of the pipeline (200 epochs, warmup+cosine LR, gradient clipping, adaptive Tsit5
# solver, train-only standardisation, raw-RMSE early stopping) is fixed in
# ``cde.train``. Capacity stays capped at hidden<=16. 32 combinations.
CDE_SWEEP_GRID: dict[str, list] = {
    "lr": [3e-3, 1e-2],
    "hidden_size": [8, 16],
    "width": [32, 64],
    "depth": [1, 2],
    "batch_size": [16, 32],
}

# Strongest config recovered from the pre-seed lightweight sweep. Always evaluated
# in addition to the random sample so a wider search can never regress below it.
CDE_ANCHOR_CONFIG: dict[str, Any] = {
    "lr": 1e-2,
    "hidden_size": 8,
    "width": 32,
    "depth": 1,
    "batch_size": 32,
}

XGB_SWEEP_GRID: dict[str, list] = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.08],
    "n_estimators": [200, 300, 500],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda": [0.5, 1.0, 3.0],
    "min_child_weight": [1, 3, 5],
}


def _json_ready(value: Any) -> Any:
    """Convert numpy/pandas scalars and NaNs into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _sample_grid(grid: dict[str, list], n_configs: int, seed: int) -> list[dict[str, Any]]:
    """Deterministically sample exactly ``n_configs`` hyperparameter configs."""
    keys = list(grid)
    all_configs = list(itertools.product(*grid.values()))
    if n_configs > len(all_configs):
        raise ValueError(f"Requested {n_configs} configs but grid has only {len(all_configs)}.")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(all_configs), size=n_configs, replace=False)
    return [dict(zip(keys, all_configs[int(idx)], strict=True)) for idx in chosen]


def sample_cde_configs(n_configs: int = CDE_N_CONFIGS, seed: int = SEED) -> list[dict[str, Any]]:
    """Deterministically sample the CDE sweep configs."""
    return _sample_grid(CDE_SWEEP_GRID, n_configs, seed)


def sample_xgb_configs(n_configs: int = XGB_N_CONFIGS, seed: int = SEED) -> list[dict[str, Any]]:
    """Deterministically sample the XGBoost sweep configs."""
    return _sample_grid(XGB_SWEEP_GRID, n_configs, seed)


def _cde_configs_with_anchor(n_configs: int, seed: int) -> list[dict[str, Any]]:
    """Sample ``n_configs`` CDE configs and prepend the anchor (deduplicated).

    The anchor guarantees the strongest known config is always scored, so a
    wider random search can never regress below what we recovered.
    """
    sampled = sample_cde_configs(n_configs, seed)
    return [CDE_ANCHOR_CONFIG] + [c for c in sampled if c != CDE_ANCHOR_CONFIG]


def sweep_cde(
    data_path: str | Path,
    targets_path: str | Path,
    out_path: str | Path,
    model_path: str | Path = "artifacts/cde_best.eqx",
    metadata_path: str | Path = "artifacts/cde_best_metadata.json",
    n_configs: int = CDE_N_CONFIGS,
    seed: int = SEED,
    n_seeds: int = CDE_N_SEEDS,
) -> pd.DataFrame:
    """Run the reproducible CDE sweep, then refit and save the best config.

    Each config is scored across ``n_seeds`` seeds (``seed .. seed + n_seeds-1``),
    each fit on its own train split only (``cde.train(refit_all=False)``). We
    record the mean and std of the per-seed validation metrics and select on the
    **mean** ``val_r2`` — a single 20% holdout is too noisy to rank configs. Rows
    are written incrementally to ``out_path`` so a partial sweep is still usable.
    """
    configs = _cde_configs_with_anchor(n_configs, seed)
    seeds = list(range(seed, seed + n_seeds))
    metric_keys = ("train_mse", "val_rmse", "val_mae", "val_mape", "val_r2")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_index, cfg in enumerate(configs, start=1):
        logger.info("cde sweep %d/%d (x%d seeds): %s", run_index, len(configs), n_seeds, cfg)
        started = time.perf_counter()
        per_seed: dict[str, list[float]] = {k: [] for k in metric_keys}
        for s in seeds:
            _, val_metrics, history = cde.train(
                data_path,
                targets_path,
                refit_all=False,
                seed=s,
                **cfg,
            )
            record = {
                "train_mse": history[-1]["train_mse"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            for k in metric_keys:
                per_seed[k].append(float(record[k]))
        runtime_s = time.perf_counter() - started

        aggregates: dict[str, float] = {}
        for k in metric_keys:
            values = np.asarray(per_seed[k], dtype=float)
            aggregates[f"{k}_mean"] = float(np.mean(values))
            aggregates[f"{k}_std"] = float(np.std(values))
        rows.append(
            {
                "run_index": run_index,
                "is_anchor": cfg == CDE_ANCHOR_CONFIG,
                "n_seeds": n_seeds,
                "seeds": str(seeds),
                **cfg,
                **aggregates,
                "runtime_s": runtime_s,
            }
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)  # incremental save

    results = pd.DataFrame(rows)
    best_idx = results["val_r2_mean"].astype(float).idxmax()
    best_row = results.loc[best_idx].to_dict()
    best_config = {
        "lr": float(best_row["lr"]),
        "hidden_size": int(best_row["hidden_size"]),
        "width": int(best_row["width"]),
        "depth": int(best_row["depth"]),
        "batch_size": int(best_row["batch_size"]),
    }

    logger.info(
        "Best CDE config (val_r2=%.3f+/-%.3f): %s",
        best_row["val_r2_mean"],
        best_row["val_r2_std"],
        best_config,
    )
    logger.info("Refitting best CDE config on all data with seed=%d", seed)
    bundle, _, _ = cde.train(
        data_path,
        targets_path,
        refit_all=True,
        seed=seed,
        **best_config,
    )
    metadata = {
        "model_type": "cde",
        "created_utc": datetime.now(UTC).isoformat(),
        "selection_metric": "val_r2_mean",
        "n_configs": len(configs),
        "seed": seed,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "best_config": best_config,
        "best_validation": best_row,
        "sweep_results": str(out_path),
        "model_path": str(model_path),
    }
    bundle.config["sweep"] = metadata
    cde.save_bundle(bundle, model_path)

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    logger.info("Wrote CDE sweep results to %s and metadata to %s", out_path, metadata_path)
    return results


def _xgb_params(config: dict[str, Any]) -> dict[str, Any]:
    """Merge a sampled config onto the defaults (random_state is set by build_model)."""
    return {
        **regression.DEFAULT_XGB_PARAMS,
        "max_depth": int(config["max_depth"]),
        "learning_rate": float(config["learning_rate"]),
        "n_estimators": int(config["n_estimators"]),
        "subsample": float(config["subsample"]),
        "colsample_bytree": float(config["colsample_bytree"]),
        "reg_lambda": float(config["reg_lambda"]),
        "min_child_weight": int(config["min_child_weight"]),
    }


def sweep_xgb(
    data_path: str | Path,
    targets_path: str | Path,
    out_path: str | Path = "artifacts/xgb_sweep_results.csv",
    model_path: str | Path = "artifacts/xgb_best.joblib",
    metadata_path: str | Path = "artifacts/xgb_best_metadata.json",
    n_configs: int = XGB_N_CONFIGS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run the reproducible XGBoost sweep, then refit and save the best config."""
    dataset = feats.build_baseline_dataset(data_path, targets_path)
    X, y = dataset.features, dataset.targets
    configs = sample_xgb_configs(n_configs, seed)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_index, cfg in enumerate(configs, start=1):
        params = _xgb_params(cfg)
        logger.info("xgb sweep %d/%d: %s", run_index, n_configs, cfg)
        started = time.perf_counter()
        cv_results = regression.cross_validate(X, y, params=params, seed=seed)
        runtime_s = time.perf_counter() - started

        xgb_metrics = cv_results["xgboost"]
        baseline_metrics = cv_results["baseline_mean"]
        rows.append(
            {
                "run_index": run_index,
                "seed": seed,
                **cfg,
                **{f"xgb_{k}": v for k, v in xgb_metrics.items()},
                **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
                "runtime_s": runtime_s,
            }
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)

    results = pd.DataFrame(rows)
    best_idx = results["xgb_r2"].astype(float).idxmax()
    best_row = results.loc[best_idx].to_dict()
    best_config = {
        "max_depth": int(best_row["max_depth"]),
        "learning_rate": float(best_row["learning_rate"]),
        "n_estimators": int(best_row["n_estimators"]),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
        "reg_lambda": float(best_row["reg_lambda"]),
        "min_child_weight": int(best_row["min_child_weight"]),
    }
    refit_params = _xgb_params(best_config)

    logger.info("Refitting best XGBoost config on all data: %s", refit_params)
    bundle, cv_results = regression.train(
        data_path,
        targets_path,
        params=refit_params,
        seed=seed,
    )
    metadata = {
        "model_type": "xgboost",
        "created_utc": datetime.now(UTC).isoformat(),
        "selection_metric": "xgb_r2",
        "n_configs": n_configs,
        "seed": seed,
        "best_config": best_config,
        "refit_params": refit_params,
        "best_validation": best_row,
        "final_cv": cv_results,
        "sweep_results": str(out_path),
        "model_path": str(model_path),
    }
    bundle.metadata["sweep"] = metadata
    regression.save_bundle(bundle, model_path)

    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(_json_ready(metadata), indent=2), encoding="utf-8")
    logger.info("Wrote XGBoost sweep results to %s and metadata to %s", out_path, metadata_path)
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible model hyperparameter sweeps.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--model-kind", choices=("cde", "xgb"), default="cde")
    parser.add_argument("--data", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--n-configs", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=CDE_N_SEEDS, help="Seeds per CDE config.")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.model_kind == "cde":
        sweep_cde(
            args.data,
            args.targets,
            args.out or "artifacts/cde_sweep_results.csv",
            model_path=args.model or "artifacts/cde_best.eqx",
            metadata_path=args.metadata or "artifacts/cde_best_metadata.json",
            n_configs=args.n_configs or CDE_N_CONFIGS,
            seed=args.seed,
            n_seeds=args.n_seeds,
        )
    else:
        sweep_xgb(
            args.data,
            args.targets,
            args.out or "artifacts/xgb_sweep_results.csv",
            model_path=args.model or "artifacts/xgb_best.joblib",
            metadata_path=args.metadata or "artifacts/xgb_best_metadata.json",
            n_configs=args.n_configs or XGB_N_CONFIGS,
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
