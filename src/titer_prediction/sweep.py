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

from . import cde

logger = logging.getLogger(__name__)

CDE_SWEEP_SEED = 20260708
CDE_SPLIT_SEED = 20260709
CDE_REFIT_SEED = 20260710
CDE_N_CONFIGS = 20
CDE_MODEL_SEED_BASE = 2026070800

# Bounded grid; exactly CDE_N_CONFIGS configurations are sampled from it.
CDE_SWEEP_GRID: dict[str, list] = {
    "epochs": [250, 400],
    "lr": [1e-3, 3e-3, 1e-2],
    "hidden_size": [8, 16, 32],
    "width": [32, 64],
    "depth": [1, 2],
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


def _sample_grid(
    grid: dict[str, list],
    n_configs: int,
    sweep_seed: int,
    model_seed_base: int,
) -> list[dict[str, Any]]:
    """Sample exactly ``n_configs`` configs without replacement."""
    keys = list(grid)
    all_configs = list(itertools.product(*grid.values()))
    if n_configs > len(all_configs):
        raise ValueError(f"Requested {n_configs} configs but grid has only {len(all_configs)}.")

    rng = np.random.default_rng(sweep_seed)
    chosen = rng.choice(len(all_configs), size=n_configs, replace=False)
    configs: list[dict[str, Any]] = []
    for run_index, idx in enumerate(chosen, start=1):
        cfg = dict(zip(keys, all_configs[int(idx)], strict=True))
        cfg["model_seed"] = model_seed_base + run_index
        configs.append(cfg)
    return configs


def sample_cde_configs(
    n_configs: int = CDE_N_CONFIGS,
    sweep_seed: int = CDE_SWEEP_SEED,
) -> list[dict[str, Any]]:
    """Deterministically sample the CDE sweep configs."""
    return _sample_grid(CDE_SWEEP_GRID, n_configs, sweep_seed, CDE_MODEL_SEED_BASE)


def sweep_cde(
    data_path: str | Path,
    targets_path: str | Path,
    out_path: str | Path,
    model_path: str | Path = "artifacts/cde_best.eqx",
    metadata_path: str | Path = "artifacts/cde_best_metadata.json",
    n_configs: int = CDE_N_CONFIGS,
    sweep_seed: int = CDE_SWEEP_SEED,
    split_seed: int = CDE_SPLIT_SEED,
    refit_seed: int = CDE_REFIT_SEED,
) -> pd.DataFrame:
    """Run the reproducible CDE sweep, then refit and save the best config.

    Each config is fit on the train split only (``cde.train(refit_all=False)``);
    results — config, final train MSE, and validation RMSE/MAE/MAPE/R2 — are
    written incrementally to ``out_path`` so a partial sweep is still usable.
    """
    configs = sample_cde_configs(n_configs, sweep_seed)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run_index, cfg in enumerate(configs, start=1):
        logger.info("cde sweep %d/%d: %s", run_index, n_configs, cfg)
        started = time.perf_counter()
        _, val_metrics, history = cde.train(
            data_path,
            targets_path,
            refit_all=False,
            split_seed=split_seed,
            refit_seed=refit_seed,
            **cfg,
        )
        runtime_s = time.perf_counter() - started
        rows.append(
            {
                "run_index": run_index,
                "sweep_seed": sweep_seed,
                "split_seed": split_seed,
                "refit_seed": refit_seed,
                **cfg,
                "train_mse": history[-1]["train_mse"],
                "val_mse": history[-1].get("val_mse", np.nan),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "runtime_s": runtime_s,
            }
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)  # incremental save

    results = pd.DataFrame(rows)
    best_idx = results["val_r2"].astype(float).idxmax()
    best_row = results.loc[best_idx].to_dict()
    best_config = {
        "epochs": int(best_row["epochs"]),
        "lr": float(best_row["lr"]),
        "hidden_size": int(best_row["hidden_size"]),
        "width": int(best_row["width"]),
        "depth": int(best_row["depth"]),
    }
    best_config["model_seed"] = int(best_row["model_seed"])

    logger.info("Refitting best CDE config on all data: %s", best_config)
    bundle, _, _ = cde.train(
        data_path,
        targets_path,
        refit_all=True,
        split_seed=split_seed,
        refit_seed=refit_seed,
        **best_config,
    )
    metadata = {
        "model_type": "cde",
        "created_utc": datetime.now(UTC).isoformat(),
        "selection_metric": "val_r2",
        "n_configs": n_configs,
        "seeds": {
            "sweep_seed": sweep_seed,
            "split_seed": split_seed,
            "model_seed": best_config["model_seed"],
            "refit_seed": refit_seed,
        },
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible model hyperparameter sweeps.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--data", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", default="artifacts/cde_sweep_results.csv")
    parser.add_argument("--model", default="artifacts/cde_best.eqx")
    parser.add_argument("--metadata", default="artifacts/cde_best_metadata.json")
    parser.add_argument("--n-configs", type=int, default=CDE_N_CONFIGS)
    parser.add_argument("--sweep-seed", type=int, default=CDE_SWEEP_SEED)
    parser.add_argument("--split-seed", type=int, default=CDE_SPLIT_SEED)
    parser.add_argument("--refit-seed", type=int, default=CDE_REFIT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sweep_cde(
        args.data,
        args.targets,
        args.out,
        model_path=args.model,
        metadata_path=args.metadata,
        n_configs=args.n_configs,
        sweep_seed=args.sweep_seed,
        split_seed=args.split_seed,
        refit_seed=args.refit_seed,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
