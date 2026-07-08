"""Lightweight hyperparameter sweep for the neural CDE.

Kept separate from :mod:`titer_prediction.cde`: sweeping is an
experiment-management concern (sample configs, train each, tabulate metrics), not
part of the model or its training loop. This module is a thin wrapper around
:func:`titer_prediction.cde.train`.

Run as a CLI::

    python -m titer_prediction.sweep \
        --data data/datahow_interview_train_data.csv \
        --targets data/datahow_interview_train_targets.csv \
        --out artifacts/cde_sweep_results.csv --max-configs 30
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import cde

logger = logging.getLogger(__name__)

# Bounded grid; we randomly sample a handful of configurations from it.
SWEEP_GRID: dict[str, list] = {
    "epochs": [300, 600],
    "lr": [1e-3, 3e-3, 1e-2],
    "hidden_size": [8, 16, 32],
    "width": [32, 64],
    "depth": [1, 2],
    "seed": [0, 1, 2],
}


def sweep(
    data_path: str | Path,
    targets_path: str | Path,
    out_path: str | Path,
    max_configs: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Train a random sample of <= ``max_configs`` configs and log holdout metrics.

    Each config is fit on the train split only (``cde.train(refit_all=False)``);
    results — config, final train MSE, and validation RMSE/MAE/MAPE/R2 — are
    written incrementally to ``out_path`` so a partial sweep is still usable.
    """
    keys = list(SWEEP_GRID)
    all_configs = list(itertools.product(*SWEEP_GRID.values()))
    rng = np.random.default_rng(seed)
    n_pick = min(max_configs, len(all_configs))
    chosen = rng.choice(len(all_configs), size=n_pick, replace=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, idx in enumerate(chosen, start=1):
        cfg = dict(zip(keys, all_configs[idx], strict=True))
        logger.info("sweep %d/%d: %s", i, n_pick, cfg)
        _, val_metrics, history = cde.train(data_path, targets_path, refit_all=False, **cfg)
        rows.append(
            {
                **cfg,
                "train_mse": history[-1]["train_mse"],
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )
        pd.DataFrame(rows).to_csv(out_path, index=False)  # incremental save

    logger.info("Wrote %d sweep results to %s", len(rows), out_path)
    return pd.DataFrame(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample & evaluate neural-CDE hyperparameters.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--data", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", default="artifacts/cde_sweep_results.csv")
    parser.add_argument("--max-configs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sweep(args.data, args.targets, args.out, max_configs=args.max_configs, seed=args.seed)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
