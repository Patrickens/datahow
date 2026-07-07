"""Preprocess the raw bioprocess CSVs into model-ready representations.

Two representations are produced from the same parsed source:

* **Tabular features** (one row per experiment) for the XGBoost baseline. Each
  variable-length trajectory is collapsed into a fixed set of aggregate
  statistics, alongside the pass-through ``Z:`` design scalars.
* **Padded sequences** (experiments x time x channels) for the neural CDE,
  together with a validity mask and the per-step times.

Run as a CLI to materialise the tabular features to disk::

    python -m titer_prediction.data_preprocessing \
        --data data/datahow_interview_train_data.csv \
        --targets data/datahow_interview_train_targets.csv \
        --out artifacts/train_features.parquet

The ``--targets`` argument is optional: omit it for the test inputs, whose
targets are withheld.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

logger = logging.getLogger(__name__)

# numpy renamed ``trapz`` -> ``trapezoid`` in 2.0 (and removed ``trapz`` in 2.x).
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class TabularDataset:
    """Per-experiment feature matrix with an optional aligned target.

    Attributes:
        features: Index = experiment id, columns = engineered features.
        targets: Index = experiment id, values = final titer, or ``None`` when
            targets are not available (e.g. the held-out test inputs).
    """

    features: pd.DataFrame
    targets: pd.Series | None

    @property
    def has_targets(self) -> bool:
        return self.targets is not None


@dataclass
class SequenceDataset:
    """Padded sequence tensors for path-based models (the neural CDE).

    Attributes:
        exp_ids: Experiment ids, one per sample (length ``n``).
        times: ``(n, t_max)`` per-step times in days; padded steps repeat the
            last real time so the path stays constant after the experiment ends.
        channels: ``(n, t_max, c)`` control + state values over time.
        static: ``(n, s)`` the ``Z:`` design scalars.
        mask: ``(n, t_max)`` boolean, ``True`` for real (non-padded) steps.
        channel_names: Names of the ``c`` channel columns, in order.
        static_names: Names of the ``s`` static columns, in order.
        targets: ``(n,)`` final titer, or ``None`` if unavailable.
    """

    exp_ids: list[str]
    times: np.ndarray
    channels: np.ndarray
    static: np.ndarray
    mask: np.ndarray
    channel_names: list[str]
    static_names: list[str]
    targets: np.ndarray | None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_inputs(path: str | Path) -> pd.DataFrame:
    """Load a raw inputs CSV and forward-fill the static (``Z:``) columns.

    The design parameters are recorded only on day 0; we fill them across every
    time step of the experiment so downstream code can treat them uniformly.
    """
    df = pd.read_csv(path)
    _validate_input_columns(df)
    df = df.sort_values([schema.EXP_COL, schema.TIME_COL]).reset_index(drop=True)

    static_cols = schema.static_columns(df)
    # Fill within each experiment only, then back-fill in case day 0 is absent.
    df[static_cols] = df.groupby(schema.EXP_COL, sort=False)[static_cols].ffill().bfill()
    return df


def read_targets(path: str | Path) -> pd.Series:
    """Load a targets CSV into a Series indexed by experiment id."""
    df = pd.read_csv(path)
    if schema.TARGET_COL not in df.columns:
        raise ValueError(
            f"Targets file {path!r} is missing the {schema.TARGET_COL!r} column; "
            f"found {list(df.columns)}."
        )
    if df[schema.EXP_COL].duplicated().any():
        raise ValueError(f"Targets file {path!r} has more than one row per experiment.")
    return df.set_index(schema.EXP_COL)[schema.TARGET_COL].rename(schema.TARGET_COL)


def _validate_input_columns(df: pd.DataFrame) -> None:
    """Fail fast if the identifier columns or prefix groups are missing."""
    for col in (schema.EXP_COL, schema.TIME_COL):
        if col not in df.columns:
            raise ValueError(f"Input data is missing required column {col!r}.")
    for name, cols in (
        ("static Z:", schema.static_columns(df)),
        ("control W:", schema.control_columns(df)),
        ("state X:", schema.state_columns(df)),
    ):
        if not cols:
            raise ValueError(f"Input data has no {name} columns.")


# ---------------------------------------------------------------------------
# Feature engineering (tabular / baseline)
# ---------------------------------------------------------------------------
def _auc(time: np.ndarray, values: np.ndarray) -> float:
    """Trapezoidal integral of a trajectory over time (0 if <2 points)."""
    if time.size < 2:
        return 0.0
    return float(_trapezoid(values, time))


def _slope(time: np.ndarray, values: np.ndarray) -> float:
    """Ordinary least-squares slope of ``values`` against ``time``."""
    if time.size < 2 or np.ptp(time) == 0:
        return 0.0
    return float(np.polyfit(time, values, 1)[0])


def _channel_stats(name: str, time: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Aggregate one trajectory into a fixed set of named statistics.

    NaNs are dropped before aggregation so partially-missing channels still
    yield sensible summaries. The chosen aggregates capture level (first/last/
    mean), spread (min/max/std), accumulation (AUC — e.g. integral of viable
    cells) and trend (slope).
    """
    finite = np.isfinite(values)
    t, v = time[finite], values[finite]
    if v.size == 0:
        keys = ("first", "last", "min", "max", "mean", "std", "auc", "slope")
        return {f"{name}_{k}": np.nan for k in keys}
    return {
        f"{name}_first": float(v[0]),
        f"{name}_last": float(v[-1]),
        f"{name}_min": float(v.min()),
        f"{name}_max": float(v.max()),
        f"{name}_mean": float(v.mean()),
        f"{name}_std": float(v.std()),
        f"{name}_auc": _auc(t, v),
        f"{name}_slope": _slope(t, v),
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long per-timestep frame into one feature row per experiment.

    Features:
      * the ``Z:`` design scalars (constant per experiment), passed through;
      * ``duration_observed`` (last observed day) and ``n_timepoints``;
      * per-channel aggregates for every ``W:`` and ``X:`` trajectory.
    """
    static_cols = schema.static_columns(df)
    traj_cols = schema.control_columns(df) + schema.state_columns(df)

    rows: dict[str, dict[str, float]] = {}
    for exp, group in df.groupby(schema.EXP_COL, sort=False):
        group = group.sort_values(schema.TIME_COL)
        time = group[schema.TIME_COL].to_numpy(dtype=float)

        feat: dict[str, float] = {}
        for col in static_cols:
            series = group[col].dropna()
            feat[col] = float(series.iloc[0]) if not series.empty else np.nan

        feat["duration_observed"] = float(time.max()) if time.size else np.nan
        feat["n_timepoints"] = float(time.size)

        for col in traj_cols:
            feat.update(_channel_stats(col, time, group[col].to_numpy(dtype=float)))

        rows[exp] = feat

    features = pd.DataFrame.from_dict(rows, orient="index")
    features.index.name = schema.EXP_COL
    return features


def build_tabular_dataset(
    data_path: str | Path, targets_path: str | Path | None = None
) -> TabularDataset:
    """End-to-end: raw CSV(s) -> aligned per-experiment features and targets."""
    df = read_inputs(data_path)
    features = build_features(df)

    targets: pd.Series | None = None
    if targets_path is not None:
        targets = read_targets(targets_path)
        missing = set(features.index) - set(targets.index)
        if missing:
            raise ValueError(
                f"{len(missing)} experiments have no target (e.g. {sorted(missing)[:3]})."
            )
        targets = targets.reindex(features.index)

    return TabularDataset(features=features, targets=targets)


# ---------------------------------------------------------------------------
# Sequence building (path-based / CDE)
# ---------------------------------------------------------------------------
def build_sequence_dataset(
    data_path: str | Path, targets_path: str | Path | None = None
) -> SequenceDataset:
    """Build padded sequence tensors for path-based models (the neural CDE).

    Sequences are right-padded to the longest experiment. Padded time steps
    repeat the final real time and final channel values, so a CDE integrating
    over the padded region sees a constant (zero-derivative) path and the
    terminal hidden state is unaffected. The ``mask`` records which steps are
    real.
    """
    df = read_inputs(data_path)
    channel_names = schema.control_columns(df) + schema.state_columns(df)
    static_names = schema.static_columns(df)

    groups = list(df.groupby(schema.EXP_COL, sort=False))
    exp_ids = [str(exp) for exp, _ in groups]
    t_max = max(len(g) for _, g in groups)
    n, c, s = len(groups), len(channel_names), len(static_names)

    times = np.zeros((n, t_max), dtype=float)
    channels = np.zeros((n, t_max, c), dtype=float)
    static = np.zeros((n, s), dtype=float)
    mask = np.zeros((n, t_max), dtype=bool)

    for i, (_, group) in enumerate(groups):
        group = group.sort_values(schema.TIME_COL)
        length = len(group)
        t = group[schema.TIME_COL].to_numpy(dtype=float)

        times[i, :length] = t
        times[i, length:] = t[-1]  # hold last time on the padded tail
        channels[i, :length, :] = group[channel_names].to_numpy(dtype=float)
        if length < t_max:
            channels[i, length:, :] = channels[i, length - 1, :]
        static[i, :] = group[static_names].iloc[0].to_numpy(dtype=float)
        mask[i, :length] = True

    targets: np.ndarray | None = None
    if targets_path is not None:
        target_series = read_targets(targets_path).reindex(exp_ids)
        if target_series.isna().any():
            raise ValueError("Some experiments are missing a target value.")
        targets = target_series.to_numpy(dtype=float)

    return SequenceDataset(
        exp_ids=exp_ids,
        times=times,
        channels=channels,
        static=static,
        mask=mask,
        channel_names=channel_names,
        static_names=static_names,
        targets=targets,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_tabular(dataset: TabularDataset, out_path: str | Path) -> None:
    """Write the feature matrix (plus target column if present) to parquet."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = dataset.features.copy()
    if dataset.targets is not None:
        table[schema.TARGET_COL] = dataset.targets
    # Reset index so the experiment id survives the round-trip as a column.
    table.reset_index().to_parquet(out_path, index=False)
    logger.info("Wrote %d experiments x %d features to %s", *table.shape, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build model-ready features from the raw bioprocess CSVs.",
    )
    parser.add_argument("--data", required=True, help="Path to the inputs CSV.")
    parser.add_argument(
        "--targets",
        default=None,
        help="Path to the targets CSV (omit for held-out test inputs).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the parquet feature table (optional).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build features and optionally persist them."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dataset = build_tabular_dataset(args.data, args.targets)
    n_exp, n_feat = dataset.features.shape
    logger.info("Built features for %d experiments with %d features.", n_exp, n_feat)
    if dataset.has_targets:
        t = dataset.targets
        logger.info("Target titer: min=%.1f max=%.1f mean=%.1f", t.min(), t.max(), t.mean())

    if args.out:
        save_tabular(dataset, args.out)
    else:
        logger.info("No --out given; not writing to disk.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
