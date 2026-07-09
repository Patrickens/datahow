"""Load and parse the raw bioprocess CSVs into model-ready representations.

This module owns the shared *loading* concerns — reading the CSVs, forward-
filling the day-0 ``Z:`` design scalars, and validating the column schema — plus
two container shapes built from the same parsed source:

* :class:`TabularDataset` — a per-experiment feature matrix (built by
  :mod:`titer_prediction.features`) aligned with its targets, for the XGBoost
  baseline.
* :class:`SequenceData` — ragged per-experiment trajectories for the neural CDE.
  Padding, standardisation, and path interpolation are intentionally left to the
  CDE module (diffrax) rather than re-implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema

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
class ExperimentSequence:
    """One experiment's raw, *ragged* trajectory for path-based models.

    Deliberately unpadded: batching, standardisation, and path interpolation are
    left to the CDE module (see :func:`titer_prediction.cde.make_mixed_cde_path`)
    rather than re-implementing them here.

    Attributes:
        exp_id: Experiment id.
        times: ``(t,)`` observation times in days (strictly increasing).
        channels: ``(t, c)`` control + state values over time.
        static: ``(s,)`` the ``Z:`` design scalars.
        target: Final titer, or ``None`` if unavailable.
    """

    exp_id: str
    times: np.ndarray
    channels: np.ndarray
    static: np.ndarray
    target: float | None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_inputs(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load raw inputs and forward-fill the static (``Z:``) columns.

    ``source`` may be a path to a CSV or an already-in-memory DataFrame (used by
    the inference service, which builds a one-experiment frame from an API
    payload). The design parameters are recorded only on day 0; we fill them
    across every time step of the experiment so downstream code can treat them
    uniformly.
    """
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
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
    """Trapezoidal integral of a trajectory over time.

    NaNs in either array are dropped pairwise first; returns NaN if fewer than
    two finite points remain. Shared by the feature builders in
    :mod:`titer_prediction.features`.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(time) & np.isfinite(values)
    if finite.sum() < 2:
        return np.nan
    return float(_trapezoid(values[finite], time[finite]))


def assemble_tabular(features: pd.DataFrame, targets_path: str | Path | None) -> TabularDataset:
    """Align an already-built feature matrix with its targets into a dataset.

    Shared by the feature-matrix builders: loads the targets (if given), checks
    every experiment has one, and reindexes them to the feature rows.
    """
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
@dataclass
class SequenceData:
    """A collection of ragged experiment trajectories plus their column order."""

    experiments: list[ExperimentSequence]
    channel_names: list[str]
    static_names: list[str]

    def __len__(self) -> int:
        return len(self.experiments)

    @property
    def has_targets(self) -> bool:
        return all(exp.target is not None for exp in self.experiments)


def build_sequences(data_path: str | Path, targets_path: str | Path | None = None) -> SequenceData:
    """Extract one ragged :class:`ExperimentSequence` per experiment.

    No padding, masking, or interpolation is done here — those are the CDE's
    concern and are delegated to diffrax. Channels are the ``W:`` controls
    followed by the ``X:`` states; static features are the ``Z:`` scalars.
    """
    df = read_inputs(data_path)
    channel_names = schema.control_columns(df) + schema.state_columns(df)
    static_names = schema.static_columns(df)

    targets: pd.Series | None = None
    if targets_path is not None:
        targets = read_targets(targets_path)

    experiments: list[ExperimentSequence] = []
    for exp, group in df.groupby(schema.EXP_COL, sort=False):
        group = group.sort_values(schema.TIME_COL)
        target: float | None = None
        if targets is not None:
            if exp not in targets.index:
                raise ValueError(f"Experiment {exp!r} has no target value.")
            target = float(targets.loc[exp])
        experiments.append(
            ExperimentSequence(
                exp_id=str(exp),
                times=group[schema.TIME_COL].to_numpy(dtype=float),
                channels=group[channel_names].to_numpy(dtype=float),
                static=group[static_names].iloc[0].to_numpy(dtype=float),
                target=target,
            )
        )

    return SequenceData(
        experiments=experiments,
        channel_names=channel_names,
        static_names=static_names,
    )
