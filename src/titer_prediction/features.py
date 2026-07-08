"""Baseline (XGBoost) feature engineering.

Turns the parsed long-format frame into a fixed per-experiment feature matrix by
combining three complementary views of each variable-length trajectory:

1. **Simple aggregates + design scalars** — reused from
   :func:`titer_prediction.data_preprocessing.build_features` (first/last/min/
   max/mean/std/AUC/slope per channel, plus the pass-through ``Z:`` parameters).
2. **Gompertz growth-curve parameters** — a 4-parameter-with-baseline Gompertz
   curve is fit to the viable-cell-density (``X:VCD``) trajectory and its
   parameters extracted (amplitude, shape, inflection time, growth rate,
   baseline). These compress a whole growth curve into a handful of interpretable
   numbers, robust to differing series lengths. Note the limitation: a single
   monotone sigmoid *cannot* capture sequential substrate dynamics — the ordered
   depletion of glucose then glutamine, feed-driven replenishment, or the lactate
   production->consumption switch. Those coupled dynamics are left to catch22 and
   are handled more naturally by the CDE.
3. **catch22 features** — the canonical 22-feature time-series set applied per
   measured-state (``X:``) channel. Deliberately compact (well-suited to ~100
   samples) and numba-free, so it shares a single environment with the
   JAX/diffrax stack.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pycatch22
from scipy.optimize import curve_fit
from scipy.stats.mstats import winsorize

from . import data_preprocessing as dp
from . import schema

# ---------------------------------------------------------------------------
# Gompertz growth-curve features
# ---------------------------------------------------------------------------
GOMPERTZ_PARAMS: tuple[str, ...] = ("a", "b", "t_i", "k_g", "y0")
GOMPERTZ_VCD_CHANNEL = "X:VCD"

# Minimum number of finite points required to fit the 5-parameter curve.
_MIN_POINTS_FOR_FIT = len(GOMPERTZ_PARAMS) + 1


def gompertz(t, a, b, t_i, k_g, y0):
    """4-parameter Gompertz growth model with a baseline offset.

    ``y(t) = y0 + a * exp(-b * exp(-k_g * (t - t_i)))``

    Parameters (all also used as extracted features):
        a: amplitude — the total rise from baseline to upper asymptote.
        b: displacement / shape — governs the lag before growth.
        t_i: location parameter in time (couples with ``b`` to set the
            inflection point at ``t_i + ln(b) / k_g``).
        k_g: growth rate — steepness of the exponential phase.
        y0: baseline — the lower asymptote.
    """
    return y0 + a * np.exp(-b * np.exp(-k_g * (t - t_i)))


def fit_gompertz(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Fit the Gompertz curve to a single trajectory.

    Returns ``(params, ok)`` where ``params`` is the length-5 parameter vector in
    the order of :data:`GOMPERTZ_PARAMS` (NaNs if the fit fails) and ``ok`` flags
    a successful fit.

    The routine derives its initial guess and bounds from the data itself (no
    domain-specific magic constants), making it usable across channels and
    scales. It is adapted from a prior Gompertz-fitting implementation.
    """
    nan_params = np.full(len(GOMPERTZ_PARAMS), np.nan)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(t) & np.isfinite(y)
    t, y = t[finite], y[finite]
    if t.size < _MIN_POINTS_FOR_FIT or np.ptp(t) == 0:
        return nan_params, False

    try:
        y = np.clip(y, 1e-3, None)
        y_min, y_max = float(np.min(y)), float(np.max(y))
        y_range = max(y_max - y_min, 1e-3)
        y0_est = y_min * 0.9  # baseline just below the minimum

        # Locate the (robust) peak to split growth vs plateau for the guess.
        y_clean = np.asarray(winsorize(y, limits=[0.05, 0.05]))
        cut = int(np.argmax(y_clean))

        # Growth-phase slope (start -> peak) seeds the growth-rate guess k_g.
        if cut > 0:
            design = np.column_stack((np.ones(cut + 1), t[: cut + 1]))
            slope = np.linalg.lstsq(design, np.log(y[: cut + 1] - y0_est + 1e-3), rcond=None)[0][1]
        else:
            slope = 0.01

        p0 = [
            y_range,  # a
            1.0,  # b
            float(t[cut]),  # t_i
            max(float(slope), 0.01),  # k_g
            max(y0_est, 0.0),  # y0
        ]
        lower = [y_range * 0.1, 1e-3, float(t.min()), 1e-3, 0.0]
        upper = [y_range * 3.0, 1e4, float(t.max()), 5.0, max(y_min, 1e-3)]
        p0 = np.clip(p0, lower, upper)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params, _ = curve_fit(
                gompertz,
                t,
                y,
                p0=p0,
                bounds=(lower, upper),
                method="trf",
                maxfev=10000,
            )
        return np.asarray(params, dtype=float), True
    except Exception:
        return nan_params, False


def _r2(t: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
    """Coefficient of determination of a Gompertz fit (fit-quality feature)."""
    pred = gompertz(t, *params)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def gompertz_features(df: pd.DataFrame, channel: str = GOMPERTZ_VCD_CHANNEL) -> pd.DataFrame:
    """Per-experiment Gompertz parameters (+ fit quality) for one channel."""
    rows: dict[str, dict[str, float]] = {}
    for exp, group in df.groupby(schema.EXP_COL, sort=False):
        group = group.sort_values(schema.TIME_COL)
        t = group[schema.TIME_COL].to_numpy(dtype=float)
        y = group[channel].to_numpy(dtype=float)

        params, ok = fit_gompertz(t, y)
        feat = {
            f"gompertz_{channel}_{name}": value
            for name, value in zip(GOMPERTZ_PARAMS, params, strict=True)
        }
        feat[f"gompertz_{channel}_r2"] = _r2(t, y, params) if ok else np.nan
        feat[f"gompertz_{channel}_ok"] = float(ok)
        rows[exp] = feat

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = schema.EXP_COL
    return out


# ---------------------------------------------------------------------------
# catch22 features
# ---------------------------------------------------------------------------
# Fetch the canonical feature names once so output columns stay consistent even
# when a particular series fails (e.g. constant channels -> NaNs).
CATCH22_NAMES: list[str] = pycatch22.catch22_all(list(range(10)), catch24=False)["names"]


def _catch22_channel(values: np.ndarray) -> dict[str, float]:
    """catch22 features for a single trajectory, NaN-safe."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3 or np.ptp(finite) == 0:
        return dict.fromkeys(CATCH22_NAMES, np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pycatch22.catch22_all(finite.tolist(), catch24=False)
        return dict(zip(res["names"], res["values"], strict=True))
    except Exception:
        return dict.fromkeys(CATCH22_NAMES, np.nan)


def catch22_features(df: pd.DataFrame, channels: list[str] | None = None) -> pd.DataFrame:
    """Per-experiment catch22 features for each requested channel.

    Defaults to the measured-state (``X:``) channels: they carry the dynamic,
    informative signal, whereas the ``W:`` controls are largely determined by the
    ``Z:`` design scalars and are often constant (yielding uninformative NaNs).
    """
    if channels is None:
        channels = schema.state_columns(df)

    rows: dict[str, dict[str, float]] = {}
    for exp, group in df.groupby(schema.EXP_COL, sort=False):
        group = group.sort_values(schema.TIME_COL)
        feat: dict[str, float] = {}
        for channel in channels:
            stats = _catch22_channel(group[channel].to_numpy(dtype=float))
            feat.update({f"catch22_{channel}_{name}": value for name, value in stats.items()})
        rows[exp] = feat

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = schema.EXP_COL
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_baseline_features(
    df: pd.DataFrame, catch22_channels: list[str] | None = None
) -> pd.DataFrame:
    """Assemble the full baseline feature matrix (one row per experiment).

    Concatenates simple aggregates + ``Z:`` scalars, Gompertz(VCD) parameters,
    and catch22 features, aligned on the experiment id.
    """
    aggregates = dp.build_features(df)
    gompertz = gompertz_features(df, GOMPERTZ_VCD_CHANNEL)
    catch22 = catch22_features(df, catch22_channels)

    features = pd.concat([aggregates, gompertz, catch22], axis=1)
    features.index.name = schema.EXP_COL
    return features


def build_baseline_dataset(
    data_path, targets_path=None, catch22_channels: list[str] | None = None
) -> dp.TabularDataset:
    """End-to-end: raw CSV(s) -> baseline features aligned with targets."""
    parsed = dp.read_inputs(data_path)
    features = build_baseline_features(parsed, catch22_channels)

    targets = None
    if targets_path is not None:
        targets = dp.read_targets(targets_path)
        missing = set(features.index) - set(targets.index)
        if missing:
            raise ValueError(
                f"{len(missing)} experiments have no target (e.g. {sorted(missing)[:3]})."
            )
        targets = targets.reindex(features.index)

    return dp.TabularDataset(features=features, targets=targets)
