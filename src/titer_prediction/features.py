"""Baseline (XGBoost) feature engineering.

Turns the parsed long-format frame into a fixed per-experiment feature matrix by
combining three complementary views of each variable-length trajectory:

1. **Static + meta features** — the pass-through ``Z:`` design scalars plus the
   observed duration and number of timepoints.
2. **TSFEL features** — a curated set of interpretable statistical and temporal
   features (TSFEL) applied per measured-state (``X:``) channel. Crucially this
   includes domain-meaningful quantities such as the **area under the curve**
   (e.g. the integral of viable cells), which generic dynamical-systems feature
   sets omit.
3. **Gompertz growth-curve parameters** — fit to the viable-cell-density
   (``X:VCD``) trajectory, extracted as **custom TSFEL features** (decorated with
   ``@set_domain``). A single monotone sigmoid summarises the growth curve but
   *cannot* capture sequential substrate dynamics (ordered glucose-then-glutamine
   depletion, feed-driven replenishment, or the lactate production->consumption
   switch); those are left to the TSFEL features and the CDE.
4. **Substrate feed-accounting features** — small, biologically motivated
   summaries for the fed substrates glucose and glutamine: initial/final level,
   total feed integral, initial plus fed amount, and apparent amount consumed.
5. **Cell-population accounting features** — estimated total cell density from
   viable cell density and lysed fraction.

On the feature-library choice: we considered **tsfresh** but it conflicts with
the JAX/diffrax stack (its numba/stumpy dependency pins an incompatible numpy)
and emits 200+ features. We then tried **catch22**, but its 22 canonical
dynamical-systems features are not well matched to this problem — it has no AUC,
for instance. **TSFEL** is the compromise: numba-free (one environment),
interpretable, includes the bioprocess-relevant features we want, and is
extensible, which lets us fold Gompertz in as a personalised feature.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import tsfel
from scipy.optimize import curve_fit
from scipy.stats.mstats import winsorize
from tsfel.feature_extraction.features_utils import set_domain

from . import data_preprocessing as dp
from . import schema

# ---------------------------------------------------------------------------
# Gompertz growth-curve model
# ---------------------------------------------------------------------------
GOMPERTZ_PARAMS: tuple[str, ...] = ("a", "b", "t_i", "k_g", "y0")
GOMPERTZ_VCD_CHANNEL = "X:VCD"
FED_SUBSTRATE_CHANNELS: tuple[str, ...] = ("X:Glc", "X:Gln")
MATCHING_FEEDS: dict[str, str] = {
    "X:Glc": "W:FeedGlc",
    "X:Gln": "W:FeedGln",
}
TOTAL_CELL_FEATURE_PREFIX = "bio_total_cell_density"

_MIN_POINTS_FOR_FIT = len(GOMPERTZ_PARAMS) + 1


def gompertz(t, a, b, t_i, k_g, y0):
    """4-parameter Gompertz growth model with a baseline offset.

    ``y(t) = y0 + a * exp(-b * exp(-k_g * (t - t_i)))``

    Parameters (also the extracted features): amplitude ``a``, displacement/shape
    ``b``, location ``t_i``, growth rate ``k_g``, baseline ``y0``.
    """
    return y0 + a * np.exp(-b * np.exp(-k_g * (t - t_i)))


def fit_gompertz(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Fit the Gompertz curve to one trajectory; returns ``(params, ok)``.

    Initial guess and bounds are derived from the data (no domain-specific magic
    constants). ``params`` follows the order of :data:`GOMPERTZ_PARAMS` (NaNs on
    failure).
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
        y0_est = y_min * 0.9

        y_clean = np.asarray(winsorize(y, limits=[0.05, 0.05]))
        cut = int(np.argmax(y_clean))

        if cut > 0:
            design = np.column_stack((np.ones(cut + 1), t[: cut + 1]))
            slope = np.linalg.lstsq(design, np.log(y[: cut + 1] - y0_est + 1e-3), rcond=None)[0][1]
        else:
            slope = 0.01

        p0 = [y_range, 1.0, float(t[cut]), max(float(slope), 0.01), max(y0_est, 0.0)]
        lower = [y_range * 0.1, 1e-3, float(t.min()), 1e-3, 0.0]
        upper = [y_range * 3.0, 1e4, float(t.max()), 5.0, max(y_min, 1e-3)]
        p0 = np.clip(p0, lower, upper)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params, _ = curve_fit(
                gompertz, t, y, p0=p0, bounds=(lower, upper), method="trf", maxfev=10000
            )
        return np.asarray(params, dtype=float), True
    except Exception:
        return nan_params, False


def _r2(t: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
    """Coefficient of determination of a Gompertz fit."""
    pred = gompertz(t, *params)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ---------------------------------------------------------------------------
# Gompertz as personalised TSFEL features
# ---------------------------------------------------------------------------
# The parameter-features share a single curve fit per signal via this cache; the
# extraction is per-signal so a small dict keyed by the raw bytes is enough.
_FIT_CACHE: dict[bytes, tuple[np.ndarray, bool]] = {}


def _gompertz_fit_for(signal: np.ndarray) -> tuple[np.ndarray, bool]:
    """Cached Gompertz fit for a (uniformly daily-sampled) 1-D signal."""
    arr = np.asarray(signal, dtype=float)
    key = arr.tobytes()
    cached = _FIT_CACHE.get(key)
    if cached is None:
        t = np.arange(arr.size, dtype=float)  # uniform daily sampling -> t = 0..T-1
        cached = fit_gompertz(t, arr)
        if len(_FIT_CACHE) > 8192:
            _FIT_CACHE.clear()
        _FIT_CACHE[key] = cached
    return cached


def _make_gompertz_feature(index: int, name: str):
    """Build a single-value custom TSFEL feature for one Gompertz parameter."""

    @set_domain("domain", "temporal")
    def _feature(signal, parameters=None):
        params, ok = _gompertz_fit_for(signal)
        return float(params[index]) if ok else np.nan

    _feature.__name__ = f"gompertz_{name}"
    _feature.__doc__ = f"Gompertz growth-curve parameter '{name}' (custom TSFEL feature)."
    return _feature


# One personalised TSFEL feature per Gompertz parameter, plus the fit R^2.
GOMPERTZ_TSFEL_FEATURES = {
    name: _make_gompertz_feature(i, name) for i, name in enumerate(GOMPERTZ_PARAMS)
}


@set_domain("domain", "temporal")
def gompertz_r2(signal, parameters=None):
    """Goodness-of-fit (R^2) of the Gompertz curve (custom TSFEL feature)."""
    arr = np.asarray(signal, dtype=float)
    params, ok = _gompertz_fit_for(arr)
    if not ok:
        return np.nan
    return _r2(np.arange(arr.size, dtype=float), arr, params)


def gompertz_features(df: pd.DataFrame, channel: str = GOMPERTZ_VCD_CHANNEL) -> pd.DataFrame:
    """Per-experiment Gompertz features (parameters + R^2 + ok flag) for one channel."""
    rows: dict[str, dict[str, float]] = {}
    for exp, group in df.groupby(schema.EXP_COL, sort=False):
        group = group.sort_values(schema.TIME_COL)
        signal = group[channel].to_numpy(dtype=float)

        feat = {
            f"gompertz_{channel}_{name}": fn(signal) for name, fn in GOMPERTZ_TSFEL_FEATURES.items()
        }
        feat[f"gompertz_{channel}_r2"] = gompertz_r2(signal)
        feat[f"gompertz_{channel}_ok"] = float(_gompertz_fit_for(signal)[1])
        rows[exp] = feat

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = schema.EXP_COL
    return out


# ---------------------------------------------------------------------------
# TSFEL features
# ---------------------------------------------------------------------------
# A curated, interpretable subset of TSFEL's statistical and temporal domains,
# chosen to be meaningful for concentration/growth trajectories and to keep the
# feature count modest (~25 per channel). Spectral/fractal domains are omitted:
# little signal on ~10-point daily series and harder to interpret.
_TSFEL_KEEP: frozenset[str] = frozenset(
    {
        # temporal
        "Area under the curve",
        "Autocorrelation",
        "Centroid",
        "Mean diff",
        "Mean absolute diff",
        "Slope",
        "Positive turning points",
        "Negative turning points",
        "Zero crossing rate",
        "Signal distance",
        "Neighbourhood peaks",
        # statistical
        "Absolute energy",
        "Entropy",
        "Interquartile range",
        "Kurtosis",
        "Max",
        "Mean",
        "Mean absolute deviation",
        "Median",
        "Min",
        "Peak to peak distance",
        "Root mean square",
        "Skewness",
        "Standard deviation",
        "Variance",
    }
)


def _tsfel_config() -> dict:
    """Curated TSFEL config: kept statistical + temporal features only."""
    full = tsfel.get_features_by_domain()
    return {
        domain: {name: spec for name, spec in full[domain].items() if name in _TSFEL_KEEP}
        for domain in ("statistical", "temporal")
    }


def tsfel_features(df: pd.DataFrame, channels: list[str] | None = None) -> pd.DataFrame:
    """Per-experiment TSFEL features for each requested channel.

    Defaults to the measured-state (``X:``) channels. One extraction call per
    experiment over all channels; TSFEL prefixes each output column with the
    channel's positional index, which we map back to its name.
    """
    if channels is None:
        channels = schema.state_columns(df)
    config = _tsfel_config()

    rows: dict[str, dict[str, float]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for exp, group in df.groupby(schema.EXP_COL, sort=False):
            group = group.sort_values(schema.TIME_COL)
            signals = group[channels].to_numpy(dtype=float)
            out = tsfel.time_series_features_extractor(config, signals, fs=1, verbose=0)

            feat: dict[str, float] = {}
            for col in out.columns:
                idx_str, fname = col.split("_", 1)
                channel = channels[int(idx_str)]
                value = out.iloc[0][col]
                # TSFEL returns None where a feature is undefined for a series;
                # NaN is fine — XGBoost handles it natively.
                feat[f"tsfel_{channel}_{fname}"] = np.nan if value is None else float(value)
            rows[exp] = feat

    out_df = pd.DataFrame.from_dict(rows, orient="index")
    out_df.index.name = schema.EXP_COL
    return out_df


# ---------------------------------------------------------------------------
# Static + meta features
# ---------------------------------------------------------------------------
def static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through ``Z:`` design scalars plus observed duration and length."""
    static_cols = schema.static_columns(df)
    rows: dict[str, dict[str, float]] = {}
    for exp, group in df.groupby(schema.EXP_COL, sort=False):
        group = group.sort_values(schema.TIME_COL)
        time = group[schema.TIME_COL].to_numpy(dtype=float)
        feat = {
            col: (float(group[col].dropna().iloc[0]) if group[col].notna().any() else np.nan)
            for col in static_cols
        }
        feat["duration_observed"] = float(time.max()) if time.size else np.nan
        feat["n_timepoints"] = float(time.size)
        rows[exp] = feat

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = schema.EXP_COL
    return out


# ---------------------------------------------------------------------------
# Bioprocess accounting features
# ---------------------------------------------------------------------------
# Trajectory integral shared with the sequence loaders; see data_preprocessing.
_auc = dp._auc


def substrate_consumption_features(
    df: pd.DataFrame,
    channels: tuple[str, ...] = FED_SUBSTRATE_CHANNELS,
    feed_map: dict[str, str] = MATCHING_FEEDS,
) -> pd.DataFrame:
    """Biologically motivated concentration/feed features for XGBoost.

    TSFEL already provides concentration AUCs for the ``X:`` channels, so this
    custom feature family only adds feed-accounting quantities that combine a
    measured concentration with its matching ``W:`` feed. In the provided data
    this applies to glucose and glutamine.
    """
    rows: dict[str, dict[str, float]] = {}
    for exp, group in df.groupby(schema.EXP_COL, sort=False):
        group = group.sort_values(schema.TIME_COL)
        time = group[schema.TIME_COL].to_numpy(dtype=float)

        feat: dict[str, float] = {}
        for channel in channels:
            if channel not in group:
                continue
            feed_col = feed_map.get(channel)
            if feed_col is None or feed_col not in group.columns:
                continue

            values = group[channel].to_numpy(dtype=float)
            initial = float(values[0]) if values.size else np.nan
            final = float(values[-1]) if values.size else np.nan
            total_fed = _auc(time, group[feed_col].to_numpy(dtype=float))
            initial_plus_fed = initial + total_fed
            apparent_consumed = initial_plus_fed - final

            prefix = f"bio_{channel}"
            feat[f"{prefix}_initial"] = initial
            feat[f"{prefix}_final"] = final
            feat[f"{prefix}_total_fed"] = total_fed
            feat[f"{prefix}_initial_plus_fed"] = initial_plus_fed
            feat[f"{prefix}_apparent_consumed"] = apparent_consumed
        rows[exp] = feat

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = schema.EXP_COL
    return out


def _total_cell_density(vcd: np.ndarray, lysed_fraction: np.ndarray) -> np.ndarray:
    """Estimate total cell density from viable density and lysed fraction."""
    vcd = np.asarray(vcd, dtype=float)
    lysed_fraction = np.asarray(lysed_fraction, dtype=float)
    valid = np.isfinite(vcd) & np.isfinite(lysed_fraction) & (lysed_fraction >= 0.0)
    valid &= lysed_fraction < 1.0

    total = np.full_like(vcd, np.nan, dtype=float)
    total[valid] = vcd[valid] / (1.0 - lysed_fraction[valid])
    return total


def cell_density_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cell-population features derived from VCD and lysed fraction.

    If ``X:Lysed`` is interpreted as the fraction of total cells that have lysed,
    then ``X:VCD`` is the viable fraction of the total population:

    ``total_cell_density = X:VCD / (1 - X:Lysed)``.
    """
    rows: dict[str, dict[str, float]] = {}
    for exp, group in df.groupby(schema.EXP_COL, sort=False):
        group = group.sort_values(schema.TIME_COL)
        feat: dict[str, float] = {}
        if "X:VCD" in group.columns and "X:Lysed" in group.columns:
            time = group[schema.TIME_COL].to_numpy(dtype=float)
            total = _total_cell_density(
                group["X:VCD"].to_numpy(dtype=float),
                group["X:Lysed"].to_numpy(dtype=float),
            )
            finite = np.isfinite(total)
            if finite.any():
                feat[f"{TOTAL_CELL_FEATURE_PREFIX}_initial"] = float(total[0])
                feat[f"{TOTAL_CELL_FEATURE_PREFIX}_final"] = float(total[-1])
                feat[f"{TOTAL_CELL_FEATURE_PREFIX}_max"] = float(np.nanmax(total))
                feat[f"{TOTAL_CELL_FEATURE_PREFIX}_auc"] = _auc(time, total)
            else:
                feat[f"{TOTAL_CELL_FEATURE_PREFIX}_initial"] = np.nan
                feat[f"{TOTAL_CELL_FEATURE_PREFIX}_final"] = np.nan
                feat[f"{TOTAL_CELL_FEATURE_PREFIX}_max"] = np.nan
                feat[f"{TOTAL_CELL_FEATURE_PREFIX}_auc"] = np.nan
        rows[exp] = feat

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = schema.EXP_COL
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_baseline_features(
    df: pd.DataFrame, tsfel_channels: list[str] | None = None
) -> pd.DataFrame:
    """Assemble the full baseline feature matrix (one row per experiment)."""
    static = static_features(df)
    tsfel_feats = tsfel_features(df, tsfel_channels)
    gompertz = gompertz_features(df, GOMPERTZ_VCD_CHANNEL)
    bio = substrate_consumption_features(df)
    cells = cell_density_features(df)

    features = pd.concat([static, tsfel_feats, gompertz, bio, cells], axis=1)
    features.index.name = schema.EXP_COL
    return features


def build_baseline_dataset(
    data_path, targets_path=None, tsfel_channels: list[str] | None = None
) -> dp.TabularDataset:
    """End-to-end: raw CSV(s) -> baseline features aligned with targets."""
    parsed = dp.read_inputs(data_path)
    features = build_baseline_features(parsed, tsfel_channels)
    return dp.assemble_tabular(features, targets_path)
