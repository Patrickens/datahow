"""Shared plotting helpers for the exploration notebook and the README figures.

Each ``plot_*`` function returns a matplotlib ``Figure`` so it can be rendered
inline in the marimo notebook *and* saved to ``assets/`` for the README from one
source of truth. Regenerate all README figures with::

    uv run python -m titer_prediction.plotting
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.model_selection import KFold, cross_val_predict

from . import data_preprocessing as dp
from . import features as feats
from . import regression as reg
from . import schema

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
FIGURES_DIR = REPO_ROOT / "figures"

TRAIN_DATA = DATA_DIR / "datahow_interview_train_data.csv"
TRAIN_TARGETS = DATA_DIR / "datahow_interview_train_targets.csv"

_CMAP = plt.get_cmap("viridis")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def load_train() -> tuple[pd.DataFrame, pd.Series]:
    """Long training frame (Z: forward-filled) and per-experiment titer."""
    df = dp.read_inputs(TRAIN_DATA)
    targets = dp.read_targets(TRAIN_TARGETS)
    return df, targets


def _titer_norm(targets: pd.Series) -> Normalize:
    return Normalize(vmin=float(targets.min()), vmax=float(targets.max()))


def _add_titer_colorbar(fig, norm: Normalize, label: str = "Final titer") -> None:
    sm = ScalarMappable(norm=norm, cmap=_CMAP)
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, label=label, fraction=0.046, pad=0.02)


# ---------------------------------------------------------------------------
# 0. Target distribution
# ---------------------------------------------------------------------------
def plot_titer_distribution(targets: pd.Series):
    """Violin + jittered strip of the final-titer target across experiments."""
    values = targets.to_numpy(dtype=float)
    norm = _titer_norm(targets)

    fig, ax = plt.subplots(figsize=(5.5, 6), constrained_layout=True)
    parts = ax.violinplot(
        values, showmeans=False, showmedians=False, showextrema=False, widths=0.8
    )
    for body in parts["bodies"]:
        body.set_facecolor("#b3cde3")
        body.set_alpha(0.4)
        body.set_edgecolor("grey")

    # Jittered points, coloured by titer.
    rng = np.random.default_rng(0)
    jitter = 1 + (rng.random(values.size) - 0.5) * 0.18
    ax.scatter(
        jitter, values, c=_CMAP(norm(values)), s=28, edgecolor="k", linewidth=0.3, zorder=3
    )

    median, mean = float(np.median(values)), float(np.mean(values))
    ax.axhline(median, ls="--", color="#444444", lw=1, label=f"median = {median:.0f}")
    ax.axhline(mean, ls=":", color="#d62728", lw=1, label=f"mean = {mean:.0f}")
    ax.set(
        xticks=[1],
        xticklabels=[f"train ({values.size} experiments)"],
        ylabel="Final titer",
        title="Distribution of the target (final titer)",
    )
    ax.legend(fontsize=9, loc="upper right")
    return fig


# ---------------------------------------------------------------------------
# 1. Input time courses
# ---------------------------------------------------------------------------
def _plot_titer_colored(ax, df, targets, column, norm, step=False):
    """One faint line per experiment, coloured by that run's final titer.

    ``step=True`` draws a rectilinear staircase (``steps-post``) — the correct
    representation for the piecewise-constant control inputs, where a value holds
    until the next day and then jumps.
    """
    drawstyle = "steps-post" if step else "default"
    for exp, group in df.groupby(schema.EXP_COL, sort=False):
        group = group.sort_values(schema.TIME_COL)
        ax.plot(
            group[schema.TIME_COL], group[column],
            color=_CMAP(norm(targets.get(exp, np.nan))), alpha=0.6, lw=1.0,
            drawstyle=drawstyle,
        )


def _plot_multichannel(ax, df, columns, colors):
    """Overlay several channels (fixed colour per channel) across experiments."""
    for column, color in zip(columns, colors, strict=True):
        for _, group in df.groupby(schema.EXP_COL, sort=False):
            group = group.sort_values(schema.TIME_COL)
            ax.plot(group[schema.TIME_COL], group[column], color=color, alpha=0.25, lw=0.8)
    handles = [Line2D([], [], color=c, label=col) for col, c in zip(columns, colors, strict=True)]
    ax.legend(handles=handles, fontsize=8, loc="best")


def plot_state_timecourses(df: pd.DataFrame, targets: pd.Series):
    """State trajectories (``X:``) grouped by biological role."""
    norm = _titer_norm(targets)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    _plot_titer_colored(axes[0, 0], df, targets, "X:VCD", norm)
    axes[0, 0].set(title="Viable cell density (growth)", ylabel="VCD")

    _plot_multichannel(axes[0, 1], df, ["X:Glc", "X:Gln"], ["#1f77b4", "#ff7f0e"])
    axes[0, 1].set(title="Substrates (nutrients consumed)", ylabel="concentration")

    _plot_multichannel(axes[1, 0], df, ["X:Lac", "X:Amm"], ["#d62728", "#9467bd"])
    axes[1, 0].set(title="Byproducts (waste metabolites)", ylabel="concentration")

    _plot_titer_colored(axes[1, 1], df, targets, "X:Lysed", norm)
    axes[1, 1].set(title="Lysed-cell fraction", ylabel="lysed")

    for ax in axes.flat:
        ax.set_xlabel("Time [day]")
    fig.suptitle("Measured state trajectories (100 experiments)", fontweight="bold")
    _add_titer_colorbar(fig, norm)
    return fig


def plot_control_timecourses(df: pd.DataFrame, targets: pd.Series):
    """Control trajectories (``W:``) — note the discontinuous feed switches."""
    norm = _titer_norm(targets)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [
        ("W:temp", "Temperature setpoint"),
        ("W:pH", "pH setpoint"),
        ("W:FeedGlc", "Glucose feed (on/off)"),
        ("W:FeedGln", "Glutamine feed (on/off)"),
    ]
    for ax, (column, title) in zip(axes.flat, panels, strict=True):
        _plot_titer_colored(ax, df, targets, column, norm, step=True)
        ax.set(title=title, xlabel="Time [day]", ylabel=column)
    fig.suptitle("Control inputs — feeds are step-like (challenge #2)", fontweight="bold")
    _add_titer_colorbar(fig, norm)
    return fig


# ---------------------------------------------------------------------------
# 2. Preprocessing: Gompertz
# ---------------------------------------------------------------------------
def plot_gompertz_examples(df: pd.DataFrame, targets: pd.Series, n_examples: int = 6):
    """Gompertz fits on VCD for a spread of experiments across the titer range."""
    order = targets.sort_values().index.tolist()
    picks = [order[int(round(i))] for i in np.linspace(0, len(order) - 1, n_examples)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, exp in zip(axes.flat, picks, strict=True):
        group = df[df[schema.EXP_COL] == exp].sort_values(schema.TIME_COL)
        t = group[schema.TIME_COL].to_numpy(float)
        y = group["X:VCD"].to_numpy(float)
        params, ok = feats.fit_gompertz(t, y)

        ax.scatter(t, y, s=20, color="#333333", zorder=3, label="VCD data")
        if ok:
            t_dense = np.linspace(t.min(), t.max(), 100)
            ax.plot(t_dense, feats.gompertz(t_dense, *params), color="#d62728", label="Gompertz")
            r2 = feats._r2(t, y, params)
            ax.set_title(f"{exp}  (titer={targets[exp]:.0f}, R²={r2:.3f})", fontsize=9)
        ax.set(xlabel="Time [day]", ylabel="VCD")
        ax.legend(fontsize=7)
    fig.suptitle("Gompertz growth-curve fits on VCD", fontweight="bold")
    return fig


def plot_gompertz_signal(df: pd.DataFrame, targets: pd.Series):
    """Interpretable Gompertz parameters vs. titer (do they carry signal?)."""
    gomp = feats.gompertz_features(df, "X:VCD")
    y = targets.reindex(gomp.index)

    pairs = [
        ("gompertz_X:VCD_a", "amplitude a (max growth)"),
        ("gompertz_X:VCD_k_g", "growth rate k_g"),
        ("gompertz_X:VCD_t_i", "inflection time t_i"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, (col, label) in zip(axes, pairs, strict=True):
        ax.scatter(gomp[col], y, s=18, alpha=0.7, color="#2ca02c")
        corr = np.corrcoef(gomp[col], y)[0, 1]
        ax.set(xlabel=label, ylabel="Final titer", title=f"corr = {corr:+.2f}")
    fig.suptitle("Gompertz parameters vs. titer (interpretable features)", fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# 3. Regression
# ---------------------------------------------------------------------------
def baseline_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """Full baseline feature matrix and aligned targets."""
    ds = feats.build_baseline_dataset(TRAIN_DATA, TRAIN_TARGETS)
    return ds.features, ds.targets


def plot_cv_predictions(X: pd.DataFrame | None = None, y: pd.Series | None = None):
    """Out-of-fold predicted vs. actual titer + residuals for the baseline.

    The scatter uses out-of-fold predictions from one representative 5-fold split;
    the headline R²/RMSE in the title are the **repeated** 5-fold CV means (the
    same protocol reported in the README/CLI), so the figure and text agree.
    """
    if X is None or y is None:
        X, y = baseline_matrix()
    oof = cross_val_predict(reg.build_model(), X, y, cv=KFold(5, shuffle=True, random_state=0))
    resid = y.to_numpy() - oof

    # Robust headline metrics: repeated 5-fold CV (matches regression.py / README).
    cv = reg.cross_validate(X, y)["xgboost"]
    r2, rmse = cv["r2"], cv["rmse"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    lo, hi = float(min(y.min(), oof.min())), float(max(y.max(), oof.max()))
    axes[0].plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
    axes[0].scatter(y, oof, s=22, alpha=0.7, color="#1f77b4")
    axes[0].set(
        xlabel="Actual titer", ylabel="Predicted titer (out-of-fold)",
        title=f"XGBoost baseline — repeated 5-fold CV (R²={r2:.2f}, RMSE={rmse:.0f})",
    )
    axes[1].axhline(0, ls="--", color="grey", lw=1)
    axes[1].scatter(oof, resid, s=22, alpha=0.7, color="#ff7f0e")
    axes[1].set(xlabel="Predicted titer", ylabel="Residual", title="Residuals")
    return fig


def plot_feature_importance(X: pd.DataFrame | None = None, y: pd.Series | None = None, top: int = 15):
    """Top XGBoost feature importances (which engineered features matter)."""
    if X is None or y is None:
        X, y = baseline_matrix()
    model = reg.build_model()
    model.fit(X, y)
    importances = pd.Series(model.regressor_.feature_importances_, index=X.columns)
    top_feats = importances.sort_values(ascending=False).head(top).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.barh(top_feats.index, top_feats.to_numpy(), color="#4c72b0")
    ax.set(xlabel="Importance (gain)", title=f"Top {top} baseline features")
    ax.tick_params(axis="y", labelsize=8)
    return fig


# ---------------------------------------------------------------------------
# Generate all README figures
# ---------------------------------------------------------------------------
def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    FIGURES_DIR.mkdir(exist_ok=True)
    df, targets = load_train()
    X, yt = baseline_matrix()  # built once, reused for the regression figures

    figures = {
        "titer_distribution.png": plot_titer_distribution(targets),
        "input_state_timecourses.png": plot_state_timecourses(df, targets),
        "input_control_timecourses.png": plot_control_timecourses(df, targets),
        "gompertz_fits.png": plot_gompertz_examples(df, targets),
        "gompertz_signal.png": plot_gompertz_signal(df, targets),
        "regression_cv.png": plot_cv_predictions(X, yt),
        "feature_importance.png": plot_feature_importance(X, yt),
    }
    for name, fig in figures.items():
        out = FIGURES_DIR / name
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
