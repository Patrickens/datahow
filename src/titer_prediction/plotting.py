"""Shared plotting helpers for the exploration notebook and the README figures.

Each ``plot_*`` function returns a matplotlib ``Figure`` so it can be rendered
inline in the marimo notebook *and* saved to ``assets/`` for the README from one
source of truth. Regenerate all README figures with::

    uv run python -m titer_prediction.plotting
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
XGB_BEST_METADATA = REPO_ROOT / "artifacts" / "xgb_best_metadata.json"
CDE_BEST_METADATA = REPO_ROOT / "artifacts" / "cde_best_metadata.json"
FEATURE_IMPORTANCE_TABLE = REPO_ROOT / "artifacts" / "feature_importance.csv"
CDE_TRAINING_HISTORY = REPO_ROOT / "artifacts" / "cde_training_history.csv"

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


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _best_xgb_cv_params() -> dict:
    metadata = _load_json(XGB_BEST_METADATA)
    params = dict(metadata["best_config"])
    params["random_state"] = int(metadata["seed"])
    params["n_jobs"] = -1
    return params


def _best_xgb_refit_params() -> dict:
    return dict(_load_json(XGB_BEST_METADATA)["refit_params"])


def _pretty_feature_label(name: str) -> str:
    """Readable labels for feature-importance plots."""
    label = name
    replacements = {
        "bio_total_cell_density_": "Total cell density ",
        "bio_X:Glc_": "Glc ",
        "bio_X:Gln_": "Gln ",
        "tsfel_X:": "",
        "gompertz_X:": "Gompertz ",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return label.replace("_", " ")


# ---------------------------------------------------------------------------
# 0. Target distribution
# ---------------------------------------------------------------------------
def plot_titer_distribution(targets: pd.Series):
    """Violin + jittered strip of the final-titer target across experiments."""
    values = targets.to_numpy(dtype=float)
    norm = _titer_norm(targets)

    fig, ax = plt.subplots(figsize=(5.5, 6), constrained_layout=True)
    parts = ax.violinplot(values, showmeans=False, showmedians=False, showextrema=False, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("#b3cde3")
        body.set_alpha(0.4)
        body.set_edgecolor("grey")

    # Jittered points, coloured by titer.
    rng = np.random.default_rng(0)
    jitter = 1 + (rng.random(values.size) - 0.5) * 0.18
    ax.scatter(jitter, values, c=_CMAP(norm(values)), s=28, edgecolor="k", linewidth=0.3, zorder=3)

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
            group[schema.TIME_COL],
            group[column],
            color=_CMAP(norm(targets.get(exp, np.nan))),
            alpha=0.6,
            lw=1.0,
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
    metadata = _load_json(XGB_BEST_METADATA)
    oof = cross_val_predict(
        reg.build_model(_best_xgb_cv_params()),
        X,
        y,
        cv=KFold(5, shuffle=True, random_state=int(metadata["seed"])),
    )
    resid = y.to_numpy() - oof

    # Robust headline metrics from the final repeated-CV pass in the sweep metadata.
    cv = metadata["final_cv"]["xgboost"]
    r2, rmse = cv["r2"], cv["rmse"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    lo, hi = float(min(y.min(), oof.min())), float(max(y.max(), oof.max()))
    axes[0].plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
    axes[0].scatter(y, oof, s=22, alpha=0.7, color="#1f77b4")
    axes[0].set(
        xlabel="Actual titer",
        ylabel="Predicted titer (out-of-fold)",
        title=f"XGBoost baseline — repeated 5-fold CV (R²={r2:.2f}, RMSE={rmse:.0f})",
    )
    axes[1].axhline(0, ls="--", color="grey", lw=1)
    axes[1].scatter(oof, resid, s=22, alpha=0.7, color="#ff7f0e")
    axes[1].set(xlabel="Predicted titer", ylabel="Residual", title="Residuals")
    return fig


def feature_importance_table(top: int = 15, regenerate: bool = False) -> pd.DataFrame:
    """Top XGBoost feature importances as a table (``feature``, ``label``, ``importance``).

    The full ranking is cached to ``artifacts/feature_importance.csv`` so the
    notebook need not refit the model on every run. Pass ``regenerate=True`` (or
    delete the cache) to refit the best model and overwrite it.
    """
    if regenerate or not FEATURE_IMPORTANCE_TABLE.exists():
        X, y = baseline_matrix()
        model = reg.build_model(_best_xgb_refit_params())
        model.fit(X, y)
        importances = pd.Series(model.regressor_.feature_importances_, index=X.columns)
        table = (
            importances.sort_values(ascending=False)
            .rename_axis("feature")
            .reset_index(name="importance")
        )
        table.insert(1, "label", [_pretty_feature_label(n) for n in table["feature"]])
        FEATURE_IMPORTANCE_TABLE.parent.mkdir(exist_ok=True)
        table.to_csv(FEATURE_IMPORTANCE_TABLE, index=False)
    else:
        table = pd.read_csv(FEATURE_IMPORTANCE_TABLE)
    return table.head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CDE path illustration (toy example)
# ---------------------------------------------------------------------------
def _toy_cde_path():
    """A tiny synthetic experiment and its mixed CDE path, for illustration.

    Reuses ``cde.make_mixed_cde_path`` so the pictures match the real construction.
    """
    from . import cde  # lazy: only this illustration pulls in the JAX stack

    t = np.array([0.0, 1, 2, 3, 4, 5])
    w = np.array([0.0, 0, 5, 5, 0, 0])  # feed: switches on at day 2, off at day 4
    x = np.array([1.0, 2, 4, 6, 7, 7.5])  # a continuous-ish state
    real = np.column_stack([t, w, x]).astype(np.float32)
    ys = np.vstack([real, np.repeat(real[-1:], 3, axis=0)])
    s, path = cde.make_mixed_cde_path(ys, n_w=1)
    padding_start_s = np.asarray(s)[2 * len(real) - 2]
    return t, w, x, np.asarray(s), np.asarray(path), float(padding_start_s)


def plot_interpolation_comparison():
    """Toy feed control: linear interpolation fabricates ramps; step does not."""
    t, w, _, _, _, _ = _toy_cde_path()
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    tt = np.linspace(t.min(), t.max(), 400)
    ax.plot(
        tt, np.interp(tt, t, w), color="#d62728", lw=1.5, ls="--", label="linear — fabricates ramps"
    )
    ax.step(t, w, where="post", color="#1f77b4", lw=2, label="step (used for W: controls)")
    ax.scatter(t, w, color="k", zorder=3, s=30, label="daily samples")
    ax.set(
        xlabel="real time [day]",
        ylabel="feed W",
        title="A step control: linear vs step interpolation",
    )
    ax.legend(fontsize=8)
    return fig


def plot_path_parameter():
    """Real time t(s) and control W(s) against the artificial path parameter s."""
    _, _, _, s, path, padding_start_s = _toy_cde_path()
    time_s, w_s = path[:, 0], path[:, 1]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    axes[0].plot(s, time_s, "-o", color="#2ca02c")
    axes[0].set(ylabel="real time  t(s)", title="Real time is a channel; s is the solver clock")
    axes[1].plot(s, w_s, "-o", color="#1f77b4")
    axes[1].set(xlabel="path parameter s  (the solver clock)", ylabel="feed  W(s)")
    # Shade the control-jump segments (where W moves but real time is flat).
    for i in np.where(np.abs(np.diff(w_s)) > 1e-9)[0]:
        for ax in axes:
            ax.axvspan(s[i], s[i + 1], color="#1f77b4", alpha=0.15)
    for ax in axes:
        ax.axvspan(padding_start_s, s[-1], color="#ffbf00", alpha=0.18)
        ax.axvline(padding_start_s, color="#8c6d1f", lw=1, ls=":")
    axes[0].text(
        padding_start_s,
        time_s[-1],
        " flat padding\n C(s)=C(S)",
        ha="left",
        va="top",
        fontsize=8,
        color="#6b5500",
    )
    axes[1].legend(
        handles=[
            Patch(facecolor="#1f77b4", alpha=0.15, label="control-jump segment"),
            Patch(facecolor="#ffbf00", alpha=0.18, label="flat padded tail"),
        ],
        fontsize=8,
        loc="best",
    )
    return fig


def plot_cde_toy_state():
    """Toy hidden state under a fixed field: it updates on flows AND control jumps."""
    _, _, _, s, path, padding_start_s = _toy_cde_path()
    # A trivial constant vector field f, so dh = f . dC(s) (a linear CDE).
    f = np.array([0.2, 0.5, 0.3])  # weights on [time, W, X] increments
    dh = np.diff(path, axis=0) @ f
    h = np.concatenate([[0.0], np.cumsum(dh)])
    w_s = path[:, 1]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(s, h, "-o", color="#9467bd")
    for i in np.where(np.abs(np.diff(w_s)) > 1e-9)[0]:
        ax.axvspan(s[i], s[i + 1], color="#1f77b4", alpha=0.15)
    ax.axvspan(padding_start_s, s[-1], color="#ffbf00", alpha=0.18)
    ax.set(
        xlabel="path parameter s",
        ylabel="toy hidden state h(s)",
        title="Hidden state updates on time increments AND control jumps\n"
        "(blue shaded = control jumps; gold shaded = flat padding)",
    )
    return fig


def cde_training_history(epochs: int | None = None, regenerate: bool = False) -> pd.DataFrame:
    """Per-epoch training history for the selected neural CDE config.

    Cached to ``artifacts/cde_training_history.csv`` so the notebook / HTML export
    need not retrain the CDE on every run (a few minutes on the JAX stack). Pass
    ``regenerate=True`` (or delete the cache) to retrain and overwrite.
    """
    if not regenerate and CDE_TRAINING_HISTORY.exists():
        return pd.read_csv(CDE_TRAINING_HISTORY)

    from . import cde  # lazy: only this illustration pulls in the JAX stack

    metadata = _load_json(CDE_BEST_METADATA)
    cfg = metadata["best_config"]
    epochs = int(cfg["epochs"] if epochs is None else epochs)
    _, _, history = cde.train(
        TRAIN_DATA,
        TRAIN_TARGETS,
        hidden_size=int(cfg["hidden_size"]),
        width=int(cfg["width"]),
        depth=int(cfg["depth"]),
        epochs=epochs,
        lr=float(cfg["lr"]),
        seed=int(metadata["seed"]),
        refit_all=False,
    )
    hist = pd.DataFrame(history)
    CDE_TRAINING_HISTORY.parent.mkdir(exist_ok=True)
    hist.to_csv(CDE_TRAINING_HISTORY, index=False)
    return hist


def plot_cde_training_curves(epochs: int | None = None, regenerate: bool = False):
    """Plot the neural CDE learning curves from cached (or freshly trained) history."""
    hist = cde_training_history(epochs=epochs, regenerate=regenerate)
    plot_hist = hist[hist["epoch"] > 0].copy()
    n_epochs = int(hist["epoch"].max()) + 1  # last logged epoch is epochs - 1

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(
        plot_hist["epoch"],
        plot_hist["train_rmse"],
        "-o",
        ms=3,
        color="#1f77b4",
        label="train RMSE",
    )
    if "val_rmse" in plot_hist:
        ax.plot(
            plot_hist["epoch"],
            plot_hist["val_rmse"],
            "-o",
            ms=3,
            color="#d62728",
            label="validation RMSE",
        )
    # Log scale keeps the early high-RMSE epochs from flattening the tail.
    ax.set_yscale("log")
    ax.set(
        xlabel="epoch",
        ylabel="RMSE (titer units, log scale)",
        title=f"Best neural CDE learning curves ({n_epochs} epochs; epoch 0 omitted)",
    )

    lines = ax.get_lines()
    if "val_r2" in plot_hist:
        ax2 = ax.twinx()
        ax2.plot(
            plot_hist["epoch"],
            plot_hist["val_r2"],
            "-s",
            ms=3,
            color="#2ca02c",
            label="validation R² (mAb titer)",
        )
        ax2.set_ylabel("validation R²  (mAb titer)", color="#2ca02c")
        ax2.tick_params(axis="y", labelcolor="#2ca02c")
        ax2.set_ylim(-0.05, 1.0)
        ax2.axhline(0.0, color="#2ca02c", lw=0.8, ls=":", alpha=0.5)
        lines = lines + ax2.get_lines()

    lines = [ln for ln in lines if not ln.get_label().startswith("_")]
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Generate all README / notebook figures
# ---------------------------------------------------------------------------
FIGURE_NAMES = (
    "titer_distribution.png",
    "input_state_timecourses.png",
    "input_control_timecourses.png",
    "gompertz_fits.png",
    "regression_cv.png",
    "cde_interpolation.png",
    "cde_path_parameter.png",
    "cde_toy_state.png",
    "cde_training_curves.png",
)


def main(force: bool = False) -> None:
    """(Re)generate all figures. Skips figures that already exist unless ``force``.

    With ``force``, the cached tables (feature importances, CDE training history)
    are regenerated too, so the figures reflect the current best models.
    """
    import matplotlib

    matplotlib.use("Agg")
    FIGURES_DIR.mkdir(exist_ok=True)

    needed = [n for n in FIGURE_NAMES if force or not (FIGURES_DIR / n).exists()]
    if not needed:
        print("all figures present (pass --force to regenerate)")
        return

    if force:
        feature_importance_table(regenerate=True)
        cde_training_history(regenerate=True)

    df, targets = load_train()
    X, yt = baseline_matrix()  # built once, reused for the regression figures
    builders = {
        "titer_distribution.png": lambda: plot_titer_distribution(targets),
        "input_state_timecourses.png": lambda: plot_state_timecourses(df, targets),
        "input_control_timecourses.png": lambda: plot_control_timecourses(df, targets),
        "gompertz_fits.png": lambda: plot_gompertz_examples(df, targets),
        "regression_cv.png": lambda: plot_cv_predictions(X, yt),
        "cde_interpolation.png": plot_interpolation_comparison,
        "cde_path_parameter.png": plot_path_parameter,
        "cde_toy_state.png": plot_cde_toy_state,
        "cde_training_curves.png": plot_cde_training_curves,
    }
    for name in needed:
        fig = builders[name]()
        out = FIGURES_DIR / name
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.relative_to(REPO_ROOT)}")


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate the README / notebook figures.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the cached tables and overwrite existing figures.",
    )
    main(force=parser.parse_args(argv).force)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
