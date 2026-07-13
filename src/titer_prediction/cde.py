"""Neural Controlled Differential Equation (diffrax) for titer prediction.

A path-based sequence model that consumes the ragged trajectories from
:mod:`titer_prediction.data_preprocessing` directly. Design choices tied to the
two core challenges:

* **Variable length** is handled natively — the CDE integrates whatever path it
  is given. Batches are padded by holding the last observation (a flat,
  zero-contribution tail), so no masking is needed.
* **Mixed interpolation** encodes the right inductive bias per channel group:
  the ``W:`` controls are **step-interpolated** (feeds and setpoint switches are
  genuinely discontinuous — linear interpolation would fabricate ramps), while
  the ``X:`` state observations are **linearly interpolated** (they are sampled
  from continuous-ish process variables, so a staircase would fabricate jumps).
  Real time is carried as channel 0. Because control jumps have zero duration in
  real time, we integrate over a strictly-increasing *path parameter* ``s`` so
  every jump is seen by the solver (cf. Kidger et al.; Morrill et al. 2021 on
  path parametrisation for neural CDEs). See :func:`make_mixed_cde_path`.

Only the design scalars *without* a ``W:`` counterpart — stirring and dissolved
oxygen (``Z:Stir``, ``Z:DO``) — initialise the hidden state (``z0``). The feed /
pH / temperature design is already carried over time by the ``W:`` path channels,
and the planned duration by the time channel, so passing all of ``Z:`` would only
duplicate it. The terminal hidden state is read out to the (standardised, log)
titer.

CLI mirrors the regression module::

    python -m titer_prediction.cde train \
        --data data/datahow_interview_train_data.csv \
        --targets data/datahow_interview_train_targets.csv \
        --model artifacts/cde.eqx

    python -m titer_prediction.cde predict \
        --data data/datahow_interview_test_data.csv \
        --model artifacts/cde.eqx --out artifacts/cde_predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

from . import data_preprocessing as dp
from . import schema

logger = logging.getLogger(__name__)

# Single seed for all randomness in this model (train/val split + both model
# initialisations). Hardcoded default; ``train`` takes ``seed=`` to override it.
SEED = 0


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------
@dataclass
class Standardizer:
    """Per-feature standardisation stats fit on the training data.

    Channels include real time as channel 0. The target is standardised in
    log1p space (titer is positive and right-skewed).
    """

    channel_mean: np.ndarray
    channel_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    target_mean: float
    target_std: float

    def norm_channels(self, m: np.ndarray) -> np.ndarray:
        return (m - self.channel_mean) / self.channel_std

    def norm_static(self, v: np.ndarray) -> np.ndarray:
        return (v - self.static_mean) / self.static_std

    def norm_target(self, y: np.ndarray) -> np.ndarray:
        return (np.log1p(y) - self.target_mean) / self.target_std

    def denorm_target(self, y_std: np.ndarray) -> np.ndarray:
        return np.expm1(y_std * self.target_std + self.target_mean)


def _raw_matrix(exp: dp.ExperimentSequence) -> np.ndarray:
    """Stack an experiment into ``(t, 1 + n_channels)`` with time as channel 0."""
    return np.concatenate([exp.times[:, None], exp.channels], axis=1)


# Design scalars used to initialise the hidden state. We keep only the ``Z:``
# columns that have NO ``W:`` trajectory counterpart — stirring and dissolved
# oxygen. The feed / pH / temperature design (rates, setpoints, shift and
# start/end days) is already carried over time by the ``W:`` path channels, and
# the planned duration is already the path's time channel; feeding all of ``Z:``
# into z0 would just duplicate that information.
STATIC_INIT_COLS: tuple[str, ...] = ("Z:Stir", "Z:DO")


def _static_matrix(seq: dp.SequenceData) -> tuple[np.ndarray, list[str]]:
    """Select the static-init columns (:data:`STATIC_INIT_COLS`) as an ``(n, k)`` array."""
    idx = [seq.static_names.index(c) for c in STATIC_INIT_COLS if c in seq.static_names]
    if not idx:
        raise ValueError(
            f"None of the static-init columns {STATIC_INIT_COLS} are present; "
            f"available static columns: {seq.static_names}."
        )
    names = [seq.static_names[i] for i in idx]
    matrix = np.stack([e.static[idx] for e in seq.experiments], axis=0)
    return matrix, names


def fit_standardizer(seq: dp.SequenceData) -> Standardizer:
    """Fit standardisation stats over all real (unpadded) timepoints."""
    matrices = [_raw_matrix(e) for e in seq.experiments]
    stacked = np.concatenate(matrices, axis=0)
    statics, _ = _static_matrix(seq)
    targets = np.array([e.target for e in seq.experiments], dtype=float)

    eps = 1e-8
    log_t = np.log1p(targets)
    return Standardizer(
        channel_mean=stacked.mean(0),
        channel_std=stacked.std(0) + eps,
        static_mean=statics.mean(0),
        static_std=statics.std(0) + eps,
        target_mean=float(log_t.mean()),
        target_std=float(log_t.std() + eps),
    )


def build_arrays(
    seq: dp.SequenceData, standardizer: Standardizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Pad + standardise a ragged dataset into batched arrays.

    Returns ``(ys, static, targets)`` where ``ys`` is ``(n, t_max, c)`` with the
    last real observation held over the padded tail (a flat, zero-contribution
    region), ``static`` is ``(n, s)``, and ``targets`` is ``(n,)`` standardised
    log-titer (or ``None`` if unavailable).
    """
    matrices = [standardizer.norm_channels(_raw_matrix(e)) for e in seq.experiments]
    t_max = max(m.shape[0] for m in matrices)
    n, c = len(matrices), matrices[0].shape[1]

    ys = np.zeros((n, t_max, c), dtype=np.float32)
    for i, m in enumerate(matrices):
        length = m.shape[0]
        ys[i, :length] = m
        # Repeat the *entire* last row, including the real-time channel (col 0), so
        # the padded tail is flat: C(s) = C(s_end) => dC = 0 there, and it
        # contributes nothing to the CDE integral (padding is only for batching).
        ys[i, length:] = m[-1]

    static_matrix, _ = _static_matrix(seq)
    static = standardizer.norm_static(static_matrix).astype(np.float32)

    targets = None
    if seq.has_targets:
        raw = np.array([e.target for e in seq.experiments], dtype=float)
        targets = standardizer.norm_target(raw).astype(np.float32)

    return ys, static, targets


# ---------------------------------------------------------------------------
# Mixed control path
# ---------------------------------------------------------------------------
def make_mixed_cde_path(ys: jnp.ndarray, n_w: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build the CDE control path with per-group interpolation, over parameter ``s``.

    Args:
        ys: ``(T, C)`` augmented observations, columns ordered ``[real time,
            W: controls (n_w), X: states]``.
        n_w: number of ``W:`` control channels.

    Returns:
        ``(s, path_aug)`` where ``s`` is a strictly-increasing path parameter of
        length ``2T - 1`` and ``path_aug`` is ``(2T - 1, C)``.

    Construction, per interval between real observations ``k -> k+1``:

    * a **flow** segment advances real time and the ``X:`` states *linearly* while
      the ``W:`` controls are held fixed; then
    * a **jump** segment holds real time and ``X:`` fixed while the ``W:`` controls
      move to their next value (a step).

    So ``W:`` is step-interpolated and ``X:`` (and time) is piecewise linear, all
    within a single path. A flat padded tail (repeated last row) yields zero
    increments in both segment types, so it contributes nothing to the integral.
    Integration runs over ``s`` (strictly increasing) rather than real time, so the
    zero-real-duration control jumps still have a finite extent for the solver.
    """
    length = ys.shape[0]
    # Two interleavings of the row index into the 2T-1 knot grid:
    move = jnp.repeat(jnp.arange(length), 2)[1:]  # time & X: move-then-hold (linear)
    hold = jnp.repeat(jnp.arange(length), 2)[:-1]  # W: hold-then-move (step)

    ys_move, ys_hold = ys[move], ys[hold]
    time = ys_move[:, 0:1]
    controls = ys_hold[:, 1 : 1 + n_w]
    states = ys_move[:, 1 + n_w :]

    path_aug = jnp.concatenate([time, controls, states], axis=1)
    s = jnp.arange(path_aug.shape[0], dtype=ys.dtype)
    return s, path_aug


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CDEFunc(eqx.Module):
    """Neural vector field ``f(z)`` returning an ``(hidden, channels)`` matrix."""

    mlp: eqx.nn.MLP
    hidden_size: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)

    def __init__(self, hidden_size, channels, width, depth, *, key):
        self.hidden_size = hidden_size
        self.channels = channels
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * channels,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            final_activation=jax.nn.tanh,  # bound the field for stable solves
            key=key,
        )

    def __call__(self, t, z, args):
        return self.mlp(z).reshape(self.hidden_size, self.channels)


class NeuralCDE(eqx.Module):
    """Static-initialised neural CDE mapping a path to a scalar (log) titer."""

    initial: eqx.nn.MLP
    func: CDEFunc
    readout: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    n_w: int = eqx.field(static=True)

    def __init__(self, n_static, n_channels, n_w, hidden_size, width, depth, *, key):
        k_init, k_func, k_out = jax.random.split(key, 3)
        self.hidden_size = hidden_size
        self.n_w = n_w
        # z0 is set from the static design AND the initial observation C0 (see
        # __call__), hence in_size = n_static + n_channels.
        self.initial = eqx.nn.MLP(
            n_static + n_channels,
            hidden_size,
            width,
            depth,
            activation=jax.nn.softplus,
            key=k_init,
        )
        self.func = CDEFunc(hidden_size, n_channels, width, depth, key=k_func)
        self.readout = eqx.nn.Linear(hidden_size, 1, key=k_out)

    def __call__(self, ys: jnp.ndarray, static: jnp.ndarray) -> jnp.ndarray:
        # Mixed path: W: step-interpolated, X: linear, time as channel 0. Integrate
        # over the strictly-increasing path parameter s so control jumps are seen.
        s, path = make_mixed_cde_path(ys, self.n_w)
        control = dfx.LinearInterpolation(s, path)
        term = dfx.ControlTerm(self.func, control)

        # Initialise from the static design and the initial observation
        # C0 = ys[0] = [t0, W0..., X0...]. A CDE only sees control *increments*, so
        # the absolute starting state (initial VCD, substrate levels) would be
        # invisible unless injected into z0 here.
        z0 = self.initial(jnp.concatenate([static, ys[0]]))
        # Adaptive higher-order solve: Tsit5 with PID step control subdivides each
        # unit-s segment as needed (finer than one fixed step per knot).
        sol = dfx.diffeqsolve(
            term,
            dfx.Tsit5(),
            t0=s[0],
            t1=s[-1],
            dt0=1.0,
            y0=z0,
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
            saveat=dfx.SaveAt(t1=True),
            max_steps=4096,
        )
        return self.readout(sol.ys[-1])[0]


def _predict_batch(model: NeuralCDE, ys: jnp.ndarray, static: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(model)(ys, static)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class CDEBundle:
    """Serialisable neural-CDE artifact (model + preprocessing + config)."""

    model: NeuralCDE
    standardizer: Standardizer
    channel_names: list[str]
    static_names: list[str]
    config: dict


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(
    data_path: str | Path,
    targets_path: str | Path,
    hidden_size: int = 8,
    width: int = 32,
    depth: int = 1,
    epochs: int = 200,
    lr: float = 3e-3,
    val_frac: float = 0.2,
    seed: int = SEED,
    refit_all: bool = True,
    batch_size: int = 32,
) -> tuple[CDEBundle, dict[str, float], list[dict]]:
    """Train the neural CDE with a validation holdout, then refit on all data.

    A single ``seed`` drives all randomness — the train/validation split, model
    initialisation, and minibatch shuffling. Set ``refit_all=False`` to skip the
    deploy-time refit (used by the sweep, which only needs the holdout metrics).

    Optimisation is fixed: an Adam step with global-norm gradient clipping on a
    warmup+cosine LR schedule, minibatches of ``batch_size`` reshuffled each epoch,
    and an adaptive Tsit5 solve. Standardisation is fit on the **train split only**
    for the honest holdout fit (no leakage into validation) and refit on all data
    for the deployed model. The holdout fit uses **early stopping** on the raw-scale
    validation RMSE; the deployed model is refit for that many epochs.
    """
    seq = dp.build_sequences(data_path, targets_path)
    raw_titer = np.array([e.target for e in seq.experiments], dtype=float)
    # Dynamic channels are ordered W: then X:, so the W: count is the split point.
    n_w = sum(c.startswith(schema.CONTROL_PREFIX) for c in seq.channel_names)

    n = len(seq.experiments)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Fit standardisation on the TRAIN split only (no val/test leakage), then apply
    # those stats to every experiment for the holdout fit.
    train_seq = dp.SequenceData(
        experiments=[seq.experiments[i] for i in train_idx],
        channel_names=seq.channel_names,
        static_names=seq.static_names,
    )
    std_train = fit_standardizer(train_seq)
    ys_np, static_np, y_np = build_arrays(seq, std_train)

    config = {
        "hidden_size": hidden_size,
        "width": width,
        "depth": depth,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "val_frac": val_frac,
        "seed": seed,
        "refit_all": refit_all,
        "n_channels": int(ys_np.shape[2]),
        "n_static": int(static_np.shape[1]),
        "n_w": int(n_w),
        "static_init_cols": _static_matrix(seq)[1],
        "train_indices": [int(i) for i in train_idx],
        "val_indices": [int(i) for i in val_idx],
    }

    def _make_optimizer(total_steps: int):
        """Adam on a warmup+cosine schedule with global-norm gradient clipping.

        ``total_steps`` is the number of optimisation updates (epochs × batches),
        so the schedule spans the real update count under minibatching.
        """
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=max(1, int(0.1 * total_steps)),
            decay_steps=total_steps,
            end_value=lr * 0.1,
        )
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

    def fit(
        train_indices, key, ys_all, static_all, y_all, standardizer, val_indices=None, n_epochs=None
    ):
        """Train a model; return ``(model, history, best_epoch)``.

        Minibatches of ``batch_size`` are reshuffled each epoch, giving
        ``len(train)//batch_size`` updates/epoch. With a validation set, ``model``
        is the **early-stopped** checkpoint (lowest raw-scale val RMSE); ``history``
        is the full per-checkpoint trajectory and ``best_epoch`` marks the pick.
        """
        n_epochs = epochs if n_epochs is None else n_epochs
        model = NeuralCDE(
            static_all.shape[1],
            ys_all.shape[2],
            n_w,
            hidden_size,
            width,
            depth,
            key=key,
        )
        ys_tr = jnp.asarray(ys_all[train_indices])
        static_tr = jnp.asarray(static_all[train_indices])
        y_tr = jnp.asarray(y_all[train_indices])
        raw_tr = raw_titer[train_indices]
        y_tr_np = np.asarray(y_all[train_indices])

        n_train = len(train_indices)
        bs = min(int(batch_size), n_train)
        steps_per_epoch = max(1, n_train // bs)
        optim = _make_optimizer(n_epochs * steps_per_epoch)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        has_val = val_indices is not None and len(val_indices) > 0
        if has_val:
            ys_va = jnp.asarray(ys_all[val_indices])
            static_va = jnp.asarray(static_all[val_indices])
            raw_va = raw_titer[val_indices]

        @eqx.filter_jit
        def step(model, opt_state, ys_b, static_b, y_b):
            def loss_fn(m):
                pred = _predict_batch(m, ys_b, static_b)
                return jnp.mean((pred - y_b) ** 2)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        shuffle_rng = np.random.default_rng(seed)
        history: list[dict] = []
        every = max(1, n_epochs // 30)
        best_model, best_val_rmse, best_epoch = model, np.inf, n_epochs - 1
        for epoch in range(n_epochs):
            order = shuffle_rng.permutation(n_train)
            for b in range(steps_per_epoch):
                bi = order[b * bs : (b + 1) * bs]
                model, opt_state, _ = step(model, opt_state, ys_tr[bi], static_tr[bi], y_tr[bi])
            if epoch % every == 0 or epoch == n_epochs - 1:
                train_pred_std = np.asarray(_predict_batch(model, ys_tr, static_tr))
                record = {
                    "epoch": epoch,
                    "train_mse": float(np.mean((train_pred_std - y_tr_np) ** 2)),
                }
                train_metrics = _metrics(raw_tr, standardizer.denorm_target(train_pred_std))
                record.update(
                    train_rmse=train_metrics["rmse"],
                    train_mae=train_metrics["mae"],
                    train_r2=train_metrics["r2"],
                )
                if has_val:
                    val_pred_std = np.asarray(_predict_batch(model, ys_va, static_va))
                    m = _metrics(raw_va, standardizer.denorm_target(val_pred_std))
                    record.update(val_rmse=m["rmse"], val_mae=m["mae"], val_r2=m["r2"])
                    if record["val_rmse"] < best_val_rmse:
                        best_val_rmse = record["val_rmse"]
                        best_model, best_epoch = model, epoch
                history.append(record)
                logger.info(
                    "  epoch %3d/%d  train MSE=%.4f%s",
                    epoch,
                    n_epochs,
                    record["train_mse"],
                    f"  val R2={record['val_r2']:.3f}" if has_val else "",
                )
        # Return the full history (not truncated at the best epoch) so learning-curve
        # plots show the complete trajectory; selection still uses ``best_model``.
        if has_val:
            return best_model, history, best_epoch
        return model, history, n_epochs - 1

    # One seed for both fits; they train on different data so identical init keys
    # are fine and keep the randomness controlled by a single knob.
    k_val = jax.random.PRNGKey(seed)
    k_full = jax.random.PRNGKey(seed)

    # 1) Fit on the train split; report honest (early-stopped) validation metrics.
    logger.info("Fitting holdout model (%d train / %d val)...", len(train_idx), n_val)
    val_model, history, best_epoch = fit(
        train_idx, k_val, ys_np, static_np, y_np, std_train, val_indices=val_idx
    )
    val_pred_std = np.asarray(
        _predict_batch(val_model, jnp.asarray(ys_np[val_idx]), jnp.asarray(static_np[val_idx]))
    )
    val_metrics = _metrics(raw_titer[val_idx], std_train.denorm_target(val_pred_std))
    logger.info(
        "[cde:val] best epoch %d  RMSE=%.1f  MAE=%.1f  MAPE=%.1f%%  R2=%.3f",
        best_epoch,
        val_metrics["rmse"],
        val_metrics["mae"],
        val_metrics["mape"] * 100,
        val_metrics["r2"],
    )

    # 2) Refit on all data for the deployed model (skipped during sweeps). Standardise
    # on all data (legitimate — the deployed model trains on everything) and train for
    # the early-stopped epoch count so the deployed fit inherits the same stopping.
    if refit_all:
        std_all = fit_standardizer(seq)
        ys_all, static_all, y_all = build_arrays(seq, std_all)
        logger.info("Refitting on all %d experiments for %d epochs...", n, best_epoch + 1)
        deploy_model, _, _ = fit(
            np.arange(n), k_full, ys_all, static_all, y_all, std_all, n_epochs=best_epoch + 1
        )
        standardizer = std_all
    else:
        deploy_model = val_model
        standardizer = std_train

    bundle = CDEBundle(
        model=deploy_model,
        standardizer=standardizer,
        channel_names=seq.channel_names,
        static_names=seq.static_names,
        config={**config, "val_metrics": val_metrics},
    )
    return bundle, val_metrics, history


def predict(bundle: CDEBundle, data_path: str | Path) -> pd.Series:
    """Predict final titer for every experiment in ``data_path``."""
    seq = dp.build_sequences(data_path, None)
    ys_np, static_np, _ = build_arrays(seq, bundle.standardizer)
    pred_std = np.asarray(_predict_batch(bundle.model, jnp.asarray(ys_np), jnp.asarray(static_np)))
    preds = bundle.standardizer.denorm_target(pred_std)
    index = [e.exp_id for e in seq.experiments]
    return pd.Series(preds, index=index, name=schema.TARGET_COL)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_bundle(bundle: CDEBundle, path: str | Path) -> None:
    """Persist the bundle: a pickled metadata header + equinox model leaves.

    equinox models embed functions and static metadata that don't pickle
    reliably, so we serialise the array leaves with equinox's own tool and store
    everything needed to rebuild the model skeleton in the header.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "standardizer": bundle.standardizer,
        "channel_names": bundle.channel_names,
        "static_names": bundle.static_names,
        "config": bundle.config,
    }
    with open(path, "wb") as f:
        pickle.dump(meta, f)
        eqx.tree_serialise_leaves(f, bundle.model)
    logger.info("Saved CDE bundle to %s", path)


def save_history(history: list[dict], path: str | Path) -> None:
    """Write the per-checkpoint training history to a CSV (for diagnostics/plots)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)
    logger.info("Saved training history to %s", path)


def _skeleton(config: dict) -> NeuralCDE:
    """Rebuild an untrained model with the right shapes for deserialisation."""
    return NeuralCDE(
        config["n_static"],
        config["n_channels"],
        config["n_w"],
        config["hidden_size"],
        config["width"],
        config["depth"],
        key=jax.random.PRNGKey(0),
    )


def load_bundle(path: str | Path) -> CDEBundle:
    with open(path, "rb") as f:
        meta = pickle.load(f)
        model = eqx.tree_deserialise_leaves(f, _skeleton(meta["config"]))
    return CDEBundle(
        model=model,
        standardizer=meta["standardizer"],
        channel_names=meta["channel_names"],
        static_names=meta["static_names"],
        config=meta["config"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neural CDE for titer prediction.")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the neural CDE and save it.")
    p_train.add_argument("--data", required=True)
    p_train.add_argument("--targets", required=True)
    p_train.add_argument("--model", required=True)
    p_train.add_argument("--hidden-size", type=int, default=8)
    p_train.add_argument("--width", type=int, default=32)
    p_train.add_argument("--depth", type=int, default=1)
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=3e-3)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--val-frac", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=SEED)
    p_train.add_argument(
        "--history",
        default=None,
        help="CSV path for the training history (default: next to model).",
    )

    p_pred = sub.add_parser("predict", help="Predict titer from a saved CDE.")
    p_pred.add_argument("--data", required=True)
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--out", required=True)

    return parser


def _run_train(args: argparse.Namespace) -> int:
    bundle, _, history = train(
        args.data,
        args.targets,
        hidden_size=args.hidden_size,
        width=args.width,
        depth=args.depth,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    save_bundle(bundle, args.model)
    history_path = args.history or Path(args.model).with_suffix(".history.csv")
    save_history(history, history_path)
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    bundle = load_bundle(args.model)
    preds = predict(bundle, args.data)
    times = (
        dp.read_inputs(args.data)
        .groupby(schema.EXP_COL, sort=False)[schema.TIME_COL]
        .max()
        .reindex(preds.index)
    )
    out = pd.DataFrame(
        {
            schema.EXP_COL: preds.index,
            schema.TIME_COL: times.to_numpy(),
            schema.TARGET_COL: preds.to_numpy(),
        }
    )
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
