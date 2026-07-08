"""Project-wide tests, focused on data integrity and preprocessing invariants.

The raw CSVs are git-ignored, so these tests locate them under ``data/`` and
skip cleanly when they are absent (e.g. in a fresh clone or CI without the
confidential data). A small synthetic fixture covers the preprocessing logic
without depending on the real files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from titer_prediction import data_preprocessing as dp
from titer_prediction import features as feats
from titer_prediction import schema

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_DATA = DATA_DIR / "datahow_interview_train_data.csv"
TRAIN_TARGETS = DATA_DIR / "datahow_interview_train_targets.csv"
TEST_DATA = DATA_DIR / "datahow_interview_test_data.csv"

requires_real_data = pytest.mark.skipif(
    not TRAIN_DATA.exists(), reason="confidential data/ CSVs not present"
)


# ---------------------------------------------------------------------------
# Synthetic fixture: two tiny experiments of unequal length
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_long() -> pd.DataFrame:
    rows = [
        # Exp A: 3 days; Z only on day 0 (to be forward-filled)
        {
            "RowID": 0,
            "Exp": "A",
            "Time[day]": 0,
            "Z:ExpDuration": 3,
            "Z:Stir": 200.0,
            "Z:DO": 40.0,
            "W:temp": 37.0,
            "X:VCD": 1.0,
        },
        {
            "RowID": 1,
            "Exp": "A",
            "Time[day]": 1,
            "Z:ExpDuration": np.nan,
            "W:temp": 37.0,
            "X:VCD": 2.0,
        },
        {
            "RowID": 2,
            "Exp": "A",
            "Time[day]": 2,
            "Z:ExpDuration": np.nan,
            "W:temp": 36.0,
            "X:VCD": 4.0,
        },
        # Exp B: 2 days
        {
            "RowID": 3,
            "Exp": "B",
            "Time[day]": 0,
            "Z:ExpDuration": 2,
            "Z:Stir": 210.0,
            "Z:DO": 45.0,
            "W:temp": 35.0,
            "X:VCD": 1.0,
        },
        {
            "RowID": 4,
            "Exp": "B",
            "Time[day]": 1,
            "Z:ExpDuration": np.nan,
            "W:temp": 35.0,
            "X:VCD": 3.0,
        },
    ]
    return pd.DataFrame(rows)


def test_static_columns_are_forward_filled(tmp_path, synthetic_long):
    csv = tmp_path / "inputs.csv"
    synthetic_long.to_csv(csv, index=False)
    df = dp.read_inputs(csv)
    # Every row of experiment A must carry the day-0 design value.
    assert (df.loc[df["Exp"] == "A", "Z:ExpDuration"] == 3).all()


def test_read_inputs_accepts_dataframe(synthetic_long):
    # The inference service feeds an in-memory frame instead of a CSV path.
    df = dp.read_inputs(synthetic_long)
    assert (df.loc[df["Exp"] == "A", "Z:ExpDuration"] == 3).all()
    # Source frame is not mutated (read_inputs copies).
    assert synthetic_long["Z:ExpDuration"].isna().sum() == 3


def test_features_have_one_row_per_experiment(synthetic_long):
    features = dp.build_features(synthetic_long)
    assert list(features.index) == ["A", "B"]
    assert features.loc["A", "n_timepoints"] == 3
    assert features.loc["B", "n_timepoints"] == 2


def test_channel_aggregates_are_correct(synthetic_long):
    features = dp.build_features(synthetic_long)
    # Exp A VCD trajectory = [1, 2, 4] over t = [0, 1, 2]
    assert features.loc["A", "X:VCD_first"] == 1.0
    assert features.loc["A", "X:VCD_last"] == 4.0
    assert features.loc["A", "X:VCD_max"] == 4.0
    # Trapezoidal AUC of [1,2,4] over [0,1,2] = 1.5 + 3.0 = 4.5
    assert features.loc["A", "X:VCD_auc"] == pytest.approx(4.5)


def test_substrate_consumption_features_add_feed_accounting_only():
    df = pd.DataFrame(
        {
            "Exp": ["A", "A", "A"],
            "Time[day]": [0.0, 1.0, 2.0],
            "W:FeedGlc": [0.0, 2.0, 0.0],
            "W:FeedGln": [0.0, 1.0, 1.0],
            "X:VCD": [1.0, 2.0, 3.0],
            "X:Glc": [10.0, 9.0, 7.0],
            "X:Gln": [3.0, 2.0, 1.0],
            "X:Amm": [0.0, 2.0, 4.0],
            "X:Lac": [1.0, 1.5, 0.5],
        }
    )

    features = feats.substrate_consumption_features(df)
    row = features.loc["A"]

    assert row["bio_X:Glc_total_fed"] == pytest.approx(2.0)
    assert row["bio_X:Glc_initial_plus_fed"] == pytest.approx(12.0)
    assert row["bio_X:Glc_apparent_consumed"] == pytest.approx(5.0)

    assert row["bio_X:Gln_total_fed"] == pytest.approx(1.5)
    assert row["bio_X:Gln_initial_plus_fed"] == pytest.approx(4.5)
    assert row["bio_X:Gln_apparent_consumed"] == pytest.approx(3.5)

    assert "bio_X:Glc_auc" not in features.columns
    assert "bio_X:Glc_apparent_consumed_per_day" not in features.columns
    assert not any(col.startswith("bio_X:Amm") for col in features.columns)
    assert not any(col.startswith("bio_X:Lac") for col in features.columns)


def _write_synthetic(synthetic_long, tmp_path):
    csv = tmp_path / "inputs.csv"
    synthetic_long.to_csv(csv, index=False)
    targets = pd.DataFrame({"Exp": ["A", "B"], "Time[day]": [2, 1], "Y:Titer": [100.0, 50.0]})
    targets_csv = tmp_path / "targets.csv"
    targets.to_csv(targets_csv, index=False)
    return csv, targets_csv


def test_sequences_are_ragged_and_unpadded(synthetic_long, tmp_path):
    csv, targets_csv = _write_synthetic(synthetic_long, tmp_path)

    seq = dp.build_sequences(csv, targets_csv)
    assert len(seq) == 2
    assert seq.channel_names == ["W:temp", "X:VCD"]
    # Ragged: experiments keep their own (unpadded) lengths.
    a, b = seq.experiments
    assert a.exp_id == "A" and a.times.tolist() == [0.0, 1.0, 2.0]
    assert b.exp_id == "B" and b.times.tolist() == [0.0, 1.0]
    assert a.channels.shape == (3, 2)
    assert a.target == 100.0
    assert seq.has_targets


def test_cde_build_arrays_pads_flat_tail(synthetic_long, tmp_path):
    # Imported lazily so the JAX/diffrax stack is only loaded for this test.
    from titer_prediction import cde

    csv, targets_csv = _write_synthetic(synthetic_long, tmp_path)
    seq = dp.build_sequences(csv, targets_csv)
    standardizer = cde.fit_standardizer(seq)
    ys, static, targets = cde.build_arrays(seq, standardizer)

    n, t_max, c = ys.shape
    assert (n, t_max, c) == (2, 3, 3)  # 2 exps, padded to 3 steps, time + 2 channels
    # Exp B (length 2) has its final step held over the padded tail -> flat.
    np.testing.assert_allclose(ys[1, 2], ys[1, 1])
    # Hidden state is initialised only from the static-only design (Z:Stir, Z:DO).
    assert static.shape == (2, 2)
    assert targets is not None and targets.shape == (2,)


def test_make_mixed_cde_path():
    from titer_prediction import cde

    # cols = [real time, W (1), X (1)]; W changes at t=2, and a flat padded tail.
    #   t:   0   1   2   2(pad)
    #   W:   0   0   5   5
    #   X:  10  12  14  14
    ys = np.array(
        [[0.0, 0.0, 10.0], [1.0, 0.0, 12.0], [2.0, 5.0, 14.0], [2.0, 5.0, 14.0]],
        dtype=float,
    )
    s, path = cde.make_mixed_cde_path(ys, n_w=1)
    s, path = np.asarray(s), np.asarray(path)

    # Path parameter is strictly increasing (2T-1 knots).
    assert path.shape == (2 * ys.shape[0] - 1, 3)
    assert np.all(np.diff(s) > 0)

    time, w, x = path[:, 0], path[:, 1], path[:, 2]
    d_time, d_w, d_x = np.diff(time), np.diff(w), np.diff(x)

    # Every segment is either a "flow" (W held) or a "jump" (time & X held).
    is_jump = d_w != 0
    # W changes only where time and X are held fixed (pure control jump).
    assert np.allclose(d_time[is_jump], 0)
    assert np.allclose(d_x[is_jump], 0)
    # X moves only together with real time, never as an artificial jump.
    assert np.all((d_x == 0) | (d_time != 0))
    # The one real W change (0 -> 5) is captured as a single jump segment.
    assert is_jump.sum() == 1 and np.isclose(d_w[is_jump][0], 5.0)

    # The flat padded tail (last two rows identical) contributes zero increments.
    np.testing.assert_allclose(path[-1], path[-2])


def test_cde_prediction_is_padding_invariant():
    # A flat padded tail must not change the prediction: dC = 0 there.
    import jax
    import jax.numpy as jnp

    from titer_prediction import cde

    model = cde.NeuralCDE(
        n_static=2,
        n_channels=3,
        n_w=1,
        hidden_size=4,
        width=8,
        depth=1,
        key=jax.random.PRNGKey(0),
    )
    ys = jnp.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [2.0, 5.0, 3.0], [3.0, 5.0, 3.5]],
        dtype=jnp.float32,
    )
    static = jnp.array([0.3, -0.2], dtype=jnp.float32)

    pred = float(model(ys, static))
    # Append extra copies of the final row -> a longer but still-flat tail.
    ys_padded = jnp.concatenate([ys, jnp.repeat(ys[-1:], 3, axis=0)], axis=0)
    pred_padded = float(model(ys_padded, static))

    assert np.isclose(pred, pred_padded, atol=1e-4)


# ---------------------------------------------------------------------------
# Integrity checks against the real (confidential) data
# ---------------------------------------------------------------------------
@requires_real_data
def test_train_schema_matches_expectations():
    df = dp.read_inputs(TRAIN_DATA)
    assert schema.static_columns(df) == list(schema.EXPECTED_STATIC_COLS)
    assert schema.control_columns(df) == list(schema.EXPECTED_CONTROL_COLS)
    assert schema.state_columns(df) == list(schema.EXPECTED_STATE_COLS)


@requires_real_data
def test_every_train_experiment_has_exactly_one_target():
    ds = dp.build_tabular_dataset(TRAIN_DATA, TRAIN_TARGETS)
    assert ds.has_targets
    assert not ds.targets.isna().any()
    assert ds.features.index.is_unique


@requires_real_data
def test_no_missing_values_in_train_features():
    ds = dp.build_tabular_dataset(TRAIN_DATA, TRAIN_TARGETS)
    assert not ds.features.isna().any().any()


@requires_real_data
def test_static_columns_constant_within_experiment():
    df = dp.read_inputs(TRAIN_DATA)
    static_cols = schema.static_columns(df)
    spread = df.groupby(schema.EXP_COL)[static_cols].nunique(dropna=True)
    assert (spread <= 1).all().all()


@requires_real_data
def test_targets_are_positive():
    ds = dp.build_tabular_dataset(TRAIN_DATA, TRAIN_TARGETS)
    assert (ds.targets > 0).all()
