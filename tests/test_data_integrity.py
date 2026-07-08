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
        {"RowID": 0, "Exp": "A", "Time[day]": 0, "Z:ExpDuration": 3, "W:temp": 37.0, "X:VCD": 1.0},
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
        {"RowID": 3, "Exp": "B", "Time[day]": 0, "Z:ExpDuration": 2, "W:temp": 35.0, "X:VCD": 1.0},
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
    assert static.shape == (2, 1)
    assert targets is not None and targets.shape == (2,)


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
