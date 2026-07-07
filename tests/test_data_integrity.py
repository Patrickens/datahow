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


def test_sequences_are_padded_with_mask(synthetic_long, tmp_path):
    csv = tmp_path / "inputs.csv"
    synthetic_long.to_csv(csv, index=False)
    seq = dp.build_sequence_dataset(csv)
    # Padded to the longest experiment (3 steps).
    assert seq.channels.shape[1] == 3
    assert seq.mask[0].tolist() == [True, True, True]  # Exp A: full
    assert seq.mask[1].tolist() == [True, True, False]  # Exp B: last step padded
    # Padded step holds the last real value and time.
    assert seq.times[1, 2] == seq.times[1, 1]


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
