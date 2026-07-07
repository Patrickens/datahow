"""Column schema and prefix conventions for the bioprocess dataset.

The raw CSVs use a prefix convention that we rely on throughout the project:

* ``Z:`` — scalar design / setpoint parameters (constant per experiment,
  recorded only on day 0).
* ``W:`` — control-input trajectories actually applied over time.
* ``X:`` — measured state trajectories.
* ``Y:`` — the target (final titer).

Column groups are derived from these prefixes at runtime so the code adapts if
the exact set of measured variables changes, while the expected names below let
tests and validators assert the dataset matches what we designed against.
"""

from __future__ import annotations

import pandas as pd

# --- Identifier / bookkeeping columns ---------------------------------------
ROWID_COL = "RowID"
EXP_COL = "Exp"
TIME_COL = "Time[day]"

# --- Target ------------------------------------------------------------------
TARGET_COL = "Y:Titer"

# --- Prefixes ----------------------------------------------------------------
STATIC_PREFIX = "Z:"  # scalar design parameters
CONTROL_PREFIX = "W:"  # control-input trajectories
STATE_PREFIX = "X:"  # measured state trajectories

# --- Expected column names (for validation / documentation) ------------------
EXPECTED_STATIC_COLS: tuple[str, ...] = (
    "Z:FeedStart",
    "Z:FeedEnd",
    "Z:FeedRateGlc",
    "Z:FeedRateGln",
    "Z:phStart",
    "Z:phEnd",
    "Z:phShift",
    "Z:tempStart",
    "Z:tempEnd",
    "Z:tempShift",
    "Z:Stir",
    "Z:DO",
    "Z:ExpDuration",
)
EXPECTED_CONTROL_COLS: tuple[str, ...] = (
    "W:temp",
    "W:pH",
    "W:FeedGlc",
    "W:FeedGln",
)
EXPECTED_STATE_COLS: tuple[str, ...] = (
    "X:VCD",
    "X:Glc",
    "X:Gln",
    "X:Amm",
    "X:Lac",
    "X:Lysed",
)


def columns_with_prefix(df: pd.DataFrame, prefix: str) -> list[str]:
    """Return dataframe columns starting with ``prefix``, preserving order."""
    return [c for c in df.columns if c.startswith(prefix)]


def static_columns(df: pd.DataFrame) -> list[str]:
    """Scalar design (``Z:``) columns present in ``df``."""
    return columns_with_prefix(df, STATIC_PREFIX)


def control_columns(df: pd.DataFrame) -> list[str]:
    """Control-input (``W:``) columns present in ``df``."""
    return columns_with_prefix(df, CONTROL_PREFIX)


def state_columns(df: pd.DataFrame) -> list[str]:
    """Measured-state (``X:``) columns present in ``df``."""
    return columns_with_prefix(df, STATE_PREFIX)
