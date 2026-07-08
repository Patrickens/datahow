"""Service configuration (environment-driven)."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Default to the XGBoost baseline: faster (no per-request ODE solve), stronger
# and stabler than the CDE, and a lighter runtime. Point MODEL_PATH at a
# ``.eqx`` bundle to serve the neural CDE instead.
DEFAULT_MODEL_PATH = "artifacts/xgb_baseline.joblib"


@dataclass(frozen=True)
class Settings:
    """Runtime settings, read from the environment."""

    model_path: str = DEFAULT_MODEL_PATH


def get_settings() -> Settings:
    """Build settings from the current environment."""
    return Settings(model_path=os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
