"""Load a trained model bundle and expose a uniform prediction interface.

The service is model-agnostic: it dispatches on the artifact extension
(``.joblib`` → XGBoost baseline, ``.eqx`` → neural CDE) and wraps whichever
bundle is loaded in a :class:`Predictor`. Both underlying ``predict`` functions
accept an in-memory DataFrame (via :func:`data_preprocessing.read_inputs`), so
the API never re-implements model logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from .. import schema
from .errors import ModelLoadError

logger = logging.getLogger(__name__)


@runtime_checkable
class Predictor(Protocol):
    """Everything the API needs from a loaded model (mockable in tests)."""

    model_type: str
    target: str
    expected_static: tuple[str, ...]
    expected_control: tuple[str, ...]
    expected_state: tuple[str, ...]

    def predict_frame(self, df: pd.DataFrame) -> pd.Series:
        """Predict final titer for every experiment in ``df`` (indexed by Exp)."""
        ...


@dataclass
class BundlePredictor:
    """A :class:`Predictor` backed by a trained bundle + its ``predict`` fn."""

    model_type: str
    _predict: object  # callable(df) -> pd.Series
    target: str = schema.TARGET_COL
    expected_static: tuple[str, ...] = schema.EXPECTED_STATIC_COLS
    expected_control: tuple[str, ...] = schema.EXPECTED_CONTROL_COLS
    expected_state: tuple[str, ...] = schema.EXPECTED_STATE_COLS
    metadata: dict = field(default_factory=dict)

    def predict_frame(self, df: pd.DataFrame) -> pd.Series:
        return self._predict(df)


def load_predictor(model_path: str | Path) -> BundlePredictor:
    """Load the bundle at ``model_path``; dispatch on the file extension.

    Raises :class:`ModelLoadError` if the artifact is missing, of an unknown
    type, or fails to deserialise.
    """
    path = Path(model_path)
    if not path.exists():
        raise ModelLoadError(f"Model artifact not found: {path}")

    suffix = path.suffix.lower()
    try:
        # Import the model module lazily so serving the tabular baseline does not
        # pull in the JAX/diffrax stack (and vice versa).
        if suffix == ".joblib":
            from .. import regression

            bundle = regression.load_bundle(path)
            model_type = "xgboost"

            def _predict(df: pd.DataFrame) -> pd.Series:
                return regression.predict(bundle, df)
        elif suffix == ".eqx":
            from .. import cde

            bundle = cde.load_bundle(path)
            model_type = "cde"

            def _predict(df: pd.DataFrame) -> pd.Series:
                return cde.predict(bundle, df)
        else:
            raise ModelLoadError(
                f"Unsupported model artifact {suffix!r} (expected .joblib or .eqx)."
            )
    except ModelLoadError:
        raise
    except Exception as exc:  # deserialisation / version mismatch / corrupt file
        raise ModelLoadError(f"Failed to load model from {path}: {exc}") from exc

    logger.info("Loaded %s model from %s", model_type, path)
    return BundlePredictor(model_type=model_type, _predict=_predict)
