"""Typed data-transfer objects for the inference API (Pydantic v2).

Mirrors the provided OpenAPI schema: a request carries ``timestamps`` and a
``values`` map of variable-name -> array, using the ``Z:``/``W:``/``X:`` prefix
convention. Structural validation lives here (bad payloads -> 422); model-schema
completeness is checked in :mod:`titer_prediction.service.predictor`.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator, model_validator

from .. import schema

_PREFIXES = (schema.STATIC_PREFIX, schema.CONTROL_PREFIX, schema.STATE_PREFIX)  # Z:, W:, X:


class PredictRequest(BaseModel):
    """One experiment to predict the final titer for."""

    timestamps: list[float] = Field(..., min_length=1)
    values: dict[str, list[float]]
    experiment_id: str | None = None

    @field_validator("timestamps")
    @classmethod
    def _timestamps_finite_increasing(cls, ts: list[float]) -> list[float]:
        if not all(math.isfinite(t) for t in ts):
            raise ValueError("timestamps must all be finite")
        if any(later <= earlier for earlier, later in zip(ts, ts[1:], strict=False)):
            raise ValueError("timestamps must be strictly increasing")
        return ts

    @model_validator(mode="after")
    def _validate_values(self) -> PredictRequest:
        n = len(self.timestamps)
        seen = {prefix: 0 for prefix in _PREFIXES}
        for name, arr in self.values.items():
            prefix = next((p for p in _PREFIXES if name.startswith(p)), None)
            if prefix is None:
                raise ValueError(f"variable {name!r} must start with one of {_PREFIXES}")
            if not all(math.isfinite(x) for x in arr):
                raise ValueError(f"variable {name!r} contains non-finite values")
            seen[prefix] += 1
            if prefix == schema.STATIC_PREFIX:
                if len(arr) not in (1, n):
                    raise ValueError(
                        f"Z: variable {name!r} must have length 1 or {n}, got {len(arr)}"
                    )
            elif len(arr) != n:
                raise ValueError(
                    f"variable {name!r} must have length {n} (matching timestamps), got {len(arr)}"
                )
        for prefix in _PREFIXES:
            if seen[prefix] == 0:
                raise ValueError(f"values must contain at least one {prefix} variable")
        return self


class PredictResponse(BaseModel):
    """The predicted final titer plus a little provenance."""

    prediction: float
    target: str
    model_type: str
    n_timepoints: int
    experiment_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
