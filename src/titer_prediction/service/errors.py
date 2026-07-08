"""Domain exceptions for the inference service.

These are raised by the loader/predictor layers and mapped to HTTP status codes
by the exception handlers in :mod:`titer_prediction.service.app`. Pydantic
handles request-schema validation (422) on its own; these cover the cases it
cannot (model not loaded, semantic payload problems).
"""

from __future__ import annotations


class ServiceError(Exception):
    """Base class for inference-service errors."""


class ModelLoadError(ServiceError):
    """The model artifact is missing or could not be loaded."""


class ModelNotLoadedError(ServiceError):
    """A prediction was requested but no model is loaded (-> HTTP 503)."""


class PayloadError(ServiceError):
    """The payload is structurally valid but cannot be turned into model input.

    For example, it does not provide all the variables the loaded model expects.
    Mapped to HTTP 422.
    """
