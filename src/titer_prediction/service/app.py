"""FastAPI application: GET /health and POST /predict.

Routes stay thin — they parse (Pydantic validates), delegate to
:func:`predictor.predict_one`, and let the exception handlers map errors to HTTP
status codes. The model is loaded once at startup into ``app.state`` and supplied
to routes via the ``get_predictor`` dependency (overridable in tests).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .config import get_settings
from .dto import HealthResponse, PredictRequest, PredictResponse
from .errors import ModelLoadError, ModelNotLoadedError, PayloadError, ServiceError
from .model_loader import Predictor, load_predictor
from .predictor import predict_one

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; degrade gracefully if it is missing."""
    settings = get_settings()
    try:
        app.state.predictor = load_predictor(settings.model_path)
    except ModelLoadError as exc:
        # Don't crash the app: /health reports model_loaded=false and /predict 503.
        logger.error("Starting without a model: %s", exc)
        app.state.predictor = None
    yield


app = FastAPI(
    title="Titer Prediction API",
    description="Predict the final titer of a simulated mAb bioprocess experiment.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> Predictor:
    """Dependency: the loaded model, or 503 if none. Overridden in tests."""
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise ModelNotLoadedError("no model is loaded; check MODEL_PATH")
    return predictor


@app.exception_handler(PayloadError)
async def _handle_payload_error(_request: Request, exc: PayloadError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ModelNotLoadedError)
async def _handle_not_loaded(_request: Request, exc: ModelNotLoadedError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ServiceError)
async def _handle_service_error(_request: Request, exc: ServiceError) -> JSONResponse:
    logger.exception("service error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness + whether a model is loaded. Runs no inference."""
    loaded = getattr(request.app.state, "predictor", None) is not None
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest, predictor: Annotated[Predictor, Depends(get_predictor)]
) -> PredictResponse:
    """Predict the final titer for one experiment."""
    try:
        return predict_one(predictor, request)
    except ServiceError:
        raise  # PayloadError / ModelNotLoadedError -> mapped by handlers
    except Exception as exc:  # unexpected -> 500
        raise ServiceError(f"prediction failed: {exc}") from exc
