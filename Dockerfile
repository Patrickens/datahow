# Inference service image for the titer-prediction API.
#
# The model artifact is NOT baked in (it is git-ignored, derived from confidential
# data). Mount it at runtime and point MODEL_PATH at it, e.g.:
#   docker run --rm -p 8000:8000 \
#     -e MODEL_PATH=/app/artifacts/xgb_best.joblib \
#     -v "$PWD/artifacts:/app/artifacts" datahow-titer-service
#
# Note: the image installs the full project dependencies (the ML stack), so it is
# large. It could be slimmed by splitting the notebook/CDE deps into extras.
FROM python:3.12-slim

# uv for reproducible installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    MODEL_PATH=/app/artifacts/xgb_best.joblib

# Build the package + install locked runtime deps (no dev tools).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "uvicorn", "titer_prediction.service.app:app", \
     "--host", "0.0.0.0", "--port", "8000"]
