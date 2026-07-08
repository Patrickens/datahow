# Developer commands. Requires `uv` (and Docker for the docker-* targets).
MODEL_PATH ?= artifacts/xgb_baseline.joblib
IMAGE ?= datahow-titer-service

.PHONY: help test lint format check figures run-api batch docker-build docker-run

help:
	@echo "targets: test lint format check figures run-api batch docker-build docker-run"

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: lint
	uv run ruff format --check .
	uv run pytest

figures:
	uv run python -m titer_prediction.plotting

run-api:
	uv run uvicorn titer_prediction.service.app:app --host 0.0.0.0 --port 8000 --reload

batch:
	uv run python -m titer_prediction.service.batch_predict \
		--data data/datahow_interview_test_data.csv \
		--model $(MODEL_PATH) \
		--out artifacts/test_predictions.csv

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 \
		-e MODEL_PATH=/app/artifacts/$(notdir $(MODEL_PATH)) \
		-v "$(CURDIR)/artifacts:/app/artifacts" $(IMAGE)
