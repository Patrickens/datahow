# Developer + reproduction commands. Requires `uv` (and Docker for the docker-* targets).
MODEL_PATH ?= artifacts/xgb_best.joblib
IMAGE ?= datahow-titer-service
DOCKER ?= docker.exe
DOCKER_SHELL ?= cmd.exe /C
PORT ?= 9000
ARTIFACTS_DIR ?= $(shell command -v cygpath >/dev/null 2>&1 && cygpath -m "$(CURDIR)/artifacts" || printf "%s/artifacts" "$(CURDIR)")
DATA ?= data/datahow_interview_train_data.csv
TARGETS ?= data/datahow_interview_train_targets.csv
TEST_DATA ?= data/datahow_interview_test_data.csv

.PHONY: help test lint format check models figures predict \
        run-api api-health api-predict docker-check docker-build docker-run

help:
	@echo "Reproduce:   make models [FORCE=1]   make figures [FORCE=1]   make predict"
	@echo "Serve:       make run-api   then   make api-health   make api-predict"
	@echo "Docker:      make docker-build   make docker-run   then the api-* targets"
	@echo "Dev:         make test   make lint   make format   make check"

# --- Modelling ---------------------------------------------------------------
# Rebuild the deployable best models via the sweeps. Each model is trained ONLY
# if its committed artifact is absent (or FORCE=1) — a present model is reused.
models:
	@if [ "$(FORCE)" = "1" ] || [ ! -f artifacts/xgb_best.joblib ]; then \
		echo ">> XGBoost sweep"; \
		uv run titer-sweep --model-kind xgb --data $(DATA) --targets $(TARGETS); \
	else echo "artifacts/xgb_best.joblib present — reusing (FORCE=1 to retrain)"; fi
	@if [ "$(FORCE)" = "1" ] || [ ! -f artifacts/cde_best.eqx ]; then \
		echo ">> neural CDE sweep"; \
		uv run titer-sweep --model-kind cde --data $(DATA) --targets $(TARGETS); \
	else echo "artifacts/cde_best.eqx present — reusing (FORCE=1 to retrain)"; fi

# Regenerate the figures the notebook/README use. FORCE=1 also refreshes caches.
figures:
	uv run python -m titer_prediction.plotting $(if $(filter 1,$(FORCE)),--force,)

# Assignment deliverable: predictions on the test inputs, in the target-template
# shape (RowID, Exp, Time[day], Y:Titer). Drop in the real test targets later.
predict:
	uv run python -m titer_prediction.service.batch_predict \
		--data $(TEST_DATA) --model $(MODEL_PATH) --out artifacts/test_predictions.csv

# --- Service -----------------------------------------------------------------
run-api:
	uv run uvicorn titer_prediction.service.app:app --host 0.0.0.0 --port $(PORT) --reload

# GET /health and POST /predict — work against `run-api` or `docker-run`.
api-health:
	curl -s localhost:$(PORT)/health; echo

api-predict:
	curl -s -X POST localhost:$(PORT)/predict \
		-H 'Content-Type: application/json' \
		--data @scripts/sample_payload.json; echo

# --- Docker ------------------------------------------------------------------
docker-check:
	@if ! $(DOCKER_SHELL) where "$(DOCKER)" >/dev/null 2>&1; then \
		echo "Docker CLI not found. Install/start Docker Desktop and ensure '$(DOCKER)' is on PATH."; \
		exit 127; \
	fi

docker-build: docker-check
	MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" $(DOCKER_SHELL) "$(DOCKER)" build -t $(IMAGE) .

docker-run: docker-check
	MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" $(DOCKER_SHELL) "$(DOCKER)" run --rm -p $(PORT):8000 \
		-e MODEL_PATH=/app/artifacts/$(notdir $(MODEL_PATH)) \
		--mount type=bind,source="$(ARTIFACTS_DIR)",target=/app/artifacts $(IMAGE)

# --- Dev ---------------------------------------------------------------------
test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: lint
	uv run ruff format --check .
	uv run pytest
