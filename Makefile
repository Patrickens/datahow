# Developer + reproduction commands. Requires `uv` (and Docker for the docker-* targets).
MODEL_PATH ?= artifacts/xgb_best.joblib
IMAGE ?= datahow-titer-service
DOCKER ?= docker.exe
PYTHON ?= uv run python
PORT ?= 9000
ARTIFACTS_DIR ?= artifacts
DATA ?= data/datahow_interview_train_data.csv
TARGETS ?= data/datahow_interview_train_targets.csv
TEST_DATA ?= data/datahow_interview_test_data.csv

.PHONY: help test lint format check models figures notebook predict \
        run-api api-health api-predict docker-check docker-build docker-run \
        docker-api-health docker-api-predict

help:
	@echo "Reproduce:   make models [FORCE=1]   make figures [FORCE=1]   make notebook   make predict"
	@echo "Serve:       make run-api   then   make api-health   make api-predict"
	@echo "Docker:      make docker-build   make docker-run   then   make docker-api-health   make docker-api-predict"
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

# Export the exploration notebook to a standalone HTML report. `marimo export`
# runs the notebook, which reads the model artifacts and figures — so `models`
# and `figures` are prerequisites. Output is gitignored (artifacts/, *.html).
notebook: models figures
	uv run marimo export html exploration.py -o artifacts/exploration.html -f
	@echo ">> wrote artifacts/exploration.html"

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
	$(PYTHON) scripts/docker_cli.py check --docker "$(DOCKER)"

docker-build: docker-check
	$(PYTHON) scripts/docker_cli.py build --docker "$(DOCKER)" --image "$(IMAGE)"

docker-run: docker-check
	$(PYTHON) scripts/docker_cli.py run --docker "$(DOCKER)" --image "$(IMAGE)" --port "$(PORT)" \
		--model-path "$(MODEL_PATH)" --artifacts-dir "$(ARTIFACTS_DIR)"

docker-api-health: api-health

docker-api-predict: api-predict

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
