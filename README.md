# Titer Prediction - DataHow ML Engineer Challenge

Predict final mAb product titer from simulated upstream bioprocess time series,
then serve the trained model behind a small REST API.

The repository contains:

- preprocessing for the `Z:`/`W:`/`X:` long-format process data;
- an XGBoost baseline on engineered per-experiment features;
- a neural controlled differential equation (CDE) sequence model;
- a FastAPI inference service with `/health` and `/predict`;
- tests for preprocessing, model-serving behavior, and Docker smoke checks.

Generated data, figures, and model artifacts are intentionally not committed.
They are reproducible from the provided challenge files.

## Data Schema

Each experiment is a short multivariate time series.

| Prefix | Meaning | Examples |
| --- | --- | --- |
| `Z:` | static design/setpoint scalars | feed schedule, pH/temp setpoints, stir, DO |
| `W:` | control-input trajectories | temp, pH, glucose feed, glutamine feed |
| `X:` | measured state trajectories | VCD, glucose, glutamine, ammonia, lactate, lysed |
| `Y:` | target | final titer |

Two modeling issues matter most: experiments have variable length but one scalar
target, and several controls are discontinuous step inputs.

## Modeling Approach

The practical deployed model is XGBoost. Each experiment is converted into a
fixed feature vector containing static design variables, TSFEL trajectory
features, Gompertz growth-curve features for VCD, substrate/feed accounting, and
cell-density summaries. The target is modeled in `log1p(titer)` space.

The CDE is included as a trajectory-native alternative. It preserves time order,
handles variable-length sequences, step-interpolates `W:` controls, linearly
interpolates `X:` states, and pads batches with a flat final tail so padding
contributes nothing to the CDE integral.

The XGBoost model is the better production default for this small dataset:
faster, easier to inspect, simpler to serve, and stronger in validation. The CDE
is useful as the more structurally faithful sequence-model direction.

## Repository Layout

```text
src/titer_prediction/
  schema.py              column names and Z:/W:/X:/Y: conventions
  data_preprocessing.py  CSV loading, validation, tabular and sequence containers
  features.py            XGBoost feature engineering
  regression.py          XGBoost training, CV, prediction, artifact IO
  cde.py                 neural CDE training, prediction, artifact IO
  sweep.py               reproducible hyperparameter sweeps
  plotting.py            notebook/README figure helpers
  service/
    app.py               FastAPI routes and error handlers
    dto.py               Pydantic request/response models
    model_loader.py      loads .joblib or .eqx artifacts behind one interface
    predictor.py         /predict payload -> training-shaped DataFrame -> model
    batch_predict.py     CSV batch prediction via the service conversion path
    config.py, errors.py runtime settings and service exceptions

tests/
  test_data_integrity.py       preprocessing and feature invariants
  test_service.py              API tests with a mocked model
  test_service_integration.py  real artifact tests, skipped when absent
  test_sweep.py                deterministic sweep config tests
  test_docker_smoke.py         optional Docker end-to-end smoke test
```

## Setup

Requires Python 3.11/3.12 and `uv`.

```bash
uv sync --extra dev
```

Place the provided challenge files under `data/`:

```text
data/datahow_interview_train_data.csv
data/datahow_interview_train_targets.csv
data/datahow_interview_test_data.csv
data/datahow_interview_test_targets-TEMPLATE.csv
```

`data/`, `artifacts/`, and `figures/` are git-ignored because they contain
confidential inputs or generated outputs.

## Reproduce

```bash
make models            # trains missing best model artifacts
make models FORCE=1    # retrains both model families from scratch
make predict           # writes artifacts/test_predictions.csv
make figures           # regenerates figures/*.png
```

Default artifact paths:

```text
artifacts/xgb_best.joblib
artifacts/cde_best.eqx
```

## Serve Locally

The service defaults to `MODEL_PATH=artifacts/xgb_best.joblib`.

```bash
make run-api       # foreground uvicorn server on localhost:9000
make api-health    # GET /health
make api-predict   # POST /predict using scripts/sample_payload.json
```

Manual calls:

```bash
curl -s localhost:9000/health
curl -s -X POST localhost:9000/predict \
  -H 'Content-Type: application/json' \
  --data @scripts/sample_payload.json
```

Use the CDE instead:

```bash
MODEL_PATH=artifacts/cde_best.eqx make run-api
```

If the model artifact is missing, the app still starts: `/health` reports
`model_loaded=false` and `/predict` returns 503.

## Docker

The Docker image runs the same FastAPI service. The model is mounted at runtime,
not baked into the image.

```bash
make docker-build
make docker-run          # foreground server on localhost:9000
make docker-api-health   # from another terminal
make docker-api-predict  # from another terminal
```

## Service Design

The service is intentionally thin:

```text
JSON request
  -> dto.PredictRequest validation
  -> predictor.payload_to_frame()
  -> same preprocessing path used in training
  -> loaded model bundle
  -> dto.PredictResponse
```

`model_loader.py` dispatches by artifact suffix:

- `.joblib` -> XGBoost bundle
- `.eqx` -> CDE bundle

This keeps HTTP concerns separate from model logic and lets tests replace the
real model with a small mock predictor.

## Development

```bash
uv run pytest
uv run pytest -m docker
uv run ruff check .
uv run ruff format .
```

The Docker test builds and runs the image, and skips automatically when Docker
or the model artifact is unavailable.
