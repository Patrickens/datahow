# Titer Prediction — DataHow ML Engineer Challenge

Predict the final mAb product titer of a simulated upstream bioprocess from its
time-series inputs (**Part 1**), and serve the trained model behind a small REST
API (**Part 2**).

The design rationale, modelling math, architecture, and testing strategy live in
**`exploration.py`** — run `make notebook` for the rendered HTML deep-dive.

## Quickstart

Requires Python 3.11/3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev        # install (dev extras: tests, ruff, marimo)
# then place the four challenge CSVs under data/ (see Data schema)

make models                # train the best XGBoost + CDE artifacts
make run-api               # serve on localhost:9000 (Ctrl-C to stop)
make api-health            # GET  /health
make api-predict           # POST /predict  (scripts/sample_payload.json)

make test                  # run the test suite
make notebook              # render the deep-dive to artifacts/exploration.html
```

`data/`, `artifacts/`, and `figures/` are git-ignored — they hold confidential
inputs or reproducible outputs.

## Results at a glance

Two models, both fit in `log1p(titer)` space and scored on raw titer:

- **XGBoost** on engineered per-experiment features — the deployed default;
  robust 5×5 repeated-CV **R² ≈ 0.84**.
- **Neural CDE** (diffrax), a trajectory-native sequence model — **R² ≈ 0.84** on
  a 3-seed holdout after an optimisation overhaul (adaptive solver, minibatching,
  cosine schedule, leakage-free standardisation).

XGBoost is the default: faster, simpler to serve, and its repeated-CV estimate is
more robust on ~100 experiments. Performance is not the point of the challenge —
clarity of the pipeline and decisions is; see `exploration.py` for the full story.

## Data schema

Each experiment is a short multivariate time series with one scalar target.

| Prefix | Meaning | Examples |
| --- | --- | --- |
| `Z:` | static design/setpoint scalars | feed schedule, pH/temp setpoints, stir, DO |
| `W:` | control-input trajectories | temp, pH, glucose feed, glutamine feed |
| `X:` | measured state trajectories | VCD, glucose, glutamine, ammonia, lactate, lysed |
| `Y:` | target | final titer |

Place the provided files under `data/`:

```text
data/datahow_interview_train_data.csv
data/datahow_interview_train_targets.csv
data/datahow_interview_test_data.csv
data/datahow_interview_test_targets-TEMPLATE.csv
```

The two modelling challenges: experiments have **variable length** but a single
scalar target, and several `W:` controls are **discontinuous step inputs**.

## Repository layout

```text
src/titer_prediction/
  schema.py              column names and Z:/W:/X:/Y: conventions
  data_preprocessing.py  CSV loading, validation, tabular + sequence containers
  features.py            XGBoost feature engineering
  regression.py          XGBoost training, CV, prediction, artifact IO
  cde.py                 neural CDE training, prediction, artifact IO
  sweep.py               reproducible hyperparameter sweeps
  plotting.py            notebook/README figure helpers
  service/
    app.py               FastAPI routes + exception handlers
    dto.py               Pydantic request/response models (DTOs)
    model_loader.py      loads .joblib or .eqx behind one Predictor interface
    predictor.py         request -> training-shaped DataFrame -> model
    batch_predict.py     CSV batch prediction via the same conversion path
    config.py, errors.py runtime settings + service exceptions
tests/                   data-integrity, service (mock + real), sweep, docker smoke
exploration.py           marimo deep-dive: modelling, architecture, testing, ops
```

## Architecture (brief)

The service is thin and model-agnostic — HTTP concerns stay out of model logic:

```text
JSON request -> dto.PredictRequest        (Pydantic validation)
             -> predictor.payload_to_frame()   (same preprocessing as training)
             -> loaded model bundle       (.joblib -> XGBoost, .eqx -> CDE)
             -> dto.PredictResponse
```

The model loads once at startup from `MODEL_PATH` (default
`artifacts/xgb_best.joblib`). If it is missing the app still starts, `/health`
reports `model_loaded=false`, and `/predict` returns 503; invalid payloads return
400. Swap models by pointing `MODEL_PATH` at the `.eqx` artifact. Full rationale:
`exploration.py` §Service architecture.

## Testing & tooling

```bash
make test                 # uv run pytest
uv run pytest -m docker   # opt-in Docker end-to-end smoke test
make check                # ruff lint + format-check + tests
```

Fast data/feature-invariant and API tests (mocked model) run everywhere; the
real-artifact and Docker tests skip cleanly when the model or Docker is absent.
Why each layer exists: `exploration.py` §Testing strategy.

## Docker

The image runs the same FastAPI service with the model **mounted at runtime**, not
baked in:

```bash
make docker-build
make docker-run           # foreground server on localhost:9000
make docker-api-health    # from another terminal
make docker-api-predict
```
