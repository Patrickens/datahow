# Titer Prediction — DataHow ML Engineer Challenge

Predict the **final product titer** of a simulated upstream mAb bioprocess from
per-experiment time-series data, and (Part 2) serve the model behind a REST
inference API.

> Status: both parts functional end-to-end. **Part 1** — preprocessing, the
> XGBoost baseline, and the neural CDE all train and predict from the CLI and are
> benchmarked below. **Part 2** — a FastAPI inference microservice (`/health`,
> `/predict`) with typed DTOs, tests, and Docker (see *Part 2 — Inference
> microservice*).

## Problem understanding

Each *experiment* is a simulated fed-batch bioreactor run for mAb production,
recorded as a short **daily multivariate time series**. Columns follow a prefix
convention that encodes each variable's role in the process:

| Prefix | Role | Count | Examples / notes |
| ------ | ---- | ----- | ---------------- |
| `Z:`   | **Design / setpoint scalars** — the process *recipe*, fixed before the run | 13 | Feed start/end days, feed rates, pH & temperature setpoints and their shift days, stirring, DO, planned duration. Constant per experiment (recorded on day 0). |
| `W:`   | **Control-input trajectories** — the controls *actually applied* over time | 4 | temp, pH, FeedGlc, FeedGln. The operator's levers; include deliberate step changes. |
| `X:`   | **Measured state trajectories** — the culture's biological/chemical state | 6 | VCD (viable cell density), Glc, Gln, Amm, Lac, Lysed. |
| `Y:`   | **Target** — final product titer (mAb concentration) | 1 | A single scalar per experiment; **not** present among the inputs. |

Training set: 100 experiments. Test set: 20 experiments.

## The two core challenges

1. **Variable-length inputs, single-scalar target.** Experiments differ in the
   number of time-points and in duration, yet the target is one scalar per run.
   We must map a variable-length *multivariate* trajectory to a fixed
   prediction — either by collapsing each series into a fixed feature vector
   (baseline) or with a sequence model that natively handles irregular length
   (the CDE).
2. **Discontinuous control inputs.** Feeds are switched on and off and
   pH/temperature setpoints are shifted at discrete days, so the driving signals
   are piecewise/step-like rather than smooth. This complicates
   derivative-based and interpolation-based models — a neural CDE, for instance,
   needs a control-path interpolation that respects these jumps.

## Modelling strategy

The modelling problem is deliberately small and practical: predict **one scalar
final titer per experiment** from short, variable-length bioprocess trajectories.
With ~100 training experiments, clean preprocessing, honest validation, and
reproducible choices matter more than chasing a marginal leaderboard gain.

We considered three modelling options:

1. **XGBoost on engineered features.** Fast, strong on small tabular datasets,
   easy to deploy, and interpretable through feature importance. The tradeoff is
   that all time dependence has to be engineered manually.
2. **Neural CDE.** Consumes the full trajectory, handles unequal sampling and
   missingness naturally, and lets us encode discontinuous controls through the
   interpolation choice. The tradeoff is lower interpretability and a harder
   story for biologists.
3. **Mechanistic ODE with event-driven controls.** The most biologically
   interpretable option: explicit states, yields, rates, and process events.
   For this take-home it is too slow to formulate robustly and raises
   identifiability, event-handling, solver, and misspecification risks.

The practical comparison is therefore **XGBoost versus the neural CDE**. The
XGBoost model is the dependable small-data baseline; the CDE demonstrates the
trajectory-native method we would keep developing with more data.

### Feature engineering for XGBoost

Show that the task can be solved **simply and efficiently** with a
general-purpose regressor. XGBoost is a strong default for small tabular data;
sensible alternatives include **Gaussian Process regression** (native
uncertainty, excellent in the small-data regime), random forests, and
regularised linear models. To turn each variable-length trajectory into a fixed
feature vector we combine:

- **Curve-fit features (Gompertz).** Fit a 4-parameter Gompertz growth curve
  with baseline,
  `y(t) = y0 + a·exp(-b·exp(-k_g·(t - t_i)))`,
  to the VCD trajectory and extract its parameters — amplitude `a`, shape `b`,
  inflection time `t_i`, growth rate `k_g`, and baseline `y0`. This compresses a
  whole growth curve into a handful of **interpretable** numbers and handles
  differing lengths gracefully. Its limitation: a single monotone sigmoid
  **cannot capture sequential substrate dynamics** — the ordered depletion of
  glucose then glutamine, feed-driven replenishment, or the lactate
  production→consumption switch. Those coupled dynamics are left to TSFEL and are
  handled more naturally by the CDE. Gompertz is folded in as a **custom TSFEL
  feature**, so it lives in the same extraction pipeline.
- **Automated time-series features (TSFEL).** A curated set of interpretable
  *statistical* and *temporal* features per `X:` state channel (~25 each),
  including the **area under the curve** (e.g. the integral of viable cells),
  slope, RMS, entropy, and turning points. See *Choosing a feature library*
  below for why TSFEL over tsfresh/catch22.
- **Substrate/feed-consumption features.** For glucose and glutamine we add
  initial/final concentration, total feed integral (`W:FeedGlc`, `W:FeedGln`),
  initial plus total fed, and apparent consumed amount. TSFEL already provides
  concentration AUCs for the `X:` channels, so the custom features avoid
  duplicating those.
- **Cell-population accounting features.** From viable cell density and lysed
  fraction we estimate total cell density as
  `X:VCD / (1 - X:Lysed)`, then add initial/final, max, and AUC summaries.
- Plus the pass-through `Z:` design scalars and observed duration / length.

The XGBoost target is `log1p(titer)`. This helps with the right-skewed target and
approximately proportional / heteroskedastic noise, while not strictly
guaranteeing non-negative predictions unless the inverse `expm1` output is
clipped.

**Choosing a feature library.** We considered **tsfresh** (conflicts with the
JAX/diffrax stack via its numba dependency, and emits 200+ features), then tried
**catch22** (worked — baseline R² ≈ 0.82 — but its 22 generic dynamical-systems
features aren't domain-meaningful; notably no AUC). We settled on **TSFEL**:
numba-free (one environment), interpretable, includes the bioprocess-relevant
features we want, and extensible enough to host the Gompertz custom feature.

### Neural Controlled Differential Equation (diffrax)

A sequence model that ingests the raw trajectories directly: the `W:`/`X:`
channels form a driving path, a neural CDE integrates along it, and the terminal
hidden state maps to titer. It handles variable length and irregular sampling
natively (batches are padded by holding the last observation — a flat,
zero-contribution tail — so no masking is needed).

Padding is not extra process time. It is only a rectangular-array trick for
batching. Because the full final row is repeated, including the real-time
channel, the control path is constant on the padded tail: `C(s) = C(S)` and
`dC(s) = 0`. Since `dh = f_theta(h) dC`, a pure CDE receives no update there.

**What initialises the hidden state.** The input path is
`C(s) = [t(s), W(s), X_obs(s)]` in `R^c`, with
`c = 1 + n_W + n_X`. Static variables `Z` are used only for initialisation:
`h_0 = ζ_θ(Z, C_0)`, where `C_0 = [t_0, W(t_0), X_obs(t_0)]`. The learned hidden
state `h(s) in R^d` is not biological; `d` is a capacity hyperparameter tuned in
the CDE sweep. Small `d` may underfit, while large `d` may overfit.

Two details matter. (1) `Z:` and `W:` overlap heavily, so for `Z` we keep only the
scalars with **no `W:` counterpart**, `Z:Stir` and `Z:DO`. (2) A CDE evolves via
control *increments* `dC`, so the absolute initial state would never reach the
model unless injected through `C_0`.

The model equations are:

```text
h_0 = zeta_theta(Z, C_0)
C_0 = [t_0, W(t_0), X_obs(t_0)]
dh(s) = f_theta(h(s)) dC(s),    f_theta: R^d -> R^(d x c)
y_hat = ell_theta(h(S))
```

**Interpolation as an inductive bias (challenge #2).** How we interpolate between
daily samples is a modelling assumption, chosen per channel group (see
`make_mixed_cde_path` in `cde.py`):

- `W:` controls → **step** interpolation — feeds and setpoint switches are
  genuinely discontinuous; linear would fabricate ramps that never happened.
- `X:` observations → **piecewise linear** — sampled from continuous-ish states, so
  a staircase would fabricate jumps (this is not a claim the biology is linear).
- real time → **channel 0**, so the model stays time-aware.

Because a control step has zero duration in real time, we integrate over a
strictly-increasing *path parameter* `s` (real time rides along as a channel), so a
jump becomes a finite segment the solver can see. Full rectilinear interpolation of
everything would be defensible only under an *online-information* reading; for
offline whole-trajectory regression the mixed convention is the more faithful bias.

**Why a CDE (and not a neural ODE)?** Variable-length and irregularly-sampled
trajectories are handled natively; order and timing are preserved; discontinuous
controls are represented without fabricated ramps; and it supports online prediction
as new measurements extend the path. A neural ODE evolves autonomously and cannot
ingest the external feeds/observations, whereas the CDE is *driven by the data path*.

**A mechanistic ODE alternative.** A purely mechanistic ODE (explicit growth /
uptake / byproduct / death / product-formation balances with event handling for the
feeds) would be scientifically interesting but is out of scope here: it needs
committed rate laws, nontrivial parameter identification on ~100 short trajectories,
and careful event handling for the discontinuities. The CDE is a cleaner path-based
model for ragged controlled trajectories — honestly, a sequence model, not a
mechanistic simulator. As expected, it lands below the baseline on ~100 experiments;
its value is methodological.

## Exploratory data analysis

The full, narrated walk-through — with every figure — lives in the
[`exploration.py`](exploration.py) marimo notebook. It is the detailed
explanation of the whole pipeline; open it with:

```bash
uv run marimo run exploration.py     # read-only view (or: marimo edit)
make figures                         # (re)generate figures/*.png  (FORCE=1 to refresh)
```

Figures are derived from the confidential data and are **not committed**; the
commands above regenerate them locally. Narrative highlights:

- **The target** is right-skewed with a long high-titer tail (mean > median) —
  hence the `log1p` transform, and why the sparse high-titer runs are hardest to
  predict.
- **Measured states**: the substrates (glucose, glutamine) *rise* because they
  are fed faster than consumed, accumulating while feeding is on and drawing down
  only once it stops; ammonia rises as a metabolic byproduct (it is not fed).
- **Lactate metabolic shift**: in the longer runs lactate rises and then falls —
  cells switch from net lactate *production* (glycolytic overflow) to net
  *consumption* once glucose becomes limiting. A recognised marker of healthy,
  productive fed-batch CHO cultures that tends to coincide with higher titer, so
  a genuinely informative feature rather than noise.
- **Control inputs** are step-like (feeds switch on/off) — which is exactly why
  the CDE **step-interpolates** the `W:` controls while linearly interpolating
  the continuous `X:` states.
- **Gompertz fits** compress each VCD growth curve into interpretable parameters
  (fit R² ≈ 0.99) that carry real signal against titer.
- **Baseline diagnostics** (out-of-fold predictions + feature importances): the
  model under-predicts the high-titer regime — unfortunate, as those are the most
  interesting runs — but the top features stay biologically meaningful (the AUC
  of viable/total cells, substrate/byproduct levels and AUCs, feed accounting).

Top 15 features by XGBoost gain (from `artifacts/feature_importance.csv`):

| Feature | Importance |
| ------- | ---------- |
| Total cell density AUC | 0.152 |
| Gln apparent consumed | 0.133 |
| VCD centroid | 0.093 |
| Lysed signal distance | 0.072 |
| VCD area under the curve | 0.060 |
| Gln signal distance | 0.047 |
| Glc total fed | 0.042 |
| VCD median | 0.040 |
| Lac signal distance | 0.033 |
| Amm max | 0.022 |
| Glc apparent consumed | 0.021 |
| Lac area under the curve | 0.019 |
| Amm absolute energy | 0.019 |
| Glc initial plus fed | 0.018 |
| Lysed mean diff | 0.017 |

## Results (cross-validated / held-out)

Errors are in titer units; the baseline is benchmarked with repeated 5-fold CV
and the CDE with a 20% validation holdout. Numbers will shift once the real test
targets arrive, but the ordering matches expectations.

| Model | RMSE | MAPE | R² | Protocol |
| ----- | ---- | ---- | -- | -------- |
| Mean predictor | ~730 | ~55% | ~0.00 | repeated 5-fold CV |
| **XGBoost baseline** | **~280** | **~11.5%** | **~0.85** | 10-config sweep + repeated 5-fold CV |
| Neural CDE | ~459 | ~15.7% | ~0.80 | 10-config sweep + 20% validation holdout |

The XGBoost baseline is the dependable deployment choice here: fast, stable, easy
to serve, and — on this data — the stronger model. The CDE's single-holdout score
is both lower and noisier (a 20% holdout over ~100 experiments swings a lot with
the split), which is exactly why the baseline is benchmarked with **repeated**
CV. The CDE's value here is methodological: it demonstrates the path-based
treatment of ragged trajectories and discontinuous controls, and is where we'd
invest with more data. Run `make models FORCE=1` to reproduce both sweeps and
final refits (single `seed=0`).

> **Note on evaluation.** Model **performance is explicitly not the primary
> criterion** for this challenge; clarity of preprocessing, evaluation,
> benchmarking, and architecture is. We therefore emphasise honest
> cross-validation and clear, documented decisions over squeezing out metrics.

## Repository layout

```
datahow/
├── pyproject.toml          # project + dependencies (managed by uv)
├── uv.lock                 # pinned, reproducible environment
├── README.md
├── PROMPTS.md              # log of AI prompts + decisions (transparency)
├── Dockerfile / Makefile   # inference-service image + dev commands
├── exploration.py          # marimo notebook (data -> preprocessing -> models)
├── data/                   # provided CSVs — git-ignored (see below)
├── artifacts/              # best models + metadata + caches committed; sweep scratch git-ignored
├── src/titer_prediction/
│   ├── schema.py               # Z:/W:/X: prefix conventions + column groups
│   ├── data_preprocessing.py   # load/parse CSVs (+ containers) -> ragged sequences
│   ├── features.py             # baseline features: Gompertz + TSFEL + static
│   ├── regression.py           # XGBoost baseline, CV, CLI
│   ├── cde.py                  # neural CDE via diffrax, CLI
│   ├── sweep.py                # neural-CDE hyperparameter sweep (CLI)
│   ├── plotting.py             # shared figure helpers
│   └── service/                # Part 2: FastAPI inference microservice
│       ├── app.py  config.py  dto.py  errors.py
│       ├── model_loader.py  predictor.py  batch_predict.py
└── tests/
    ├── test_data_integrity.py       # data-integrity + preprocessing/model tests
    ├── test_service.py              # inference-service tests (mocked model)
    ├── test_service_integration.py  # real-model load + /predict + batch (skips w/o artifact)
    └── test_docker_smoke.py         # builds + runs the image (skips w/o Docker daemon)
```

## Getting started

Prerequisites: [`uv`](https://docs.astral.sh/uv/) and Python 3.11/3.12.

```bash
uv sync --extra dev          # install the pinned environment

# Place the provided data files (not committed) under data/:
#   datahow_interview_train_data.csv        datahow_interview_train_targets.csv
#   datahow_interview_test_data.csv         datahow_interview_test_targets-TEMPLATE.csv
```

The **best models are committed** under `artifacts/`, so nothing needs training to
run the service or reproduce the walkthrough. A few `make` commands cover everything:

```bash
# 1. Predictions on the test inputs -> artifacts/test_predictions.csv
#    (RowID, Exp, Time[day], Y:Titer). Drop in the real test targets later.
make predict

# 2. Regenerate the figures the notebook/README describe (FORCE=1 to refresh caches)
make figures

# 3. Rebuild the best models from scratch (both sweeps; single seed=0).
#    Normally unnecessary — the models are committed — so this is a no-op unless FORCE=1.
make models FORCE=1
```

Serve the model and call it (works the same against Docker below):

```bash
make run-api            # uvicorn on :8000 (serves artifacts/xgb_best.joblib by default)
make api-health         # GET  /health   -> {"status":"ok","model_loaded":true}
make api-predict        # POST /predict with scripts/sample_payload.json -> a titer

# Or by hand:
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict \
    -H 'Content-Type: application/json' --data @scripts/sample_payload.json
```

Run it in Docker (the model is mounted, not baked into the image):

```bash
make docker-build
make docker-run         # then the same make api-health / make api-predict / curl calls
```

`MODEL_PATH` selects the artifact — `artifacts/xgb_best.joblib` (default) or
`artifacts/cde_best.eqx` to serve the neural CDE instead.

## Data confidentiality

The raw challenge CSVs and the OpenAPI spec are treated as confidential and are
**not** committed (`data/` is git-ignored). Reviewers should drop the provided
files into `data/` as shown above.

## Part 2 — Inference microservice

A small FastAPI service (`src/titer_prediction/service/`) serves the trained
model behind the provided OpenAPI spec (`GET /health`, `POST /predict`).

**Architecture.** Thin routes; all model work goes through `predict_one`, which
turns one `/predict` payload into a one-experiment DataFrame (`payload_to_frame`)
and runs it through the **same `read_inputs` preprocessing as training** — the API
never re-implements model logic. Layers: `dto.py` (typed, validated requests),
`model_loader.py` (loads a bundle once at startup; **model-agnostic**, dispatching
by artifact extension), `predictor.py` (payload → frame → prediction), `config.py`
(`MODEL_PATH`), `errors.py` + handlers (invalid payload → 400, no model → 503,
unexpected → 500, per the OpenAPI spec), `app.py` (endpoints).

**Model selection.** `MODEL_PATH` chooses the artifact: `*.joblib` → the XGBoost
model (the **default**, `artifacts/xgb_best.joblib` — fast, no per-request ODE
solve), `*.eqx` → the neural CDE (`MODEL_PATH=artifacts/cde_best.eqx`). If the
artifact is missing the app still starts; `/health` reports `model_loaded: false`
and `/predict` returns 503.

`POST /predict` takes one experiment — `Z:` scalars as single-element arrays,
`W:`/`X:` arrays matching `timestamps` (full example in `scripts/sample_payload.json`):

```jsonc
{
  "timestamps": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  "values": {
    "Z:FeedStart": [3.0], "Z:FeedEnd": [11.0], "Z:Stir": [194.7], "Z:DO": [76.1],
    "W:temp": [36.3, "…15 values"], "W:FeedGlc": ["…"],
    "X:VCD": ["…15 values"], "X:Glc": ["…"], "X:Lysed": ["…"]
  },
  "experiment_id": "Test Exp 1"          // optional
}
// -> {"prediction": 2138.9, "target": "Y:Titer", "model_type": "xgboost", "n_timepoints": 15}
```

Running it locally or in Docker, plus the `GET /health` / `POST /predict` calls and
the batch/test-prediction workflow, are all covered by the `make` targets in
[Getting started](#getting-started). The committed end-to-end Docker smoke test
(`bash scripts/smoke_docker.sh`, or `uv run pytest -m docker`) builds the image and
asserts the full contract (health, predict, 400 on a bad payload, 503 with no model).

**Assumptions & limitations.** A request must provide the full variable set the
model was trained on (extra variables are ignored, missing ones → 400). The image
installs the whole ML stack, so it is large — it could be slimmed by splitting the
notebook/CDE dependencies into extras.

## Development

```bash
uv run pytest            # tests (data integrity + service; service uses a mocked model)
uv run ruff check .      # lint
uv run ruff format .     # format
# or: make test / make lint / make check
```
