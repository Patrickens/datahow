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
- **Substrate/feed-consumption features.** For glucose, glutamine, ammonia, and
  lactate we add initial/final concentration, concentration AUC, feed AUC where
  a matching feed exists (`W:FeedGlc`, `W:FeedGln`), initial plus total added,
  approximate net consumed, and simple normalisations by duration and VCD AUC.
  Ammonia and lactate are not assumed to be fed.
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

**What initialises the hidden state.** `z₀ = ζ_θ(Z, C0)` — the static design **and
the first observation** `C0 = [t0, W(t0), X(t0)]`. Two reasons: (1) `Z:` and `W:`
overlap heavily — the `W:` trajectories are the feed / pH / temperature recipe
*unrolled over time* — so for `Z:` we keep only the scalars with **no `W:`
counterpart**, `Z:Stir` and `Z:DO` (see `STATIC_INIT_COLS`; the planned duration is
the time channel). (2) A CDE evolves via control *increments* `dC`, which are
invariant to a constant offset, so the **absolute** initial state (initial VCD and
substrate levels — strongly predictive) would never reach the model unless injected
through `C0`. (The XGBoost baseline, by contrast, uses `Z:` and *not* `W:`, so there
`Z:` is the compact stand-in for the whole control design.)

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

A full, narrated walk-through lives in the [`exploration.py`](exploration.py)
marimo notebook (`uv run marimo edit exploration.py`); figures are regenerated
with `uv run python -m titer_prediction.plotting`. Highlights:

**The target** is right-skewed with a long high-titer tail (mean > median) — hence
the `log1p` transform, and why the sparse high-titer runs are the hardest to
predict:

![Titer distribution](figures/titer_distribution.png)

**Measured states**, one line per experiment coloured by final titer. Note the
substrates (glucose, glutamine) *rise*: they are fed faster than consumed, so they
accumulate while feeding is on and only draw down once it stops. Ammonia rises as a
metabolic byproduct (it is not fed):

![State trajectories](figures/input_state_timecourses.png)

![State trajectories](figures/input_state_timecourses.png)

A notable domain signal: in the **longer runs, lactate rises and then falls**.
This is the classic **lactate metabolic shift** — cells switch from net lactate
*production* (glycolytic overflow) to net *consumption*, typically once glucose
starts to become limiting. It is a recognised marker of healthy, productive
fed-batch CHO cultures and tends to coincide with higher final titer, so it is a
genuinely informative feature rather than noise.

**Control inputs** are step-like — the feeds switch on/off — which is exactly why
the CDE **step-interpolates** the `W:` controls (while linearly interpolating the
continuous `X:` states):

![Control trajectories](figures/input_control_timecourses.png)

**Gompertz fits** compress each VCD growth curve into interpretable parameters
(fit R² ≈ 0.99), which carry real signal against titer:

![Gompertz fits](figures/gompertz_fits.png)

**Baseline diagnostics** — out-of-fold predictions and feature importances. The
model poorly predicts the high-titer regime, which is unfortunate because these
are exactly the most interesting experiments from a process-optimization
perspective. The most important features remain biologically meaningful: the
**area under the VCD curve** = integral of viable cells, plus substrate/byproduct
level, AUC, and consumption features:

![Regression CV](figures/regression_cv.png)

![Feature importance](figures/feature_importance.png)

## Results (cross-validated / held-out)

Errors are in titer units; the baseline is benchmarked with repeated 5-fold CV
and the CDE with a 20% validation holdout. Numbers will shift once the real test
targets arrive, but the ordering matches expectations.

| Model | RMSE | MAPE | R² | Protocol |
| ----- | ---- | ---- | -- | -------- |
| Mean predictor | ~730 | ~55% | ~0.00 | repeated 5-fold CV |
| **XGBoost baseline** | **~309** | **~12%** | **~0.80** | repeated 5-fold CV |
| Neural CDE | ~740 | ~19% | ~0.48–0.9 | single 20% holdout (very noisy) |

The XGBoost baseline is the dependable model here, as anticipated for a small
tabular-friendly dataset. The neural CDE's single 20% holdout is **very noisy** —
R² ranges from ~0.48 (default config) to ~0.9 across seeds/configs (`titer-cde
sweep`), so it is best read as competitive with, not clearly beating, the baseline.
A repeated holdout would give a more stable estimate; the CDE's value here is
methodological. Use `titer-cde train` (writes a training-history CSV) and
`titer-sweep` to diagnose training and explore hyperparameters.

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
├── artifacts/              # generated features / trained models — git-ignored
├── src/titer_prediction/
│   ├── schema.py               # Z:/W:/X: prefix conventions + column groups
│   ├── data_preprocessing.py   # raw CSV/frame -> tabular features + ragged sequences
│   ├── features.py             # baseline features: Gompertz + TSFEL + static
│   ├── regression.py           # XGBoost baseline, CV, CLI
│   ├── cde.py                  # neural CDE via diffrax, CLI
│   ├── sweep.py                # neural-CDE hyperparameter sweep (CLI)
│   ├── plotting.py             # shared figure helpers
│   └── service/                # Part 2: FastAPI inference microservice
│       ├── app.py  config.py  dto.py  errors.py
│       ├── model_loader.py  predictor.py  batch_predict.py
└── tests/
    ├── test_data_integrity.py  # data-integrity + preprocessing/model tests
    └── test_service.py         # inference-service tests (mocked model)
```

## Getting started

Prerequisites: [`uv`](https://docs.astral.sh/uv/) and Python 3.11/3.12.

```bash
# Install the environment from the lockfile
uv sync --extra dev

# Place the provided data files (not committed) under data/:
#   data/datahow_interview_train_data.csv
#   data/datahow_interview_train_targets.csv
#   data/datahow_interview_test_data.csv
#   data/datahow_interview_test_targets-TEMPLATE.csv

# Build the feature table from the raw CSVs
uv run titer-preprocess \
    --data data/datahow_interview_train_data.csv \
    --targets data/datahow_interview_train_targets.csv \
    --out artifacts/train_features.parquet

# Train the XGBoost baseline (repeated-CV report + saved model bundle)
uv run titer-regression train \
    --data data/datahow_interview_train_data.csv \
    --targets data/datahow_interview_train_targets.csv \
    --model artifacts/xgb_baseline.joblib

# Train the neural CDE
uv run titer-cde train \
    --data data/datahow_interview_train_data.csv \
    --targets data/datahow_interview_train_targets.csv \
    --model artifacts/cde.eqx

# Predict on new inputs with either model (same CSV output format)
uv run titer-regression predict --data data/datahow_interview_test_data.csv \
    --model artifacts/xgb_baseline.joblib --out artifacts/test_predictions.csv
```

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
(`MODEL_PATH`), `errors.py` + handlers (→ 422 / 503 / 500), `app.py` (endpoints).

**Model selection.** `MODEL_PATH` chooses the artifact: `*.joblib` → XGBoost
baseline (the **default**, `artifacts/xgb_baseline.joblib` — fast, no per-request
ODE solve), `*.eqx` → neural CDE. If the artifact is missing the app still starts;
`/health` reports `model_loaded: false` and `/predict` returns 503.

```bash
# Run locally (default model = XGBoost baseline)
uv run uvicorn titer_prediction.service.app:app --host 0.0.0.0 --port 8000
# ...or: make run-api      (serve the CDE: MODEL_PATH=artifacts/cde.eqx make run-api)

curl localhost:8000/health
# {"status":"ok","model_loaded":true}
```

`POST /predict` (one experiment; `Z:` scalars are single-element arrays, `W:`/`X:`
arrays match `timestamps`):

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

**Batch / template workflow** (the OpenAPI schema is single-experiment; this is a
convenience for the interview's test-template CSV):

```bash
uv run titer-batch-predict \
  --data data/datahow_interview_test_data.csv \
  --model artifacts/xgb_baseline.joblib \
  --out artifacts/test_predictions.csv        # -> RowID, Exp, Time[day], Y:Titer
```

**Docker** (the model is mounted at runtime — it is not baked into the image):

```bash
docker build -t datahow-titer-service .
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/app/artifacts/xgb_baseline.joblib \
  -v "$PWD/artifacts:/app/artifacts" datahow-titer-service
```

**Assumptions & limitations.** A request must provide the full variable set the
model was trained on (extra variables are ignored, missing ones → 422). The image
installs the whole ML stack, so it is large — it could be slimmed by splitting the
notebook/CDE dependencies into extras.

## Development

```bash
uv run pytest            # tests (data integrity + service; service uses a mocked model)
uv run ruff check .      # lint
uv run ruff format .     # format
# or: make test / make lint / make check
```
