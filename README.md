# Titer Prediction — DataHow ML Engineer Challenge

Predict the **final product titer** of a simulated upstream mAb bioprocess from
per-experiment time-series data, and (Part 2) serve the model behind a REST
inference API.

> Status: **work in progress.** Part 1 is functional end-to-end — preprocessing,
> the XGBoost baseline, and the neural CDE all train and predict from the CLI and
> are benchmarked below. The Part 2 inference server is the next step.

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

## Modelling philosophy

The *ideal* approach for this kind of data is **hybrid modelling**: a
mechanistic core — an extended metabolic model, or a system of ODEs for cell
growth, substrate consumption and product formation — in which a few parameters
(e.g. yield coefficients, specific productivity, growth/death rates) are
**learnable but human-interpretable**, so we can impose informative priors from
process knowledge. Such models are data-efficient, extrapolate sensibly, and
speak the language of process scientists. Fully identifying one is well beyond
the scope of this challenge, and ~100 experiments are too few to constrain it
reliably — but it is the direction we would pursue in a realistic, data-rich
setting, and the neural CDE below sits partway along that mechanistic ↔ black-box
spectrum.

For this challenge we instead demonstrate two pragmatic points on that spectrum:

### 1. Baseline — generic regression on engineered features (XGBoost)

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
- Plus the pass-through `Z:` design scalars and observed duration / length.

**Choosing a feature library.** We considered **tsfresh** (conflicts with the
JAX/diffrax stack via its numba dependency, and emits 200+ features), then tried
**catch22** (worked — baseline R² ≈ 0.82 — but its 22 generic dynamical-systems
features aren't domain-meaningful; notably no AUC). We settled on **TSFEL**:
numba-free (one environment), interpretable, includes the bioprocess-relevant
features we want, and extensible enough to host the Gompertz custom feature.

### 2. Neural Controlled Differential Equation (diffrax)

A sequence model that ingests the raw trajectories directly: the `W:`/`X:`
channels form a driving path, a neural CDE integrates along it, and the terminal
hidden state maps to titer. It handles variable length and irregular sampling
natively (batches are padded by holding the last observation — a flat,
zero-contribution tail — so no masking is needed).

**What initialises the hidden state.** `Z:` and `W:` overlap heavily — the `W:`
trajectories are the feed / pH / temperature recipe *unrolled over time*, so most
of `Z:` is already in the path (and the planned duration is the time channel). We
therefore initialise `z₀` from only the design scalars with **no `W:` counterpart**:
**`Z:Stir` and `Z:DO`** (see `STATIC_INIT_COLS` in `cde.py`), avoiding duplication.
(The XGBoost baseline, by contrast, uses `Z:` and *not* `W:`, so there `Z:` is the
compact stand-in for the whole control design.)

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

**Baseline diagnostics** — out-of-fold predictions (the model under-predicts the
few very high-titer runs) and the feature importances, led by biologically
meaningful TSFEL features (the **area under the VCD curve** = integral of viable
cells, plus AUC of lactate/glucose/ammonia):

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
| Neural CDE | ~710 | ~19% | ~0.52 | 20% holdout, 300 epochs (noisy) |

The XGBoost baseline is the stronger model here, as anticipated for a small
tabular-friendly dataset; the CDE demonstrates the path-based methodology.

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
├── data/                   # provided CSVs — git-ignored (see below)
├── artifacts/              # generated features / trained models — git-ignored
├── src/titer_prediction/
│   ├── schema.py               # Z:/W:/X: prefix conventions + column groups
│   ├── data_preprocessing.py   # raw CSV -> tabular features + ragged sequences
│   ├── features.py             # baseline features: Gompertz + TSFEL + static
│   ├── regression.py           # XGBoost baseline, CV, CLI
│   └── cde.py                  # neural CDE via diffrax, CLI
└── tests/
    └── test_data_integrity.py  # single project-wide test file
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

## Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run pytest            # tests (data integrity)
```
