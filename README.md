# Titer Prediction — DataHow ML Engineer Challenge

Predict the **final product titer** of a simulated upstream mAb bioprocess from
per-experiment time-series data, and (Part 2) serve the model behind a REST
inference API.

> Status: **work in progress.** Scaffold and data-preprocessing module are in
> place and verified against the provided CSVs. Regression (XGBoost) and neural
> CDE (diffrax) modules, plus the inference server, are the next steps.

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
  to the key trajectories (notably VCD) and extract its parameters — amplitude
  `a`, shape `b`, inflection time `t_i`, growth rate `k_g`, and baseline `y0`.
  This compresses a whole growth curve into a handful of **interpretable**
  numbers and handles differing lengths gracefully.
- **Automated time-series features (catch22).** The canonical *catch22* set —
  22 non-redundant, highly informative features per channel (trend,
  autocorrelation, spectral, distribution, entropy, …). Deliberately compact:
  22 features/channel suits ~100 experiments far better than the thousands a
  library like tsfresh would emit, and it stays numba-free so it shares one
  environment with the JAX/diffrax stack.
- Plus the pass-through `Z:` design scalars and simple aggregates (final value,
  AUC — e.g. the integral of viable cells — and slope).

### 2. Neural Controlled Differential Equation (diffrax)

A sequence model that ingests the raw trajectories directly: it treats the `Z:`
parameters as static context and the `W:`/`X:` channels as a driving path,
integrates a neural CDE to the final time, and maps the terminal hidden state to
titer. It handles variable length and irregular sampling natively and can
represent the discontinuous controls through its control path. Expected to be at
best on par with the baseline given the data size, but representative of the more
expressive, data-hungry approach.

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
│   ├── data_preprocessing.py   # long CSV -> per-experiment features (CLI)
│   ├── regression.py           # XGBoost baseline (CLI)            [next]
│   └── cde.py                  # neural CDE via diffrax (CLI)      [next]
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
