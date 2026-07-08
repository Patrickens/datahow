# Prompt Log

This document records the prompts used to drive the AI-assisted parts of this
project, in refined / flowing form. The DataHow challenge explicitly invites the
use of AI tools while asking that the author understand and defend every
decision. This log makes the collaboration transparent: each entry captures the
*intent* behind a prompt, not a verbatim transcript, and is paired with the
decisions that resulted.

---

## 1. Project framing and first scaffold

**Goal.** Set up a clean, reproducible project for the DataHow titer-prediction
challenge and lay out the modelling strategy.

**Prompt (refined).**
> Read the challenge brief and the provided data. The task is a regression
> problem: predict the single final titer of a simulated mAb bioprocess from
> per-experiment time-series inputs. I want two models:
> 1. A **baseline** built from light feature engineering (e.g. time since feed
>    stopped, trajectory aggregates) feeding a gradient-boosted model such as
>    XGBoost. Rationale: fast to build, strong on tabular data, and not
>    data-hungry — appropriate given only ~100 experiments.
> 2. A more sophisticated **neural controlled differential equation** (using
>    `diffrax`). We expect its accuracy to be at best on par with the baseline
>    given the small dataset, but it demonstrates how we would approach the
>    problem in a realistic, data-rich setting.
>
> Scaffold the repository with `uv` as the package manager, keeping the
> environment file current for easy installation. Use a `src/` package layout
> with three modules — `data_preprocessing`, `regression` (XGBoost), and `cde`
> (diffrax) — each exposing a CLI `main()` so it can run on a server. Add a
> single project-wide test file focused on data integrity. Initialise Git with a
> `.gitignore` that excludes the (confidential) data. Flag any software or ML
> design improvements and check in before major architecture decisions.

**Key decisions taken (see README for full rationale).**
- `src/titer_prediction/` package layout over flat scripts — cleaner imports and
  a smoother path to the Part 2 inference server.
- Raw data is **git-ignored**; treated as confidential challenge material.
- Build in stages with check-ins: scaffold + `data_preprocessing` first, verified
  against the real CSVs, before implementing `regression` and `cde`.
- Feature strategy for the baseline: pass-through of the 13 `Z:` design scalars,
  plus per-channel aggregates (first/last/min/max/mean/std/AUC/slope) of the 4
  `W:` control and 6 `X:` state trajectories, with the integral-of-viable-cells
  (AUC of `X:VCD`) called out as the classically strongest titer predictor.

---

## 2. README framing and baseline feature strategy

**Goal.** Lead the README with a clear statement of the problem and its
challenges, and pin down the feature-engineering approach for the baseline.

**Prompt (refined).**
> Open the README by explaining our understanding of the problem. Define what
> the `Z:`, `W:`, `X:` and `Y:` variables mean, then frame the task's two core
> challenges: (1) regressing a single final titer from variable-length inputs
> (differing numbers of time-points), and (2) discontinuous control inputs
> (e.g. the feed being switched off). State that the *ideal* solution is a
> hybrid model — an extended metabolic model, or an ODE with interpretable
> learnable parameters such as yields, for which we hold good priors — while
> acknowledging this is beyond the scope and data budget of this task.
>
> Then explain our pragmatic plan: solve the task simply and efficiently with a
> generic regressor like XGBoost (noting alternatives such as Gaussian Process
> regression). To handle the differing number of time-points, engineer features
> with pipelines like **tsfresh** and by **fitting Gompertz growth curves and
> extracting their parameters**. Use this 4-parameter-with-baseline Gompertz
> form for the curve-fit features:
>
> ```python
> def gompertz(t, a, b, t_i, k_g, y0):
>     return y0 + a * np.exp(-b * np.exp(-k_g * (t - t_i)))
> ```

**Key decisions taken.**
- README now opens with *Problem understanding*, *The two core challenges*, and
  *Modelling philosophy* sections before any code/usage detail.
- Baseline feature engineering = Gompertz curve-fit parameters (interpretable
  growth-curve summary) **+** automated time-series features **+** pass-through
  `Z:` scalars and simple aggregates, feeding XGBoost. (The automated-feature
  library was initially planned as tsfresh but changed to catch22 — see entry 3.)
- Alternatives explicitly acknowledged (Gaussian Processes et al.); hybrid /
  mechanistic modelling named as the ideal-but-out-of-scope direction.
- Gompertz reference implementation adapted from the author's prior work
  (`diffbio/experiments/process_Lrham.py`); to be cleaned up and made robust for
  the `regression` module.

## 3. Feature-library pivot: tsfresh -> catch22 (single environment)

**Goal.** Keep the whole project installable in one environment.

**Prompt (refined).**
> Are there alternatives to tsfresh that don't require a separate environment?

**Context / decision.**
- tsfresh hard-depends on `stumpy -> numba`, and numba cannot tolerate the new
  numpy that JAX/diffrax pull in. Isolating tsfresh in its own extra worked but
  meant two environments — against the "easy installation" goal.
- Chose **catch22 (`pycatch22`)** instead: a canonical 22-feature time-series
  extractor, numba-free and numpy-2.x compatible, so it coexists with the JAX
  stack in a single resolution. Its compact feature count is also a better fit
  for ~100 samples than tsfresh's thousands.
- Reverted the conflicting-extras split; back to one unified dependency set.
- Baseline features are now: **Gompertz(VCD) params + catch22 per channel + `Z:`
  scalars + simple aggregates**, feeding XGBoost.

## 4. Neural CDE + preprocessing refactor

**Goal.** Build the neural CDE, and remove hand-rolled logic that diffrax
already provides.

**Prompt (refined).**
> Build cde.py now with a rectilinear control path — rectilinear for the feed
> switches specifically, since linear interpolation makes no physical sense
> across on/off events. Also review data_preprocessing: use diffrax/JAX
> functionality where possible; it feels like a lot was double-implemented there.

**Key decisions taken.**
- Interpolation: **rectilinear (staircase)** to respect discontinuous feeds.
  Because staircase jumps have zero duration in real time, integrate over a
  strictly-increasing **path parameter** and carry real time as a path channel
  (Morrill et al. 2021), so `ControlTerm` (not `.to_ode()`) + `StepTo` capture
  every jump. Verified empirically that `.to_ode()` silently drops the jumps.
- Refactored `data_preprocessing`: removed the hand-rolled padded `SequenceDataset`
  (padding, masks, hold-last-value) — which also wrongly repeated timestamps and
  would break `LinearInterpolation`. Replaced with a lean **ragged**
  `build_sequences` / `ExperimentSequence`; padding/standardisation/interpolation
  now live in `cde.py` and lean on diffrax.
- Static `Z:` scalars initialise the CDE hidden state; batches padded by holding
  the last observation (flat, zero-contribution tail) — no mask needed.
- Persistence via `eqx.tree_serialise_leaves` + a pickled metadata header
  (joblib/pickle of equinox modules is unreliable).

## 5. marimo exploration notebook + README figures

**Goal.** A visual walk-through of data, preprocessing, and the baseline, with
figures saved for the README.

**Prompt (refined).**
> Make a marimo notebook that (1) plots the input time courses (VCD, substrates,
> products, controls) as logically grouped overlay subplots — combine components
> where it makes sense rather than one plot each — and saves them for the README;
> (2) walks through preprocessing (Gompertz fits, feature extraction); and (3)
> the regression, with plots useful for the README.

**Key decisions taken.**
- (recorded as implemented) Plotting helpers factored into an importable module
  so both the marimo notebook and a headless figure-generation step share them
  (DRY); figures saved to a committed `assets/` folder for the README.

## 6. Domain observation — the lactate metabolic shift

**Prompt (refined).**
> In the longer experiments the lactate goes down, which suggests it is being
> consumed — perhaps once glucose starts to deplete. Add a comment on this in the
> README, the prompt log, and the marimo notebook.

**Note.** This is the classic **lactate metabolic shift** in mammalian (CHO)
fed-batch culture: cells switch from net lactate *production* (glycolytic
overflow) to net *consumption*, usually as glucose becomes limiting. It is a
recognised marker of healthy, high-producing cultures and tends to coincide with
higher titer — a genuinely informative signal. Captured as a domain-insight
comment in the state-trajectory discussion (README + notebook).

## 7. Feature-library switch: catch22 -> TSFEL

**Goal.** Use features that are domain-meaningful for a bioprocess (e.g. AUC).

**Prompt (refined).**
> catch22's features don't make much sense for this project — I'd want AUC, which
> it doesn't have. Use TSFEL instead
> (https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html).
> Comment that we considered tsfresh but it failed on dependency conflicts and had
> 200+ features; that we used catch22 but its features didn't fit (no AUC); and
> that we went to TSFEL as a compromise. Since TSFEL is extensible, add the
> Gompertz parameters as a personalised feature. Then strip catch22 from the env
> and note in the notebook that catch22 reached R² = 0.82 with some features
> ranking highly.

**Key decisions taken.**
- Replaced catch22 (`pycatch22`, removed from the env) with **TSFEL**: a curated
  subset of its statistical + temporal domains (~25 features/`X:` channel),
  including **Area under the curve** (the integral of viable cells we wanted).
- Implemented Gompertz parameters as **custom TSFEL features** (`@set_domain`),
  applied to VCD, so they live in the same extraction pipeline.
- Baseline is now ~172 features; CV **R² ≈ 0.80** (catch22 was ≈ 0.82 — the switch
  buys interpretability, not accuracy). With TSFEL, the top-ranked features are
  the AUC/level of VCD and the AUC of lactate/glucose/ammonia — biologically
  sensible, unlike catch22's abstract descriptors.

## 8. Neural CDE — mixed interpolation overhaul

**Goal.** Give each channel group the right interpolation inductive bias while
keeping a plain supervised CDE; explain it clearly; deliver in small commits.

**Prompt (refined).**
> Keep the model a simple supervised neural CDE ($\mathrm{d}z = f_\theta(z)\,
> \mathrm{d}C(s)$, `Z:` initialises the hidden state, `C` carries real time + `W:`
> + `X:`). No mechanistic, neural, or hybrid ODEs; no per-experiment or
> time-varying kinetic parameters. Fix the interpolation: `W:` controls step,
> `X:` observations piecewise linear, real time as channel 0, flat padded tails.
> Build the mixed path with a helper `make_mixed_cde_path(ys, n_w)` (flow segment:
> time & `X:` move, `W:` held; then jump segment: `W:` moves, time & `X:` held),
> integrated over a strictly-increasing path parameter `s`, single `ControlTerm`.
> Add tests. Update the notebook (interpolation as an inductive bias; toy plots of
> linear-vs-step, `t(s)`/`W(s)`, path geometry, toy hidden state; a why-CDE
> section; a mechanistic-ODE-alternative discussion). Small, meaningful commits.

**Key decisions taken.**
- Replaced whole-path `rectilinear_interpolation` with `make_mixed_cde_path`
  (`W:` step, `X:` linear, time channel 0); `n_w` derived from `channel_names` and
  threaded through the model/config/skeleton. Single `ControlTerm`, no `MultiTerm`.
- Framed interpolation explicitly as an **inductive bias**; noted full rectilinear
  is defensible only under an online-information reading.
- Added toy path-parameter plots, a "why a CDE (not a neural ODE)?" section, and a
  "mechanistic ODE alternative" note (out of scope: rate-law commitment, parameter
  identification on ~100 runs, event handling, complexity).
- Planned in plan-mode; delivered as six small commits (`cab4b32` … `1d93d7a`).

## 9. CDE hidden-state init from static-only design (`Z:Stir`, `Z:DO`)

**Goal.** Remove the `Z:`/`W:` redundancy in the CDE initialisation.

**Prompt (refined).**
> `Z:` and `W:` overlap: the `W:` trajectories are the feed / pH / temperature
> recipe unrolled over time, and the planned duration is already the path's time
> channel. The only `Z:` scalars with no `W:` counterpart are `Z:Stir` and `Z:DO`
> — derive the CDE initial state from those instead of all 13 `Z:` scalars.

**Key decisions taken.**
- Initialise `z0` from `STATIC_INIT_COLS = ("Z:Stir", "Z:DO")` only; the feed / pH /
  temperature design already enters via the `W:` path channels (and duration via the
  time channel). The XGBoost baseline still uses all `Z:` (it does not use `W:`), so
  there `Z:` is the compact stand-in for the control design.
- Holdout dipped to ≈0.52 (single noisy 20% split) — expected when dropping
  redundant-but-mildly-informative inputs; kept for cleaner, non-duplicative modelling.

## 10. CDE workflow updates (init from C0, metrics, tuning)

**Goal.** Fix the "weird" initialisation and add training diagnostics + tuning.
Driven by `claude_cde_prompt.txt`; small commit per step.

**Prompt (refined).**
> 1. Initialise the CDE from `Z` **and the initial dynamic condition**:
>    `z0 = ζ_θ(Z, C0)` with `C0 = [t0, W(t0), X(t0)]` (`initial_input =
>    concatenate([static, ys[0]])`; widen the init MLP). 2. Make padding
>    mathematically flat, repeating the entire last row *including real time*, and
>    add a padding-invariance smoke test. 3. Track train/val MSE and val
>    RMSE/MAE/R² over epochs, save the history, and add a marimo training-curves
>    plot. 4. Add a bounded hyperparameter sweep (≤30 sampled configs over epochs
>    / lr / hidden / width / depth / seed) writing `artifacts/cde_sweep_results.csv`.
>    5. Briefly explain init + flat padding in the notebook.

**Key decisions taken.**
- **Init `z0 = ζ_θ(static, C0)`**: a CDE only sees increments `dC`, so the absolute
  initial state (initial VCD / substrate levels) was invisible; injecting `C0`
  fixes it. `static` stays `Z:Stir`,`Z:DO` (no `Z:`/`W:` redundancy). Init MLP
  in_size = `n_static + n_channels`.
- Padding was already flat incl. time (`ys[i, length:] = m[-1]`); clarified the
  comment and added a **padding-invariance** test (extra flat rows → same output).
- `fit()` now returns `(model, history)` (train/val MSE + val RMSE/MAE/R² at ~30
  checkpoints); `train` CLI writes a history CSV; `plot_cde_training_curves()` +
  a notebook cell visualise it.
- Added a `titer-sweep` CLI in its own `sweep.py` module (+ `train(refit_all=False)`
  in `cde.py`); ≤30 sampled configs →
  `artifacts/cde_sweep_results.csv`; smoke-tested, user-run for the full sweep.
- Honesty note: the single 20% holdout is **very noisy** — R² ranged ~0.48–0.9
  across seeds/configs — so the CDE is framed as competitive-not-clearly-better,
  with the sweep/repeated-holdout as the stable read.
- Delivered as five small commits (`8b81388` … `45461e5`).

## 11. Part 2 — inference microservice

**Goal.** A clean, typed, tested, Dockerized FastAPI service around the trained
model implementing the OpenAPI spec (`GET /health`, `POST /predict`), reusing the
training preprocessing rather than re-implementing model logic.

**Prompt (refined).** From `claude_part2_inference_service_prompt.txt`: build a
`titer_prediction/service/` package (FastAPI + Pydantic) with config + one-time
model loading, DTOs matching the YAML (`timestamps` + `values` with `Z:`/`W:`/`X:`
prefix rules), payload→model-input conversion, `/health` and `/predict`, a batch
CSV→template utility, tests (mocked model), Docker, tooling, and README — small
commits per step.

**Key decisions taken.**
- **One seam:** `read_inputs` now accepts a DataFrame, so both `cde.predict` and
  `regression.predict` serve an in-memory payload frame — the API never
  duplicates model logic.
- **Model-agnostic** service: `load_predictor` dispatches by artifact extension
  (`.joblib` → XGBoost, `.eqx` → CDE), wrapped in a `Predictor` protocol
  (mockable). **Default = XGBoost baseline** (user choice: report both, baseline
  as the main one — faster, stronger, lighter). Model modules imported lazily.
- **Validation:** Pydantic DTOs enforce the timestamp/`Z:`/`W:`/`X:` rules (→ 422);
  the predictor checks the payload matches the model's trained schema. Handlers map
  `PayloadError`→422, `ModelNotLoadedError`→503, other `ServiceError`→500. App
  starts even without a model (`/health` reports it).
- Thin routes + DI (`get_predictor`, overridden in tests). `titer-batch-predict`
  writes the template CSV. Dockerfile mounts the model (not baked, confidential);
  Makefile for dev commands. Tests use a mocked predictor (no heavy inference).
- Verified live: `uvicorn` boot → `/health` ok, `/predict` returns a titer,
  bad payloads → 422; full suite 23 passed; ruff clean. (Docker build pending a
  running daemon.) Delivered as ~11 small commits.

## 12. Final modelling workflow cleanup

**Goal.** Finish Part 1 with explicit reproducibility, biologically motivated
features, tuned sweep outputs, and a clearer modelling comparison.

**Prompt (refined).**
> Apply the final modelling workflow plan. Add substrate/feed-consumption features
> for glucose, glutamine, ammonia and lactate, verifying the real `X:`/`W:` column
> names and not assuming ammonia or lactate feeds. Make every split,
> hyperparameter sample, model initialisation, estimator seed and final refit seed
> explicit, and store those seeds in sweep CSVs and final metadata. Run exactly 20
> CDE configurations and exactly 10 XGBoost configurations, choose the best
> validation configuration for each, refit on all training data, and save final
> artifacts plus metadata. Rewrite the explanation around problem framing, three
> modelling options, feature engineering, tuned results, consistent CDE notation,
> and the fact that padding is only a flat batching trick.

**Key decisions taken.**
- XGBoost now receives targeted bioprocess accounting features in addition to
  Gompertz, TSFEL, static and meta features; only glucose and glutamine have
  matching feed channels in the provided data.
- CDE and XGBoost sweeps use separate explicit seeds for sweep sampling,
  validation splitting/CV, model or estimator initialisation, and final refits.
- The written comparison is framed as XGBoost versus CDE, with a mechanistic ODE
  kept as the interpretable but out-of-scope alternative.
- `PROMPTS.md` remains the curated AI-use record; local `claude_*.txt` scratch
  files are ignored and purged from repository history.

<!--
Template for subsequent entries:

## N. <short title>

**Goal.** <one line>

**Prompt (refined).**
> <flowing version of the instruction>

**Key decisions taken.**
- <decision + one-line rationale>
-->
