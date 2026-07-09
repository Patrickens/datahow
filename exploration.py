"""Marimo exploration notebook for the DataHow titer-prediction challenge.

Run interactively with:   uv run marimo edit exploration.py
Or view read-only with:   uv run marimo run exploration.py

The plotting functions live in ``titer_prediction.plotting`` so this notebook and
the README figures share one source of truth; the narrative lives here.
"""

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Predicting final mAb titer — modelling walkthrough

    This notebook tells the story of **Part 1** of the challenge: understand the
    data, turn variable-length bioprocess trajectories into features, compare a
    strong XGBoost baseline with a neural CDE, and predict the **final product
    titer**.

    Each experiment is a simulated fed-batch bioreactor run recorded as a short
    daily time series. Variables follow a prefix convention:

    | Prefix | Role | Examples |
    | ------ | ---- | -------- |
    | `Z:` | **design scalars** — the recipe, fixed before the run | feed start/end, pH/temp setpoints, planned duration |
    | `W:` | **control inputs** applied over time | temperature, pH, glucose/glutamine feed |
    | `X:` | **measured states** | VCD, glucose, glutamine, ammonia, lactate, lysed |
    | `Y:` | **target** — final titer (one scalar per run) | — |

    The two modelling challenges we keep in mind throughout:

    1. **Variable length → a single scalar.** Runs differ in duration; we must map
       a variable-length multivariate path to one number.
    2. **Discontinuous controls.** Feeds switch on/off at discrete days, so the
       driving signals are step-like, not smooth.

    With ~100 experiments, clean decisions and reproducible evaluation matter more
    than squeezing out maximum performance.
    """)
    return


@app.cell
def _(mo):
    from titer_prediction import plotting

    df, targets = plotting.load_train()

    n_exp = df["Exp"].nunique()
    lengths = df.groupby("Exp")["Time[day]"].size()
    summary = mo.md(
        f"""
        ## 1. The training data

        - **{n_exp} experiments**, **{len(df)} rows** total.
        - Each run spans **{int(lengths.min())}–{int(lengths.max())} daily timepoints**
          (variable length — challenge #1).
        - Final titer ranges from **{targets.min():.0f}** to **{targets.max():.0f}**
          (mean {targets.mean():.0f}), and is right-skewed — one reason we model it in
          log space later.
        """
    )
    summary
    return df, plotting, targets


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Modelling options

    The target is one scalar final titer per experiment, while the inputs are short,
    variable-length bioprocess trajectories. We considered three options:

    1. **XGBoost on engineered features** — fast, strong on small data, easy to deploy,
       and interpretable through feature importance; the cost is that time dependence
       must be engineered manually.
    2. **Neural CDE** — consumes the full path, handles unequal sampling/missingness,
       discontinuous controls, and interpolation choices; the cost is that it is less
       interpretable and harder to explain to biologists.
    3. **Mechanistic ODE with event-driven controls** — biologically interpretable
       states and parameters; the cost is formulation time, identifiability, event
       handling, solver difficulty, and misspecification risk.

    For this take-home the useful comparison is therefore **XGBoost versus the neural
    CDE**.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The target: final titer

    The violin below (with every experiment as a jittered point) shows the target's
    distribution. It is **right-skewed** with a long high-titer tail — the mean sits
    above the median. Two consequences we act on later: we train on `log1p(titer)` to
    tame the skew, and the sparse high-titer tail is what makes those (most valuable)
    runs hardest to predict.
    """)
    return


@app.cell
def _(plotting, targets):
    fig_titer = plotting.plot_titer_distribution(targets)
    fig_titer
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Measured state trajectories

    Below, every line is one experiment, coloured by its **final titer** (dark =
    low, yellow = high). Grouped by biological role rather than one plot per
    variable:

    - **VCD** grows sigmoidally then plateaus/declines — the classic growth curve.
      Higher, more sustained growth trends toward higher titer.
    - **Substrates** (glucose, glutamine) *rise* over most of the run — which can look
      surprising. It is real, not a plotting artefact: these are **fed**, and in this
      dataset the **feed rate exceeds cellular uptake**, so they *accumulate* while
      feeding is on and are only **drawn back down once the feed stops** (the mean
      glucose peaks ~day 10 at ≈12 and falls to ≈5 by day 14). The early day-1/2 dip,
      before feeding starts, is the pure-consumption phase.
    - **Byproducts** (lactate, ammonia) accumulate as waste metabolism proceeds.
      Ammonia is *produced* from glutamine/amino-acid catabolism (it is not fed), so
      its steady rise is expected. Lactate is the interesting one — in the **longer
      runs it rises and then falls again**.
      This is the classic **lactate metabolic shift**: the cells switch from net
      lactate *production* (glycolytic overflow) to net *consumption*, typically once
      glucose starts to become limiting. It is a well-known marker of healthy,
      productive fed-batch CHO cultures and tends to coincide with higher final titer —
      so it is a genuinely informative signal, not just noise.
    - **Lysed fraction** rises late as cultures age — high-titer runs are those kept
      viable and productive for longer.
    """)
    return


@app.cell
def _(df, plotting, targets):
    fig_state = plotting.plot_state_timecourses(df, targets)
    fig_state
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Control inputs — the discontinuities

    The feed profiles are **step functions**: glucose/glutamine feeding switches on
    for a window and off again, and temperature/pH shift at set days. This is
    challenge #2 made visible — a smooth (linear/spline) interpolation of these
    signals would fabricate ramps that never happened, which is exactly why the
    neural CDE **step-interpolates the `W:` controls** (while linearly interpolating
    the continuous `X:` states — see the CDE section).
    """)
    return


@app.cell
def _(df, plotting, targets):
    fig_controls = plotting.plot_control_timecourses(df, targets)
    fig_controls
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Feature engineering: from trajectories to features

    A generic regressor needs a **fixed-length feature vector** per experiment. We
    collapse each variable-length trajectory five complementary ways:

    1. **Gompertz growth-curve parameters** (this section) — fit to VCD,
    2. **Substrate/feed-consumption features** — initial/final concentrations, total
       feed integral for glucose/glutamine, initial plus fed amount, and apparent
       consumed amount,
    3. **Cell-population accounting features** — estimated total cell density from
       viable density and lysed fraction,
    4. **TSFEL features** — a curated, interpretable set of statistical & temporal
       features per state channel (including the **area under the curve**),
    5. **static + meta** — the pass-through `Z:` design scalars plus the observed
       duration and number of timepoints.

    ### Gompertz fits on VCD

    We fit a 4-parameter-with-baseline Gompertz curve,
    $y(t) = y_0 + a\,e^{-b\,e^{-k_g (t - t_i)}}$, to each VCD trajectory. It
    summarises a whole growth curve with a handful of **interpretable** numbers:
    amplitude $a$, growth rate $k_g$, inflection time $t_i$, shape $b$, baseline
    $y_0$. Across a spread of experiments the fit is excellent (R² ≈ 0.99), and the
    highest-titer run shows the full sigmoid plateau.

    **What Gompertz cannot do.** It is a single monotone sigmoid, so it captures the
    *shape of growth* but **not the sequential substrate dynamics** — sequential substrate consumption, 
    feed-driven replenishment, or the lactate production→consumption switch seen above. 
    Those coupled, order-dependent effects
    are exactly what the TSFEL features might pick up, and what the CDE models most
    naturally by integrating along the trajectory.
    """)
    return


@app.cell
def _(df, plotting, targets):
    fig_gompertz = plotting.plot_gompertz_examples(df, targets)
    fig_gompertz
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Choosing an automated feature library — and landing on TSFEL

    How to turn each channel into features automatically? We iterated:

    - **tsfresh** — **conflicts with our JAX/diffrax stack** (its `numba`/`stumpy`
      dependency pins a numpy JAX won't accept) and emits **200+ features**, more than
      helps on ~100 samples.
    - **catch22** — 22 canonical *dynamical-systems* features. Worked well (baseline
      **R² ≈ 0.82**), but they are generic descriptors, not **domain-meaningful** — no
      area-under-the-curve, and ∫VCD is one of the most physically motivated titer
      predictors.
    - **TSFEL** — the one we adopted: `numba`-free, **interpretable** statistical &
      temporal features that *include* AUC, and **extensible** so we can fold in custom
      features.

    **Curated TSFEL subset** (~25 features per `X:` channel; spectral/fractal dropped —
    little signal on ~10-point series):

    - *Level & spread* — `Mean`, `Median`, `Max`, `Min`, `Standard deviation`,
      `Variance`, `Root mean square`, `Interquartile range`, `Mean absolute deviation`,
      `Peak to peak distance`.
    - *Value distribution* — `Skewness`, `Kurtosis`, `Entropy`, `Absolute energy`.
    - *Accumulation & trend* — `Area under the curve` (∫ over time, e.g. ∫VCD), `Slope`,
      `Centroid`, `Mean diff`, `Mean absolute diff`.
    - *Trajectory shape* — `Autocorrelation`, `Positive turning points`, `Negative
      turning points`, `Zero crossing rate`, `Neighbourhood peaks`, `Signal distance`.

    **Custom features** (registered with `@set_domain`, so they live in the same
    pipeline):

    - *Gompertz* — the fitted growth-curve parameters, applied to VCD.
    - *Substrate/feed* — for glucose and glutamine: initial/final concentration, total
      feed integral (`W:FeedGlc`, `W:FeedGln`), initial-plus-fed, and apparent consumed
      amount. (Concentration AUCs already come from TSFEL, so not duplicated.)
    - *Cell-population* — from viable cells `X:VCD` and lysed fraction `X:Lysed`, the
      derived total density $\mathrm{total}(t)=\mathrm{VCD}(t)/(1-\mathrm{Lysed}(t))$,
      summarised by initial, final, max, and AUC.
    """)
    return


@app.cell
def _(mo, plotting):
    X, y = plotting.baseline_matrix()
    feat_summary = mo.md(
        f"""
        ### The assembled feature matrix

        Stacking all three families gives **{X.shape[0]} experiments ×
        {X.shape[1]} features**. That is a lot of features for so few samples, so the
        model must be regularised and evaluated honestly with cross-validation — which
        is exactly what we do next.
        """
    )
    feat_summary
    return X, y


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Baseline regression (XGBoost)

    We fit a gradient-boosted tree ensemble to predict `log1p(titer)`. This helps with
    right-skew and approximately proportional / heteroskedastic noise, but it does not
    strictly guarantee non-negative predictions unless the inverse `expm1` output is
    clipped. Evaluation uses **repeated K-fold cross-validation**, always alongside a
    **mean-predictor baseline**, so the reported numbers are honest.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The math (XGBoost, to refresh)

    Gradient boosting builds an **additive ensemble of regression trees**,

    $$ \hat{y}_i = F(x_i) = \sum_{k=1}^{K} f_k(x_i), \qquad f_k \in \mathcal{F}\ \text{(CART trees)}, $$

    fit **stage-wise**: at round $m$ we add one tree to the current model,
    $F_m = F_{m-1} + \eta\, f_m$ (learning rate $\eta$), chosen to minimise a
    *regularised* objective

    $$ \mathcal{L}^{(m)} = \sum_i \ell\big(y_i,\ F_{m-1}(x_i) + f_m(x_i)\big) + \Omega(f_m),
       \qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \lVert w \rVert^2, $$

    where $T$ is the number of leaves and $w$ their weights. A **2nd-order Taylor**
    expansion of the loss around $F_{m-1}$ gives

    $$ \mathcal{L}^{(m)} \approx \sum_i \Big[ g_i\, f_m(x_i) + \tfrac{1}{2} h_i\, f_m(x_i)^2 \Big] + \Omega(f_m),
       \qquad g_i = \partial_{\hat y}\,\ell,\ \ h_i = \partial^2_{\hat y}\,\ell. $$

    For a fixed tree structure the optimal weight of leaf $j$ (with instance set $I_j$)
    and the **gain** used to score a candidate split are

    $$
    w_j^{*} = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda},
    \qquad
    \text{gain} = \tfrac{1}{2}\left[
        \frac{G_L^2}{H_L + \lambda}
        + \frac{G_R^2}{H_R + \lambda}
        - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
    \right] - \gamma,
    $$

    where $G_{L},G_{R}$ and $H_{L},H_{R}$ are the sums of $g_i$ and $h_i$ over the
    left/right child. Trees are grown greedily by maximising that gain. With our **squared-error** loss on
    $u=\log(1+y)$ the statistics are simply $g_i = \hat{u}_i - u_i$ and $h_i = 1$.

    **Why this model here.** Trees handle heterogeneous, unscaled features and missing
    values natively, need little tuning, and — with shallow depth, column/row subsampling
    and the $\lambda,\gamma$ penalties — control variance on our ~100 samples. The `log1p`
    target makes errors effectively multiplicative and keeps the loss well-behaved across
    the wide titer range.
    """)
    return


@app.cell
def _(mo):
    import json as _json
    from pathlib import Path as _Path

    _metadata = _json.loads(_Path("artifacts/xgb_best_metadata.json").read_text())

    mo.md(
        f"""
        ### Small XGBoost sweep

        I sampled **{_metadata["n_configs"]}** shallow-tree configurations with fixed
        seeds (`sweep={_metadata["seeds"]["sweep_seed"]}`,
        `cv={_metadata["seeds"]["cv_seed"]}`). The table below is sorted by validation
        R² and shows the top configurations.
        """
    )
    return


@app.cell
def _():
    import pandas as _pd

    _xgb_sweep = _pd.read_csv("artifacts/xgb_sweep_results.csv")
    _xgb_sweep_display = (
        _xgb_sweep.sort_values("xgb_r2", ascending=False)
        .head(5)
        .loc[
            :,
            [
                "run_index",
                "max_depth",
                "learning_rate",
                "n_estimators",
                "subsample",
                "colsample_bytree",
                "reg_lambda",
                "min_child_weight",
                "xgb_rmse",
                "xgb_mape",
                "xgb_r2",
            ],
        ]
        .round({"xgb_rmse": 0, "xgb_mape": 3, "xgb_r2": 3})
    )
    _xgb_sweep_display
    return


@app.cell
def _(mo):
    import json as _json
    from pathlib import Path as _Path

    _metadata = _json.loads(_Path("artifacts/xgb_best_metadata.json").read_text())
    _best = _metadata["best_validation"]
    _final_cv = _metadata["final_cv"]["xgboost"]
    _baseline = _metadata["final_cv"]["baseline_mean"]
    _cfg = _metadata["best_config"]

    mo.md(
        f"""
        The best configuration was run **{int(_best["run_index"])}**:
        `max_depth={_cfg["max_depth"]}`, `learning_rate={_cfg["learning_rate"]}`,
        `n_estimators={_cfg["n_estimators"]}`, `subsample={_cfg["subsample"]}`,
        `colsample_bytree={_cfg["colsample_bytree"]}`, `reg_lambda={_cfg["reg_lambda"]}`,
        `min_child_weight={_cfg["min_child_weight"]}`.

        After the final refit/evaluation pass it reached **RMSE ≈
        {_final_cv["rmse"]:.0f}**, **MAPE ≈ {100 * _final_cv["mape"]:.1f}%**, and
        **R² ≈ {_final_cv["r2"]:.2f}**, versus the mean baseline at **RMSE ≈
        {_baseline["rmse"]:.0f}** and **R² ≈ {_baseline["r2"]:.2f}**.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Best XGBoost: out-of-fold predictions

    Predicted-vs-actual (left) hugs the diagonal for most runs — the selected
    sweep configuration reaches repeated-CV **R² ≈ 0.83**, far above the mean
    predictor (~0). The model poorly predicts the high-titer regime, which is
    unfortunate because these are exactly the most interesting experiments from a
    process-optimization perspective.

    Two compounding causes: (i)
    a small-data effect — very few examples exist in the high-titer regime for the
    trees to learn from; and (ii) tree ensembles **cannot extrapolate** beyond the
    range of the training targets, so they saturate. Mitigations worth exploring:
    **sample weighting** or an **asymmetric/quantile loss** to prioritise high titers,
    **targeted data collection** in that regime, or leaning on the mechanistic/CDE
    route, which can extrapolate through structure rather than interpolation.
    """)
    return


@app.cell
def _(X, plotting, y):
    fig_cv = plotting.plot_cv_predictions(X, y)
    fig_cv
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Which features matter?

    The gain-based importances are a reassuring sanity check — and a vindication of the
    feature-engineering choice. The top features are all **biologically meaningful**. Leading
    the table is the **AUC of estimated total cell density** (`bio_total_cell_density_auc`,
    our custom cell-population feature), followed by the **apparent glutamine and glucose
    consumed** (feed-accounting) and the **area under the VCD curve**
    (`tsfel_X:VCD_Area under the curve`, essentially the **integral of viable cells** — the
    classical mechanistic predictor of product). The rest are substrate/byproduct AUCs,
    levels, and feed summaries. This is exactly the kind of feature catch22 could **not**
    provide — its top-ranked features were abstract dynamical descriptors, whereas here the
    model leans on quantities a process scientist would reach for.
    """)
    return


@app.cell
def _(plotting):
    # Cached to artifacts/feature_importance.csv; pass regenerate=True to refit.
    importance_table = plotting.feature_importance_table(top=15)
    importance_table
    return


@app.cell
def _(mo):
    mo.md(r"""
    **How XGBoost scores importance (gain).** These numbers are the **gain**
    importance. Recall the split *gain* from the math above — the loss reduction a
    split buys, $\tfrac12\big[\tfrac{G_L^2}{H_L+\lambda}+\tfrac{G_R^2}{H_R+\lambda}
    -\tfrac{(G_L+G_R)^2}{H_L+H_R+\lambda}\big]-\gamma$. A feature's gain importance
    sums that reduction over **every split that uses it**, across all trees; we then
    normalise so the values sum to 1. So it answers *"how much did this feature reduce
    the loss when the model chose to split on it?"* — not merely how often it was used
    (`weight`) or how many samples its splits touched (`cover`).

    Two caveats to read it honestly: gain is biased toward **high-cardinality /
    continuous** features (they offer more candidate split points), and among
    **correlated** features the credit is split somewhat arbitrarily between them. So
    treat the ranking as a guide corroborated by domain sense — which is exactly why
    the biologically meaningful ordering above is reassuring rather than surprising.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Beyond the baseline — the neural CDE

    The baseline is strong precisely because this is a small, tabular-friendly
    dataset. The companion **neural CDE** (`titer_prediction.cde`) instead ingests
    the raw trajectories directly. A quick tour of the math it rests on.

    ### The math

    A *controlled* differential equation drives a learned hidden state
    $h(s)\in\mathbb{R}^{d}$ along an input/control path
    $C(s)\in\mathbb{R}^{|C|}$ built by interpolating the observed trajectory:

    $$
    C(s) = [t(s), W(s), X_{\mathrm{obs}}(s)], \qquad |C| = 1 + |W| + |X_{\mathrm{obs}}|
    $$

    $$
    h_0 = \zeta_\theta(Z, C_0), \qquad
    C_0 = [t_0, W(t_0), X_{\mathrm{obs}}(t_0)]
    $$

    $$
    \mathrm{d}h(s) = f_\theta(h(s))\,\mathrm{d}C(s), \qquad
    \hat{y} = \ell_\theta(h(S)).
    $$

    - The **vector field** $f_\theta:\mathbb{R}^{d}\to\mathbb{R}^{d\times |C|}$ is a
      neural network mapping the hidden state to a matrix; the integrand
      $f_\theta(h)\,\mathrm{d}C$ is a matrix–vector product, so the model learns how the
      *rates of change of the inputs* steer the latent state (a Riemann–Stieltjes
      integral).
    - $Z$ denotes static variables used only for initialisation. In code these are the
      design scalars with **no `W:` counterpart**, **stirring and dissolved oxygen**
      (`Z:Stir`, `Z:DO`).
    - $d$ is not a biological dimension; it is a model-capacity hyperparameter. Small
      $d$ may underfit, while large $d$ may overfit, so it is tuned in the CDE sweep.
    - $C_0$ matters because a CDE only sees control increments $\mathrm{d}C$, so the
      absolute initial VCD / substrate levels would otherwise never reach it.

    **Interpolation is an inductive bias.** To turn discrete samples into a path we
    must *interpolate* — and how we interpolate is a modelling assumption about the
    process between observations, not a claim about the true biology. We use a
    **mixed** convention, matched to each channel group:

    - **`W:` controls → step (rectilinear).** Feeds and setpoint switches are
      genuinely discontinuous; linear interpolation would fabricate ramps that never
      happened.
    - **`X:` observations → piecewise linear.** These are sampled from continuous-ish
      process states, so a staircase would fabricate jumps. Linear is the honest
      minimal assumption — *not* a claim the biology is exactly linear.
    - **real time → channel 0**, linearly interpolated, so the model stays time-aware.

    (Full rectilinear interpolation of *everything* would be defensible only under an
    **online-information** reading — where each new measurement is a jump in our
    *information*, not in the physical state. We are doing offline whole-trajectory
    regression, so the mixed convention is the more faithful bias.)

    **Why a path parameter $s$?** A `W:` step has *zero duration in real time*, so an
    ODE in real time would give it zero measure and silently drop it. Instead we place
    the knots on a strictly-increasing artificial parameter $s = 0,1,2,\dots$: a control
    jump becomes a segment of finite length in $s$ (with real time held constant on that
    segment, since real time is just another channel). Over such a segment $\Delta C \neq
    0$, so the increment $\int f_\theta(h)\,\mathrm{d}C = f_\theta(h)\,\Delta C$ is
    captured. This is why `make_mixed_cde_path` builds a *flow* segment (time & `X:` move,
    `W:` held) followed by a *jump* segment (`W:` moves, time & `X:` held) per interval.

    *Concept — why a CDE?* An ordinary neural ODE evolves autonomously,
    $\mathrm{d}h = f_\theta(h)\,\mathrm{d}t$; it cannot ingest an incoming data stream.
    A **controlled** DE replaces $\mathrm{d}t$ with $\mathrm{d}C$, so the *data itself*
    drives the dynamics. This is the continuous-time generalisation of an RNN — and
    unlike an RNN it handles irregular sampling and missing data by construction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What the code does, step by step

    Mapping the maths onto `titer_prediction.cde` — the forward pass lives in
    `NeuralCDE.__call__`:

    1. **Standardise & assemble the path** (`build_arrays`). Each experiment becomes a
       matrix `ys` of shape `(T, c)` whose channels are *real time* followed by the
       standardised `W:`/`X:` measurements. Shorter runs are right-padded by repeating
       the last row, so the padded tail is flat and contributes nothing to the integral.

    2. **Build the mixed control path** (`make_mixed_cde_path(ys, n_w)`). Per interval,
       a *flow* segment moves real time and the `X:` states linearly while the `W:`
       controls are held, then a *jump* segment holds time & `X:` and steps the `W:`
       controls. So `W:` is step-interpolated and `X:` is linear, inside one path $C(s)$.
       *Concept:* the control path is the continuous object the CDE is driven by, and its
       increments $\mathrm{d}C$ are what enter the integral.

    3. **Integrate over a path parameter, not time** (`s = jnp.arange(...)`,
       `LinearInterpolation(s, path)`). The knots sit on a strictly increasing
       $s = 0,1,2,\dots$; a `W:` jump (zero real-time duration) becomes a finite segment
       in $s$. *Concept — reparametrisation invariance:* a CDE's output depends on the
       **geometry of the path**, not the speed it is traversed, so integrating in $s$ is
       legitimate — and it is what lets the solver see the control jumps.

    4. **Initial state from static design + first observation**
       (`h0 = self.initial(concat([static, ys[0]]))`). The MLP $\zeta_\theta$ maps the
       no-`W:`-counterpart scalars (`Z:Stir`, `Z:DO`, see `STATIC_INIT_COLS`) **and the
       initial observation** $C_0 = $ `ys[0]` $ = [t_0, W_0, X_0]$ to
       $h(s_0)\in\mathbb{R}^{d}$. $C_0$ matters because the CDE only sees increments, so
       the absolute initial VCD / substrate levels would otherwise never reach it.

    5. **Define the controlled dynamics** (`ControlTerm(self.func, control)`). `self.func`
       is $f_\theta$, an MLP returning a $d\times c$ matrix; the term encodes
       $\mathrm{d}h = f_\theta(h)\,\mathrm{d}C$. *Concept:* the update is a learned
       **matrix–vector product with the data increment**, not a fixed recurrent cell.

    6. **Solve** (`diffeqsolve(term, Heun(), stepsize_controller=StepTo(ts=s))`). We step
       exactly on the knot grid; Heun's method (2nd-order Runge–Kutta) advances
       $h_{n+1} \approx h_n + f_\theta(h_n)\,\big(C(s_{n+1}) - C(s_n)\big)$ using the
       solver's control increment $\Delta C$. Stepping on the knots guarantees every jump
       is integrated.

    7. **Read out** (`self.readout(sol.ys[-1])`). A linear map $\ell_\theta$ turns the
       terminal state into the predicted (standardised, log) titer.

    **Training.** We minimise MSE against standardised $\log(1+\text{titer})$ with Adam
    (`optax`); gradients flow *through the ODE solve* by automatic differentiation
    (`equinox`/JAX), so $\zeta_\theta$, $f_\theta$ and $\ell_\theta$ are learned
    end-to-end. We fit on a train split, report a held-out score, then refit on all data
    for the deployed model.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **In brief.**

    - **Init:** `Z:` static (`Z:Stir`, `Z:DO`) **and** the first observation
      $C_0 = [t_0, W_0, X_{\mathrm{obs},0}]$ initialise the hidden state $h_0$.
    - **Dynamics:** the path $C(s) = [t(s), W(s), X_{\mathrm{obs}}(s)]$ drives the CDE through its
      increments $\mathrm{d}C$ — `W:` step-interpolated, `X:` and time linear.
    - **Padding** is only for batching: the whole final row *including time* is
      repeated, so the padded tail is flat ($\mathrm{d}C = 0$) and contributes nothing
      (there is a unit test for this).
    - **Diagnostics:** we watch the training curves (below) and a small hyperparameter
      sweep (`titer-sweep`) to tell undertraining from overfitting from LR
      instability.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Seeing it: real time vs the path parameter $s$

    The single most confusing thing about this construction is that **$s$ — not real
    time — is the solver's clock**, and real time rides along as *a channel* of the path.
    The toy example below (a feed that switches on at day 2 and off at day 4, with one
    continuous state) makes it concrete.

    First, why `W:` is step-interpolated: linear interpolation invents ramps between
    daily samples that never physically happened.
    """)
    return


@app.cell
def _(plotting):
    fig_interp = plotting.plot_interpolation_comparison()
    fig_interp
    return


@app.cell
def _(mo):
    mo.md(r"""
    Next, the same path plotted against $s$. On the shaded segments the **control jumps
    while real time is held flat** — a physical discontinuity becomes a segment of
    *finite* length in $s$, which is exactly why the solver can see it.

    The gold region on the right is the padded tail used for batching. Padding is not
    extra process time: because the full final row is repeated, including the real-time
    channel, the control path is constant there: $C(s)=C(S)$ and $\mathrm{d}C=0$.
    Since $\mathrm{d}h=f_\theta(h)\,\mathrm{d}C$, a pure CDE receives no update on that
    flat tail.
    """)
    return


@app.cell
def _(plotting):
    fig_params = plotting.plot_path_parameter()
    fig_params
    return


@app.cell
def _(mo):
    mo.md(r"""
    Finally, a toy hidden state under a fixed vector field. Note it updates on **both**
    the flow segments (time/`X:` increments) **and** the shaded control-jump segments — a
    control switch genuinely moves the latent state, which is the whole point of feeding
    the controls through the path.
    """)
    return


@app.cell
def _(plotting):
    fig_toy_state = plotting.plot_cde_toy_state()
    fig_toy_state
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Why a CDE here?

    For this dataset a controlled DE is a genuinely good fit:

    - **Variable-length trajectories** are handled natively — the model integrates
      whatever path it is given, no fixed window.
    - **Irregular / unequal sampling** is fine because *real time is a channel* of the
      path; the model reads timing directly instead of assuming a fixed step.
    - **Order and timing are preserved** — unlike the bag-of-features baseline, the CDE
      respects *when* things happened.
    - **Discontinuous controls** are represented honestly (step `W:`), without
      fabricating ramps.
    - **Online updates** are natural: as new measurements arrive they simply extend the
      path, so the same model can predict mid-run (the online-prediction setting).
    - **More appropriate than a neural ODE**, because the process is *externally
      controlled*. A neural ODE evolves autonomously, $\mathrm{d}h = f_\theta(h)\,
      \mathrm{d}t$, and cannot ingest the feeds/observations; the CDE's
      $\mathrm{d}h = f_\theta(h)\,\mathrm{d}C$ is *driven by the data path*.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Training curves

    Because the CDE is trained by gradient descent (unlike the closed-form baseline), we
    track **train and validation RMSE in raw titer units** over epochs. That is easier
    to interpret than standardised log-space MSE. The untrained epoch-0 checkpoint is
    omitted because random log-space predictions can explode after `expm1` and dominate
    the plot scale; the remaining checkpoints show the actual optimisation trajectory.
    """)
    return


@app.cell
def _(plotting):
    fig_curves = plotting.plot_cde_training_curves(epochs=250)
    fig_curves
    return


@app.cell
def _(mo):
    import json as _json
    from pathlib import Path as _Path

    _metadata = _json.loads(_Path("artifacts/cde_best_metadata.json").read_text())

    mo.md(
        f"""
        ### Small CDE sweep

        I sampled **{_metadata["n_configs"]}** CDE configurations with fixed seeds
        (`sweep={_metadata["seeds"]["sweep_seed"]}`,
        `split={_metadata["seeds"]["split_seed"]}`,
        `refit={_metadata["seeds"]["refit_seed"]}`). Each run used an explicit
        model-initialisation seed and the same fixed 20% validation holdout. The table
        below is sorted by validation R² and shows the top configurations.
        """
    )
    return


@app.cell
def _():
    import pandas as _pd

    _cde_sweep = _pd.read_csv("artifacts/cde_sweep_results.csv")
    _cde_sweep_display = (
        _cde_sweep.sort_values("val_r2", ascending=False)
        .head(5)
        .loc[
            :,
            [
                "run_index",
                "epochs",
                "lr",
                "hidden_size",
                "width",
                "depth",
                "model_seed",
                "val_rmse",
                "val_mape",
                "val_r2",
            ],
        ]
        .round({"lr": 4, "val_rmse": 0, "val_mape": 3, "val_r2": 3})
    )
    _cde_sweep_display
    return


@app.cell
def _(mo):
    import json as _json
    from pathlib import Path as _Path

    _metadata = _json.loads(_Path("artifacts/cde_best_metadata.json").read_text())
    _best = _metadata["best_validation"]
    _cfg = _metadata["best_config"]

    mo.md(
        f"""
        The best configuration was run **{int(_best["run_index"])}**:
        `epochs={_cfg["epochs"]}`, `lr={_cfg["lr"]}`, `hidden_size={_cfg["hidden_size"]}`,
        `width={_cfg["width"]}`, `depth={_cfg["depth"]}`,
        `model_seed={_cfg["model_seed"]}`.
        """
    )
    return


@app.cell
def _(mo):
    import json as _json
    from pathlib import Path as _Path

    _metadata = _json.loads(_Path("artifacts/cde_best_metadata.json").read_text())
    _best = _metadata["best_validation"]

    mo.md(
        f"""
        ### Best CDE: result

        The selected CDE reached **validation RMSE ≈ {_best["val_rmse"]:.0f}**,
        **MAPE ≈ {100 * _best["val_mape"]:.1f}%**, and **R² ≈ {_best["val_r2"]:.2f}**
        on the fixed 20% holdout. That is encouraging, but the single holdout is still
        noisy on ~100 experiments. I therefore keep the main conclusion pragmatic:
        XGBoost is the dependable deployment baseline, while the CDE demonstrates the
        path-based methodology for ragged, controlled trajectories.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Hybrid/mechanistic alternative

    A more interpretable production direction would replace the black-box CDE vector
    field with bioprocess mass balances, keeping only selected rates or residual terms
    learnable. A minimal skeleton might be

    $$
    \begin{aligned}
    \dot{V} &= \big(\mu(\cdot) - \mu_d(\cdot)\big)\,V, \\
    \dot{G} &= -\,q_{\mathrm{glc}}(\cdot)\,V + F_{\mathrm{glc}}(t), \\
    \dot{P} &= q_p(\cdot)\,V,
    \end{aligned}
    $$

    with $V$ = viable cell density, $G$ = glucose, and $P$ = product. The parameters
    now have process meaning: growth/death rates, glucose uptake, and specific
    productivity. A hybrid model can keep this structure and learn only how selected
    rates depend on state and conditions.

    I do not build this here because it needs explicit rate laws, identifiable
    parameters, and event handling for feed/setpoint switches. For a take-home, the CDE
    is the cleaner path-based model; mechanistic or hybrid ODEs are the more
    interpretable production direction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Takeaways

    - Clean, interpretable feature engineering plus a well-regularised,
      honestly-benchmarked baseline reaches **R² ≈ 0.83**.
    - The neural CDE demonstrates the path-based methodology; **hybrid mechanistic
      models** are the interpretable, data-efficient destination.
    - Performance was never the point of this challenge — clarity of the pipeline and of
      the decisions was.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Using the inference service

    The trained baseline is served behind a small **FastAPI** app
    (`titer_prediction.service`) with two endpoints from the provided OpenAPI spec:
    `GET /health` (liveness + whether a model is loaded) and `POST /predict` (one
    experiment → predicted final titer). The model is loaded once at startup from
    `MODEL_PATH` (default `artifacts/xgb_baseline.joblib`; point it at a `.eqx` bundle
    to serve the neural CDE instead). Invalid payloads return **400**, and if no model
    is loaded `/predict` returns **503** while `/health` still reports `model_loaded:false`.

    **Run locally**

    ```bash
    # serve the baseline (uses MODEL_PATH, default artifacts/xgb_baseline.joblib)
    uv run uvicorn titer_prediction.service.app:app --port 8000
    ```

    **Call it** — the request carries `timestamps` and a `values` map keyed by the
    `Z:`/`W:`/`X:` convention (a ready example lives in `scripts/sample_payload.json`):

    ```bash
    curl -s localhost:8000/health
    # {"status":"ok","model_loaded":true}

    curl -s -X POST localhost:8000/predict \
        -H 'Content-Type: application/json' \
        --data @scripts/sample_payload.json
    # {"prediction": <titer>, "target":"Y:Titer", "model_type":"xgboost", "n_timepoints":15, ...}
    ```

    **Batch a whole CSV** into the target-template shape (`RowID, Exp, Time[day],
    Y:Titer`) — this reuses the exact `/predict` conversion path:

    ```bash
    uv run python -m titer_prediction.service.batch_predict \
        --data data/datahow_interview_test_data.csv \
        --model artifacts/xgb_baseline.joblib \
        --out artifacts/test_predictions.csv
    ```

    **Docker** — the image doesn't bake in the (git-ignored) model, so mount the
    artifacts at runtime:

    ```bash
    docker build -t datahow-titer-service .
    docker run --rm -p 8000:8000 \
        -e MODEL_PATH=/app/artifacts/xgb_baseline.joblib \
        -v "$PWD/artifacts:/app/artifacts:ro" datahow-titer-service
    ```

    `scripts/smoke_docker.sh` builds the image and asserts this whole contract
    end-to-end (health, predict, 400 on a bad payload, 503 with no model).
    """)
    return


if __name__ == "__main__":
    app.run()
