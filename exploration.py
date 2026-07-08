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
    # Predicting final mAb titer — data & baseline walkthrough

    This notebook tells the story of **Part 1** of the challenge: understand the
    data, turn variable-length bioprocess trajectories into features, and build a
    baseline regression model for the **final product titer**.

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
    ### Measured state trajectories

    Below, every line is one experiment, coloured by its **final titer** (dark =
    low, yellow = high). Grouped by biological role rather than one plot per
    variable:

    - **VCD** grows sigmoidally then plateaus/declines — the classic growth curve.
      Higher, more sustained growth trends toward higher titer.
    - **Substrates** (glucose, glutamine) are consumed and replenished by feeding.
    - **Byproducts** (lactate, ammonia) accumulate as waste metabolism proceeds — but
      look closely at the **longer runs, where lactate rises and then falls again**.
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
    neural CDE uses **rectilinear (staircase)** interpolation.
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
    ## 2. Preprocessing: from trajectories to features

    A generic regressor needs a **fixed-length feature vector** per experiment. We
    collapse each variable-length trajectory three complementary ways:

    1. **Gompertz growth-curve parameters** (this section),
    2. **catch22** — 22 canonical time-series features per state channel,
    3. **simple aggregates** — first/last/min/max/mean/std/AUC/slope, plus the
       pass-through `Z:` design scalars.

    ### Gompertz fits on VCD

    We fit a 4-parameter-with-baseline Gompertz curve,
    $y(t) = y_0 + a\,e^{-b\,e^{-k_g (t - t_i)}}$, to each VCD trajectory. It
    summarises a whole growth curve with a handful of **interpretable** numbers:
    amplitude $a$, growth rate $k_g$, inflection time $t_i$, shape $b$, baseline
    $y_0$. Across a spread of experiments the fit is excellent (R² ≈ 0.99), and the
    highest-titer run shows the full sigmoid plateau.

    **What Gompertz cannot do.** It is a single monotone sigmoid, so it captures the
    *shape of growth* but **not the sequential substrate dynamics** — the ordered
    depletion of glucose then glutamine, feed-driven replenishment, or the lactate
    production→consumption switch seen above. Those coupled, order-dependent effects
    are exactly what the catch22 features pick up, and what the CDE models most
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
    ### Do the Gompertz parameters carry signal?

    Plotting each parameter against final titer shows they do — and in an
    interpretable way. Larger growth **amplitude** and a **later inflection time**
    (growth sustained for longer before plateauing) both trend toward higher titer,
    while a very fast early **growth rate** slightly anti-correlates (burn-bright,
    burn-out). These are the kinds of relationships a process scientist can reason
    about.
    """)
    return


@app.cell
def _(df, plotting, targets):
    fig_signal = plotting.plot_gompertz_signal(df, targets)
    fig_signal
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### catch22 — the 22 features (applied per state channel)

    Gompertz captures the *shape of growth*; **catch22** captures everything else about
    a trajectory's dynamics. It is a curated subset of the 7000+ *hctsa* features,
    selected to be minimally redundant yet collectively informative across very diverse
    time series (Lubba et al., 2019, *Data Min. Knowl. Disc.*). We compute all 22 on
    each `X:` state channel (6 channels → 132 features). Grouped by what they measure:

    **Distribution** (value shape, ignoring time order)

    - `DN_HistogramMode_5`, `DN_HistogramMode_10` — the mode (most common value) of the
      z-scored series from a 5- and 10-bin histogram.

    **Linear autocorrelation & spectrum** (characteristic timescales)

    - `CO_f1ecac` — lag at which the autocorrelation first falls to $1/e$ (a
      decorrelation time).
    - `CO_FirstMin_ac` — lag of the first minimum of the autocorrelation function.
    - `SP_Summaries_welch_rect_area_5_1` — power in the lowest fifth of frequencies of
      the (Welch) power spectrum.
    - `SP_Summaries_welch_rect_centroid` — centroid (centre of mass) of the power
      spectrum.

    **Nonlinear autocorrelation / phase space**

    - `CO_HistogramAMI_even_2_5` — automutual information at lag 2 (nonlinear dependence).
    - `IN_AutoMutualInfoStats_40_gaussian_fmmi` — lag of the first minimum of the
      automutual information.
    - `CO_trev_1_num` — time-reversibility statistic (asymmetry under reversing time).
    - `CO_Embed2_Dist_tau_d_expfit_meandiff` — spread of successive distances in a 2-D
      time-delay embedding.

    **Predictability / forecasting**

    - `FC_LocalSimple_mean1_tauresrat` — how the correlation length changes after
      differencing.
    - `FC_LocalSimple_mean3_stderr` — error of predicting the next point from the mean of
      the last three.

    **Successive differences**

    - `MD_hrv_classic_pnn40` — fraction of successive differences exceeding a threshold
      (prevalence of large jumps; a heart-rate-variability heritage).

    **Symbolic patterns / motifs**

    - `SB_BinaryStats_mean_longstretch1` — longest run of consecutive values above the
      mean.
    - `SB_BinaryStats_diff_longstretch0` — longest run of consecutive decreases.
    - `SB_MotifThree_quantile_hh` — entropy of successive symbols in a 3-letter (quantile)
      encoding.
    - `SB_TransitionMatrix_3ac_sumdiagcov` — statistics of transitions between
      coarse-grained states.

    **Self-affine scaling** (fluctuation analysis)

    - `SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1` — detrended fluctuation analysis (DFA)
      scaling.
    - `SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1` — rescaled-range (Hurst-like) scaling.

    **Extreme-event timing**

    - `DN_OutlierInclude_p_001_mdrmd`, `DN_OutlierInclude_n_001_mdrmd` — how positive /
      negative extreme events are distributed through time.

    **Periodicity**

    - `PD_PeriodicityWang_th0_01` — Wang's estimate of the dominant period.

    Recall from the importance plot below that features from several of these groups rank
    highly (e.g. `SB_BinaryStats_mean_longstretch1` on lactate, `CO_FirstMin_ac` on VCD) —
    the diversity earns its keep.
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
    ## 3. Baseline regression (XGBoost)

    We fit a gradient-boosted tree ensemble to predict `log1p(titer)` (the target is
    positive and right-skewed). Evaluation uses **repeated K-fold cross-validation**,
    always alongside a **mean-predictor baseline**, so the reported numbers are honest.
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

    $$ w_j^{*} = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}, \qquad
       \text{gain} = \tfrac{1}{2}\!\left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda}
       - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma . $$

    Trees are grown greedily by maximising that gain. With our **squared-error** loss on
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
    mo.md(r"""
    ### Out-of-fold predictions

    Predicted-vs-actual (left) hugs the diagonal for most runs — CV **R² ≈ 0.82**,
    far above the mean predictor (~0). The residuals (right) reveal the main
    weakness: the model **under-predicts the few very high-titer runs**.

    This is the practically important failure mode: the high-titer runs are exactly
    the **good experiments** we most want to predict well. Two compounding causes: (i)
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

    The gain-based importances confirm that **all three feature families
    contribute**: catch22 features on lactate/VCD/glutamine/ammonia, our own
    aggregates (notably **`X:VCD_auc`** — the integral of viable cells, the
    classical mechanistic predictor of product), and the Gompertz **inflection
    time**. This is a reassuring sanity check: the model leans on features that make
    biological sense.
    """)
    return


@app.cell
def _(X, plotting, y):
    fig_importance = plotting.plot_feature_importance(X, y)
    fig_importance
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Beyond the baseline — the neural CDE

    The baseline is strong precisely because this is a small, tabular-friendly
    dataset. The companion **neural CDE** (`titer_prediction.cde`) instead ingests
    the raw trajectories directly. A quick tour of the math it rests on.

    ### The math

    A *controlled* differential equation drives a hidden state
    $z(t)\in\mathbb{R}^{h}$ along a **control path** $X(t)\in\mathbb{R}^{c}$ built by
    interpolating the observed trajectory:

    $$
    z(t_0) = \zeta_\theta(Z), \qquad
    z(t) = z(t_0) + \int_{t_0}^{t} f_\theta\!\big(z(\tau)\big)\,\mathrm{d}X(\tau).
    $$

    - The **vector field** $f_\theta:\mathbb{R}^{h}\to\mathbb{R}^{h\times c}$ is a
      neural network mapping the hidden state to a matrix; the integrand
      $f_\theta(z)\,\mathrm{d}X$ is a matrix–vector product, so the model learns how the
      *rates of change of the inputs* steer the latent state (a Riemann–Stieltjes
      integral).
    - The static design scalars $Z$ set the initial state through a small network
      $\zeta_\theta$ — this is how the `Z:` recipe enters.
    - The prediction is a linear readout of the terminal state,
      $\hat{y}=\ell_\theta\!\big(z(T)\big)$.

    **Why this shape fits our data.** Where $X$ is differentiable the integral becomes
    an ODE,
    $\tfrac{\mathrm{d}z}{\mathrm{d}t}=f_\theta\!\big(z(t)\big)\,\tfrac{\mathrm{d}X}{\mathrm{d}t}$,
    solvable by any ODE solver — which is what makes variable-length, irregularly
    sampled series painless (challenge #1). But our controls are **step-like**, so we
    build $X$ as a **rectilinear** (piecewise-constant, instantaneous-jump) path and
    integrate over a strictly-increasing **path parameter** $s$ instead of real time.
    Over a jump, real time is constant yet $\mathrm{d}X\neq 0$, so the increment
    $\int f_\theta(z)\,\mathrm{d}X = f_\theta(z)\,\Delta X$ is still captured;
    integrating in real time would give that jump zero duration and silently drop it
    (challenge #2). Real time is carried as one channel of $X$ so the model stays
    time-aware.

    *Concept — why a CDE?* An ordinary neural ODE evolves autonomously,
    $\mathrm{d}z = f_\theta(z)\,\mathrm{d}t$; it cannot ingest an incoming data stream.
    A **controlled** DE replaces $\mathrm{d}t$ with $\mathrm{d}X$, so the *data itself*
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

    2. **Build the rectilinear control path** (`rectilinear_interpolation(idx, ys)`).
       From the knots we form the **staircase**: between observations the path holds its
       value, then jumps. This is $X$. *Concept:* the control path is the continuous
       object the CDE is driven by, and its increments $\mathrm{d}X$ are what enter the
       integral.

    3. **Integrate over a path parameter, not time** (`s = jnp.arange(...)`,
       `LinearInterpolation(s, yr)`). Since staircase jumps take zero real time, we
       re-index the knots by a strictly increasing $s = 0,1,2,\dots$ and treat $X$ as a
       function of $s$. *Concept — reparametrisation invariance:* a CDE's output depends
       on the **geometry of the path**, not the speed it is traversed, so integrating in
       $s$ is legitimate — and it is what lets the solver see the jumps.

    4. **Initial state from the recipe** (`z0 = self.initial(static)`). The MLP
       $\zeta_\theta$ maps the `Z:` design scalars to $z(s_0)\in\mathbb{R}^{h}$.

    5. **Define the controlled dynamics** (`ControlTerm(self.func, control)`). `self.func`
       is $f_\theta$, an MLP returning an $h\times c$ matrix; the term encodes
       $\mathrm{d}z = f_\theta(z)\,\mathrm{d}X$. *Concept:* the update is a learned
       **matrix–vector product with the data increment**, not a fixed recurrent cell.

    6. **Solve** (`diffeqsolve(term, Heun(), stepsize_controller=StepTo(ts=s))`). We step
       exactly on the knot grid; Heun's method (2nd-order Runge–Kutta) advances
       $z_{n+1} \approx z_n + f_\theta(z_n)\,\big(X(s_{n+1}) - X(s_n)\big)$ using the
       solver's control increment $\Delta X$. Stepping on the knots guarantees every jump
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
    ### Result and takeaways

    On ~100 experiments the CDE lands **below** the baseline (holdout R² ≈ 0.65),
    exactly as expected — its value is methodological, showing the path we would scale
    up in a data-rich setting toward hybrid, mechanism-aware models.

    ---

    **Takeaways.** Clean, interpretable feature engineering + a well-regularised,
    honestly-benchmarked baseline gets us to R² ≈ 0.82. Performance was never the point
    of this challenge — clarity of the pipeline and of the decisions was.
    """)
    return


if __name__ == "__main__":
    app.run()
