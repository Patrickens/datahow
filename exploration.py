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
    import json
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return Path, json, mo, pd


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
    ### The math: what XGBoost is doing

    XGBoost is a **gradient-boosted tree model**. It builds the prediction as a sum of
    many small regression trees:

    $$
    \hat{u}_i = F(x_i) = \sum_{k=1}^{K} f_k(x_i),
    \qquad f_k \in \mathcal{F}.
    $$

    Here:

    - $x_i$ is the feature vector for experiment $i$,
    - $u_i = \log(1 + y_i)$ is the transformed titer target,
    - $\hat{u}_i$ is the model prediction on that log scale,
    - $f_k$ is one regression tree,
    - $\mathcal{F}$ is the space of possible CART-style regression trees.

    The model is fitted **stage-wise**. At boosting round $m$, we already have a model
    $F_{m-1}$ and add one new tree:

    $$
    F_m(x_i) = F_{m-1}(x_i) + \eta f_m(x_i),
    $$

    where $\eta$ is the learning rate. The new tree is chosen to reduce the training
    loss, while also being penalised for being too complex:

    $$
    \mathcal{L}^{(m)}
    =
    \sum_i \ell\big(u_i, F_{m-1}(x_i) + f_m(x_i)\big)
    +
    \Omega(f_m),
    $$

    with the tree penalty

    $$
    \Omega(f_m)
    =
    \gamma T
    +
    \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2.
    $$

    Here:

    - $T$ is the number of leaves in the new tree,
    - $w_j$ is the prediction value assigned to leaf $j$,
    - $\gamma$ penalises adding extra leaves,
    - $\lambda$ shrinks leaf values toward zero.

    To decide what tree to add, XGBoost approximates the loss locally using a
    second-order Taylor expansion around the current prediction
    $\hat{u}_i^{(m-1)} = F_{m-1}(x_i)$:

    $$
    \ell\big(u_i, \hat{u}_i^{(m-1)} + f_m(x_i)\big)
    \approx
    \ell\big(u_i, \hat{u}_i^{(m-1)}\big)
    +
    g_i f_m(x_i)
    +
    \frac{1}{2} h_i f_m(x_i)^2,
    $$

    where

    $$
    g_i =
    \frac{\partial \ell(u_i, \hat{u})}{\partial \hat{u}}
    \bigg|_{\hat{u}=\hat{u}_i^{(m-1)}},
    \qquad
    h_i =
    \frac{\partial^2 \ell(u_i, \hat{u})}{\partial \hat{u}^2}
    \bigg|_{\hat{u}=\hat{u}_i^{(m-1)}}.
    $$

    So $g_i$ tells us the local direction in which the prediction should move, and
    $h_i$ tells us how strongly curved the loss is around the current prediction.

    For squared-error loss on the log target,

    $$
    \ell(u_i, \hat{u}_i) = \frac{1}{2}(\hat{u}_i - u_i)^2,
    $$

    these become especially simple:

    $$
    g_i = \hat{u}_i - u_i,
    \qquad
    h_i = 1.
    $$

    Now consider one leaf $j$ of a candidate tree. Let $I_j$ be the set of training
    examples that fall into that leaf. Define the summed gradient and Hessian in that
    leaf as

    $$
    G_j = \sum_{i \in I_j} g_i,
    \qquad
    H_j = \sum_{i \in I_j} h_i.
    $$

    For a fixed tree structure, the optimal value assigned to leaf $j$ is

    $$
    w_j^{*}
    =
    -\frac{G_j}{H_j + \lambda}.
    $$

    This says: if the summed gradient in a leaf is strongly positive, the model should
    decrease the prediction there; if it is strongly negative, the model should increase
    it. The $\lambda$ term prevents very large leaf corrections.

    To grow a tree, XGBoost asks whether splitting a leaf into a left and right child
    would reduce the objective. Let

    $$
    G_L = \sum_{i \in I_L} g_i,
    \qquad
    H_L = \sum_{i \in I_L} h_i,
    \qquad
    G_R = \sum_{i \in I_R} g_i,
    \qquad
    H_R = \sum_{i \in I_R} h_i.
    $$

    The split gain is

    $$
    \mathrm{gain}
    =
    \frac{1}{2}
    \left[
    \frac{G_L^2}{H_L + \lambda}
    +
    \frac{G_R^2}{H_R + \lambda}
    -
    \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
    \right]
    -
    \gamma.
    $$

    The first two terms measure how good the two child leaves would be after the
    split. The third term measures how good the original unsplit leaf was. The
    difference is the improvement from splitting, and $\gamma$ subtracts the cost of
    adding an extra leaf. A split is only worthwhile if this gain is positive.

    **Why this model here.** XGBoost is a strong baseline for this dataset because it
    works well with small tabular data, heterogeneous engineered features, nonlinear
    interactions, and missing values. We also regularise it using shallow trees,
    subsampling, and the $\lambda,\gamma$ penalties. Training on `log1p(titer)` makes the
    target less skewed and makes errors closer to relative errors on the original titer
    scale.
    """)
    return


@app.cell
def _(Path, json, mo):
    _metadata = json.loads(Path("artifacts/xgb_best_metadata.json").read_text())

    mo.md(
        f"""
        ### Small XGBoost sweep

        I sampled **{_metadata["n_configs"]}** shallow-tree configurations with a single
        fixed `seed={_metadata["seed"]}` (config sampling, CV splits and the estimator all
        derive from it). The table below is sorted by validation R² and shows the top
        configurations.
        """
    )
    return


@app.cell
def _(pd):
    _xgb_sweep = pd.read_csv("artifacts/xgb_sweep_results.csv")
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
def _(Path, json, mo):
    _metadata = json.loads(Path("artifacts/xgb_best_metadata.json").read_text())
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
    sweep configuration reaches repeated-CV **R² ≈ 0.84**, far above the mean
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
    our custom cell-population feature) and the **apparent glutamine consumed**
    (feed-accounting), followed by VCD shape/level descriptors (centroid, median, and the
    **area under the VCD curve** — essentially the **integral of viable cells**, the classical
    mechanistic predictor of product) and the lysed-fraction signal. The rest are
    substrate/byproduct AUCs, levels, and feed summaries. This is exactly the kind of feature
    catch22 could **not** provide — its top-ranked features were abstract dynamical
    descriptors, whereas here the model leans on quantities a process scientist would reach for.
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
    track **train and validation RMSE in raw titer units** (easier to read than
    standardised log-space MSE) plus validation R², as the **mean ± range over the
    sweep's 3 selection seeds** for the chosen config — a single seed's holdout is too
    noisy to read on its own.

    The dashed line is the **deployed selection metric, R² ≈ 0.84**: the mean over the 3
    seeds of *each seed's early-stopped best epoch*. The mean-R² **curve** peaks a little
    below that line because the seeds reach their best at different epochs, so no single
    epoch has all three at their peak at once — a normal consequence of multi-seed early
    stopping, not a discrepancy. The untrained epoch-0 point is omitted (random log-space
    predictions explode after `expm1` and would dominate the scale).
    """)
    return


@app.cell
def _(plotting):
    fig_curves = plotting.plot_cde_training_curves(epochs=200)
    fig_curves
    return


@app.cell
def _(Path, json, mo):
    _metadata = json.loads(Path("artifacts/cde_best_metadata.json").read_text())

    mo.md(
        f"""
        ### CDE sweep

        I sampled **{_metadata["n_configs"]}** CDE configurations and scored each across
        **{_metadata["n_seeds"]} seeds** (`seeds={_metadata["seeds"]}` — each drives its own
        20% validation split and model initialisation). Selection is on the **mean**
        validation R² across seeds, not a single lucky holdout; the table below is sorted by
        mean validation R² and reports its spread (±std).
        """
    )
    return


@app.cell
def _(pd):
    _cde_sweep = pd.read_csv("artifacts/cde_sweep_results.csv")
    _cde_sweep_display = (
        _cde_sweep.sort_values("val_r2_mean", ascending=False)
        .head(5)
        .loc[
            :,
            [
                "run_index",
                "lr",
                "hidden_size",
                "width",
                "depth",
                "batch_size",
                "val_rmse_mean",
                "val_mape_mean",
                "val_r2_mean",
                "val_r2_std",
            ],
        ]
        .round({"lr": 4, "val_rmse_mean": 0, "val_mape_mean": 3, "val_r2_mean": 3, "val_r2_std": 3})
    )
    _cde_sweep_display
    return


@app.cell
def _(Path, json, mo):
    _metadata = json.loads(Path("artifacts/cde_best_metadata.json").read_text())
    _best = _metadata["best_validation"]
    _cfg = _metadata["best_config"]

    mo.md(
        f"""
        The best configuration was run **{int(_best["run_index"])}**:
        `lr={_cfg["lr"]}`, `hidden_size={_cfg["hidden_size"]}`, `width={_cfg["width"]}`,
        `depth={_cfg["depth"]}`, `batch_size={_cfg["batch_size"]}`. The rest of the pipeline
        is fixed: 200 epochs, a warmup+cosine LR schedule with gradient clipping, an adaptive
        Tsit5 solver, train-only standardisation, and early stopping on raw-scale RMSE.
        """
    )
    return


@app.cell
def _(Path, json, mo):
    _metadata = json.loads(Path("artifacts/cde_best_metadata.json").read_text())
    _best = _metadata["best_validation"]

    mo.md(
        f"""
        ### Best CDE: result

        The selected CDE reached **validation RMSE ≈ {_best["val_rmse_mean"]:.0f}**,
        **MAPE ≈ {100 * _best["val_mape_mean"]:.1f}%**, and **R² ≈ {_best["val_r2_mean"]:.2f}
        ± {_best["val_r2_std"]:.2f}** (mean over {_best["n_seeds"]} seeds). An earlier
        version of this model sat near R² ≈ 0.75; switching to an adaptive Tsit5 solver,
        minibatch training with a cosine schedule, and honest train-only standardisation —
        i.e. fixing the *optimisation*, not the architecture — lifted it to the number above.

        This is now competitive with the XGBoost baseline (repeated-CV R² ≈ 0.84), though the
        protocols differ (3-seed holdout vs 5×5 repeated CV), so the comparison is indicative
        rather than exact. I still deploy XGBoost by default — it is faster, simpler to serve,
        and its repeated-CV estimate is more robust on ~100 experiments — but the CDE is no
        longer merely a methodological demo; it is a genuinely competitive path-based model.
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
      honestly-benchmarked XGBoost baseline reaches **R² ≈ 0.84** (5×5 repeated CV).
    - The neural CDE, after fixing the optimisation (adaptive solver, minibatching,
      cosine schedule) and the standardisation leakage, reaches **R² ≈ 0.84** on a
      3-seed holdout — competitive with the baseline, not just a methodological demo;
      **hybrid mechanistic models** remain the interpretable, data-efficient destination.
    - Performance was never the point of this challenge — clarity of the pipeline and of
      the decisions was.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Service architecture

    Part 2 is a thin, model-agnostic **FastAPI** service (`titer_prediction.service`)
    exposing the spec's `GET /health` and `POST /predict`. The guiding principle:
    **HTTP concerns never mix with model logic** — each layer has one job.

    - **`app.py`** — routes + exception handlers. Routes stay thin (validate,
      delegate, return). The model is loaded **once at startup** (`lifespan`) into
      `app.state` and injected via a `get_predictor` dependency, which is overridden
      in tests to swap in a mock.
    - **`dto.py`** — Pydantic v2 DTOs. `PredictRequest` mirrors the OpenAPI schema
      (`timestamps` + a `values` map of `Z:`/`W:`/`X:` variable → array) and enforces
      structure: finite, strictly-increasing timestamps; each channel length matching
      the timeline (`Z:` scalars may be length 1); at least one variable per prefix.
      `PredictResponse` returns the prediction plus provenance (`model_type`,
      `n_timepoints`, `experiment_id`).
    - **`predictor.py`** — the *only* place the API touches model data. It rebuilds a
      one-experiment DataFrame in **exactly the training shape** (`Exp`, `Time[day]`,
      `Z:/W:/X:`) and calls the model's `predict_frame`, which flows through the same
      `read_inputs` preprocessing as training — so serving cannot drift from training.
    - **`model_loader.py`** — dispatches on artifact suffix (`.joblib` → XGBoost,
      `.eqx` → CDE) behind a single `Predictor` `Protocol`. Model modules are imported
      **lazily**, so serving the tabular baseline never pulls in the JAX/diffrax stack.
    - **`config.py` / `errors.py`** — env-driven settings (`MODEL_PATH`) and the domain
      exceptions the handlers map to status codes.

    **Error semantics** (mapped centrally, so routes raise and never format HTTP): a
    missing or unloadable model → **503** (the app still boots; `/health` reports
    `model_loaded=false`); a payload that is structurally invalid (Pydantic) *or*
    semantically incomplete for the loaded model (`PayloadError`) → **400**, matching
    the OpenAPI spec (overriding FastAPI's default 422); anything unexpected → **500**.
    The service is **stateless** — one startup load, no per-request model IO.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Testing strategy

    A small **pyramid**, fast by default and honest about what it covers:

    - **Data & feature invariants** (`test_data_integrity.py`) — forward-filled
      statics, one feature row per experiment, trapezoidal/NaN-safe AUC,
      feed-accounting and cell-density features, ragged (unpadded) sequences, and the
      CDE's flat-tail padding + mixed path. A synthetic fixture exercises the logic;
      real-data tests skip when the confidential CSVs are absent.
    - **Sweep determinism** (`test_sweep.py`) — config sampling is reproducible under a
      fixed seed and rejects over-large requests, backing the reproducibility claims.
    - **API layer with a mock** (`test_service.py`) — endpoints tested via FastAPI
      `dependency_overrides` with a **mocked** predictor, covering validation,
      conversion and error-mapping *without* expensive inference: health, 503 without a
      model, `Z:`-scalar expansion, and rejection of bad timestamps/lengths/prefixes.
    - **Real-model integration** (`test_service_integration.py`) — loads the actual
      `xgb_best.joblib` and runs inference end-to-end over HTTP and in batch; skips
      cleanly when the artifact or test CSV is absent.
    - **Docker smoke** (`test_docker_smoke.py`, `-m docker`) — builds the image and
      asserts the full contract with and without the model mounted (200/400/503);
      pure-Python, auto-skips without a Docker daemon, so it runs on Windows and Linux CI.
    - **Lint/format** — `ruff check` + `ruff format --check`, bundled in `make check`.

    The split is deliberate: fast tests (mock + synthetic) run everywhere and pin
    behaviour; the slow, environment-dependent tests (real artifact, Docker) add
    end-to-end confidence but never block a fresh clone or CI without the data.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Reproducibility & the pipeline

    One path from CSVs to a served model, each stage its own module (see the repo
    layout): `data_preprocessing` → `features` (XGBoost) or ragged sequences (CDE) →
    `regression`/`cde` training → serialised **bundles** (`.joblib` / `.eqx`, each
    carrying its own preprocessing + config so serving needs no retraining) →
    reproducible `sweep`s that select and refit the best config.

    Reproducibility is explicit. A single `seed` drives every split, sample and
    initialisation; the XGBoost sweep selects on **5×5 repeated CV**, while the CDE
    sweep scores each config as the **mean over 3 seeds** (not one lucky holdout) and
    fits standardisation on the **train split only** to avoid validation leakage.
    `make models` rebuilds the artifacts, `make figures` the plots, and `make notebook`
    this document — all from the (git-ignored) challenge data.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Deployment & running the service

    The same app runs locally or in Docker, with the model **mounted at runtime**,
    not baked into the image — so one image serves either model family and switching
    is a config change, not a rebuild:

    ```bash
    make run-api        # serve xgb_best.joblib on :9000 (uvicorn)
    make api-health     # GET  /health   -> {"status":"ok","model_loaded":true}
    make api-predict    # POST /predict  (scripts/sample_payload.json) -> a titer

    MODEL_PATH=artifacts/cde_best.eqx make run-api   # serve the CDE instead

    # in Docker (model mounted, not baked in):
    make docker-build
    make docker-run          # foreground server on localhost:9000
    make docker-api-health   # from another terminal
    make docker-api-predict  # from another terminal
    ```

    `batch_predict` reuses the exact request-conversion path for CSV scoring
    (`make predict` → `artifacts/test_predictions.csv`), so online and batch inference
    share one code path. Model artifacts are **git-ignored and reproducible** from the
    challenge data, not committed to the repo.
    """)
    return


if __name__ == "__main__":
    app.run()
