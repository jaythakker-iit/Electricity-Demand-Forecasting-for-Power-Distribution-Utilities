# Evaluator Log

Each phase is checked against its gate before the next one starts. A gate either PASSES or the issue gets fixed and re-checked.

---

## Phase 0 — Roadmap & scaffold — PASS (2026-10-03)

- [x] Repo layout created (`src/`, `scripts/`, `tests/`, `results/`, `reports/`, `app/`, `legacy/`)
- [x] `src/config.py` is the single place for paths, split ratios and the seed
- [x] v1 script kept untouched in `legacy/` for traceability

## Phase 1 — Data, features, split — PASS after 2 fixes (2026-10-03)

| Check | Result |
|---|---|
| All 23,977 timestamps parse with the explicit day-first format | PASS (0 unparsed, 0 duplicates) |
| Regular hourly grid (24,432 slots, 455 missing hours in 241 gaps) | PASS |
| Observed values never modified by imputation | PASS (test) |
| Imputed hours never used as targets | PASS (test) — 455 rows excluded |
| `load_lag_k` equals actual load at *h−k* (v1 off-by-one fixed) | PASS (test) |
| Perturbing the future leaves past features unchanged | PASS (test) |
| No leakage through imputation next to a gap (short **and** long gaps) | PASS (test) |
| Main feature sets exclude weather at hour *h* | PASS (test) |
| Split strictly ordered: Train < Val < Test, Test = 20% | PASS (test) |

**Issues found and fixed during the gate**

1. **Data loss (13%).** Leaving long gaps as NaN deleted 3,270 rows, because each gap wiped out the next 168 hours of lag-168/rolling-168 features.
   *Fix:* fill long gaps with the same hour of the previous week, used as feature history only and flagged `is_imputed`.
   *Result:* rows dropped fell from 3,270 to 623 (168 warm-up + 455 imputed targets).
2. **Look-ahead leakage in short-gap interpolation.** Linear interpolation uses the value after the gap, and that value is the next row's target, so it leaked into `load_lag_1`.
   *Fix:* causal forward-fill.
   *Proof:* a dedicated test fails on the old method and passes on the new one.

**Observations to carry forward**

- Mean load rises from Train (4,327 MW) to Val (4,932 MW, which covers Mar–Jun, the hot season). ~~There is a growth trend plus seasonality~~ *(corrected in Phase 2: the gap is seasonal; the 2024 step-up is already inside Train)*. In Phase 5 we still need to check that trees don't fail above their training range.
- The Test period (Jun 2025 – Jan 2026) covers monsoon, post-monsoon and winter, so it is a demanding test.
- The weather label switches hour to hour (Clear → Rain → Cloudy). Expect a weak weather contribution; Phase 2 will quantify it.

## Phase 2 — EDA — PASS after 9 fixes (2026-10-03)

| Check | Result |
|---|---|
| 11 figures, each answering one question and tied to a modelling decision | PASS (`phase2_eda_findings.csv`, test) |
| Decision-driving statistics (ACF, correlations, adjusted effects) computed on Train only | PASS |
| Every number in a caption generated from data, not typed by hand | PASS |
| Every caption claim checked against the data | PASS after corrections below |
| Visual check of every figure (collisions, clipping, legends, axes) | PASS after corrections below |

**Claims the evaluator rejected and corrected**

1. "No year-on-year growth" was **false**. 2024 ran +11.2% above 2023, then 2025 was flat (−0.7%). It is now described as a step up followed by a plateau. This also corrects the Phase 1 note: the Train→Val jump in the mean is seasonal (Val = hot season), not a trend.
2. "Stronger ACF peak at 168 h" was **false**. Lag 168 (0.899) ≈ lag 144 (0.898). The weekly cycle is weak and the daily cycle dominates.
3. The "1,900 MW hump" in the histogram was unexplained. It turned out to be winter nights (Nov–Mar, 01–04 h), a distinct low regime.
4. The "outliers are summer peaks" claim was verified: all 11 hours fall on 18–19 Jun 2024 and 12–13 Jun 2025.

**Visual defects fixed:** split labels overlapping the subtitle (F01); a "23–23:00" typo (F03); temperature panels sharing the humidity x-axis (F05); sparse noisy bins (F05, now n ≥ 100); value labels colliding with error bars (F06, F07); a one-colour legend (F09); grid lines drawn over bars (all bar charts); subtitles overflowing (now auto-wrapped).

**Robustness check on the key finding (weather has no independent effect)**

| Adjustment | r(temp, residual) | r(humidity, residual) |
|---|---|---|
| hour × weekday × month | +0.000 | −0.002 |
| hour × month | −0.0002 | −0.0015 |
| Daily means, within the same month | +0.034 | — |

The conclusion is robust. In this dataset, temperature and humidity carry no information beyond time of day and season. Hotter days within a month are not higher-load days, which is unusual for real Indian grid data; it suggests the weather columns may be synthetic or derived. **This goes in the report as a data limitation.**

**Key modelling implications carried forward**

- Hour × month interaction is strong (winter peak at 10:00, Apr–May at 15:00, Jun–Sep at 22–23:00), so expect non-linear models to beat linear ones clearly.
- Load lags are highly collinear, which motivates Ridge/Lasso.
- Expect roughly zero gain from weather and a small gain from festivals (−101 MW, −2.3% on festival hours).

## Phase 2b — EDA notebook + feature-set fix — PASS (2026-10-03)

**Added:** `notebooks/01_EDA.ipynb`, the narrative version of the EDA. Each section runs Question → why it matters → analysis code → figure → finding → decision, linked to course concepts (confounding, autocorrelation, interaction effects, multicollinearity, L1/L2, convexity). Plotting code was moved into `src/eda.py` (one function per figure), which both the notebook and `scripts/run_phase2_eda.py` call. The refactor was verified to produce identical findings.

**Defect found while building the notebook (a Phase 1 feature-design error)**

| Problem | Evidence | Fix |
|---|---|---|
| `load_ramp_1 = load_lag_1 − load_lag_2` exactly | cond(X) = 1.0e16, VIF ≈ 1e15 | Kept for trees; removed from the linear/MLP set |
| `is_weekend = dow_5 + dow_6` exactly | New full-rank test: rank 47 of 52 | Removed from the linear set (weekday one-hots cover it) |
| hour/dow sin & cos are linear combinations of the hour/dow one-hots (given an intercept) | Same test | Linear set uses one-hot hour/dow + sin/cos month |

**Result:** the linear/MLP feature set went from 53 to 47 columns and is now full rank (`test_linear_feature_set_is_full_rank`). The load-feature condition number fell from 1.0e16 to 45.2. The remaining VIFs (174, 90, 58, …) are genuine time-series collinearity, which is the motivation for Ridge/Lasso in Phase 4.

**Why it matters:** without this fix, the plain OLS baseline in Phase 4 would have had non-unique coefficients, and any coefficient interpretation would have been meaningless.

Tests: 14/14 pass.

## Phase 3 — Baselines — PASS (2026-10-03)

| Check | Result |
|---|---|
| Naive forecasts recomputed independently from the raw series (`y.shift(k)`) | PASS (test) |
| Learned baselines (climatology, step) fitted strictly before the scored period | PASS (assert + test); fitting on Train only gives the same test MAE (74.34 vs 74.31) |
| Metric function verified on hand-computed values | PASS (test) |
| Report CSV reproducible from saved predictions; every model scored on identical rows | PASS (tests) |
| Peak threshold from Train P90 (6,137 MW), not the test set | PASS |
| v1 off-by-one quantified | v1 overstated naive-24 MAE by **+33.8%** and naive-168 by **+15.3%** |
| EDA expectations | ✔ persistence best of the naives; ✔ naive-24 beats naive-168 |

**Test results (n = 4,762 h, Jun 2025 – Jan 2026)**

| Baseline | MAE | MAPE | R² | Skill vs persistence |
|---|---|---|---|---|
| **Naive-1 + hourly step** | **74.3** | **1.77%** | 0.994 | **+0.64** |
| Naive-1 (persistence) | 204.7 | 5.05% | 0.959 | 0 |
| Seasonal naive-24 | 247.1 | 5.33% | 0.919 | −0.21 |
| Calendar climatology | 396.8 | 8.89% | 0.851 | −0.94 |
| Seasonal naive-168 | 413.9 | 8.99% | 0.794 | −1.02 |

**Key finding (raised by the evaluator while reviewing F13).** Persistence error spikes at fixed hours because load **steps on a schedule**: 09:00 +412 MW in Train and +459 MW in Test (std ≈ 170). A non-ML lookup baseline (persistence + the average step per hour × month, 288 numbers) reaches **74.3 MW**, the same as v1's tuned XGBoost (74.4 MW). **Consequence:** the real bar for Phases 4–6 is **74 MW**, not 205 MW. The baseline's residuals still contain structure (ACF lag 1 = 0.48, lag 24 = 0.39; r = 0.24 with `load_ramp_1`), so ML has room to improve, but it has to earn it.

**Fixes during the gate:** legend overlapping bars (F12) and lines (F13); inconsistent model colours across figures, now a project-wide `MODEL_COLORS` registry; convoluted v1-bug table code simplified; one notebook claim corrected (night drops span 00:00–02:00).

**Added:** `notebooks/02_baselines.ipynb` (explained walkthrough), `src/baselines.py`, `src/evaluation.py` (shared long-format predictions + scoring, used by every later phase). Tests: 20/20 pass.

## Phase 4 — Linear models (OLS / Ridge / Lasso) — PASS after 4 fixes (2026-10-03)

| Check | Result |
|---|---|
| sklearn OLS ≡ (X'X)⁻¹X'y and Ridge ≡ (X'X+αI)⁻¹X'y | PASS (max diff ~1e-10; tests) |
| Scaler inside the pipeline, fitted on training rows only | PASS (test) |
| α chosen by forward-chaining 5-fold CV; folds never look backwards | PASS (test) |
| Chosen α is interior, or on a flat plateau at the grid edge | PASS after widening the grids (test) |
| Lasso 1-SE model sparser than best-α model | PASS (43/47 vs 47/47; 182/309 vs 307/309) |
| Same test rows as the baselines; table reproducible from predictions | PASS (tests) |
| Both new design matrices full rank (47, 309) | PASS |

**Test results**

| Model | MAE | MAPE | Skill vs persistence | Skill vs bar (74.3) |
|---|---|---|---|---|
| **Lasso + hour×month** | **64.9** | **1.56%** | +0.68 | **+0.13** |
| OLS / Ridge + hour×month | 65.3 | 1.57% | +0.68 | +0.12 |
| OLS / Ridge / Lasso (47 features) | 92.4 | 2.16% | +0.55 | −0.24 |

**Issues found and fixed**

1. **α at the grid edge.** The first run picked the smallest α tried (Ridge 0.01, Lasso 0.13). The grids were widened to 1e-4…1e5 for Ridge and α_max·1e-6 for Lasso, and a boundary diagnostic plus test were added. Remaining edge picks sit on a provably flat CV curve.
2. **Flawed stability test.** The first coefficient-stability check used CV folds of *growing* size and wrongly showed Ridge as less stable. Replaced by equal 30-day windows, which show the bias–variance trade-off (F17): coefficient spread −44%, next-window MAE 102 → 161 MW as α rises 0 → 100.
3. **False caption (F16).** "Weather enters late" was wrong. `temp_lag_1` enters 4th, before any hour dummy, peaks at +22, then flips to −9 once the calendar enters. This is the Fig 5 confounding, seen inside the model.
4. **Label collision (F16).** α-line labels overlapped the coefficient labels.

**Expectations vs outcomes:** Ridge ≈ OLS ✔ (tied within 0.01 MW). Lasso zeroes weather ◐ (only at the 1-SE α, with a sign flip; at best α it is kept but ≤ 9 MW/SD). Interactions needed ✔ (92.4 → 64.9 MW).

**Carried forward:**
- **New reference for Phases 5–6: 64.9 MW.** The best linear model already beats v1's tuned XGBoost (74.4).
- **Peak hours:** the best linear model has MAE 62.7 vs 66.3 for the bar, but bias **+24.6 MW** (under-forecasts peaks) vs +8.6 for the bar. Investigate in Phase 7.
- Feature engineering mattered more than the estimator: interactions −30% MAE, OLS→Ridge→Lasso < 1%.

Tests: 28/28 pass.

## Phase 5 — Tree models (Decision Tree, Random Forest, XGBoost) — PASS after 6 fixes (2026-10-03)

| Check | Result |
|---|---|
| `delta` target wrapper reproduces lag_1 + f(X) exactly | PASS (test) |
| A level tree cannot exceed the training maximum (why `delta` exists) | PASS (test) |
| Saved models (refit on Train+Val) reproduce the saved test predictions exactly | PASS (test) |
| Tuning by forward-chaining 5-fold CV, scored by MAE; target mode tuned as a hyperparameter | PASS |
| Same test rows as every other model; table reproducible from predictions | PASS (tests) |
| Stage-1 edge picks searched past in stage 2; one-stage stopping rule fixed in advance | PASS |

**Test results**

| Model | MAE | MAPE | vs bar 74.3 | vs best linear 64.9 | Peak MAE / bias |
|---|---|---|---|---|---|
| **XGBoost** (delta) | **51.7** | **1.20%** | +30% | +20% | 53.7 / +8.0 |
| Random Forest (delta) | 51.9 | 1.21% | +30% | +20% | **53.0 / +4.4** |
| Decision Tree (delta) | 67.8 | 1.59% | +9% | −5% | 72.6 / +2.3 |

Diagnostic (not used for selection), level target: XGBoost 64.6, RF 71.7, DT 95.7. CV chose `delta` for all three.

**Issues found and fixed**

1. **Edge-of-grid picks.** Stage 1 chose 5 boundary values for XGBoost and 2 for RF. Stage 2 searched past them: RF's optimum is now interior (`max_features` 0.33 beats 0.15–0.25; leaf = 1 is the natural floor). XGBoost CV improved 77.17 → 76.06. The one remaining edge (learning rate 0.01) is documented, not chased, under the stopping rule.
2. **Stage 2 vs test (lesson recorded, not "fixed").** Stage 2 improved CV by 1.4% while test MAE went 51.21 → 51.73. The CV-selected model is kept; switching on test would be leakage.
3. **Crash in stage-2 bookkeeping** (string written to an int column). Fixed; stage-1 results were backed up first (`*_stage1.csv`).
4. **False caption (F20).** "Validation error rises" did not hold with min_samples_leaf = 10; the error plateaus. The sweep was redone for leaf = 1 vs 10, which shows the genuine U-shape (CV minimum 96.6 MW at depth 12, training error → 0.4) and the leaf-size cap.
5. **Overclaiming caption (F23).** "Hour features do most of the work" corrected: load history 52% (led by `load_ramp_1`, 19%), hour encodings 30%, weather 6%.
6. **Figure defects.** Annotations over curves and a wrong "overshoot" mechanism (F21; η = 0.1 actually overfits); overlapping labels and a reused colour (F24). Comparison bars are now coloured by model family (`plotstyle.FAMILY_COLORS`).

**Expectations:** 1 ✔ (a single tree loses to linear), 2 ✔ (RF −23% vs tree), 3 ◐ (XGBoost beats linear but only ties RF, 0.2 MW apart; significance in Phase 7).
**Honesty note:** the test maximum (7,610 MW) is below the training maximum (8,632), so the tree "ceiling" is *not* why `delta` wins here; the gain comes from predicting small stationary steps.

**Carried forward:** the reference for Phase 6 (MLP) is **≈ 51.7 MW**. Tree ensembles fixed the linear model's peak under-forecast (bias 24.6 → 4–8 MW).

Tests: 34/34 pass.

## Phase 6 — PyTorch MLP (EnergyMLP) — PASS after 3 investigations + 2 fixes (2026-10-03)

| Check | Result |
|---|---|
| Architecture = brief (Linear→BatchNorm→ReLU→Dropout, MSE, Adam wd 1e-4, ReduceLROnPlateau, early stopping) | PASS (test) |
| Same seed → identical predictions; different seed → different | PASS (test); deterministic flag removed after this was proven (2.5× faster) |
| Feature/target scalers fitted on the fitting rows only | PASS (test) |
| Early stopping restores the best epoch; recorded schedule length = best epoch | PASS (test) |
| Δ-target adds back lag-1 exactly | PASS (test) |
| Configuration chosen on Val, not test | PASS (test): v4 selected |
| 3 seeds per configuration; reported model = 3-seed ensemble; same test rows as all models | PASS |

**Test results (3-seed ensembles)**

| Model | Val MAE | Test MAE | Seed spread (test) | Peak bias |
|---|---|---|---|---|
| **MLP v4 [64,32] Δ-target (selected)** | **60.1** | **52.1** | **±0.4** | **+4.8** |
| MLP v1 [64,32] | 67.4 | 56.0 | ±4.1 | +28.8 |
| MLP v2 [256,128,64] + dropout 0.3 | 69.1 | 60.6 | ±4.9 | +73.3 |
| MLP v3b (diagnostic: patient scheduler) | 76.2 | 62.1 | ±4.3 | +42.7 |
| MLP v3 lr 1e-4, batch 512 (brief) | 94.2 | 86.6 | ±8.9 | +100.8 |
| *XGBoost / Random Forest (reference)* | 59.1 / 58.8 | 51.7 / 51.9 | — | +8.0 / +4.4 |

**Investigations (evaluator follow-ups)**

1. **v3 starved by the scheduler?** Yes, mostly. ReduceLROnPlateau halved lr 1e-4 six times, to ~1e-6. The v3b diagnostic (patience 15/40) recovers Val 97.2 → 78.9, but it still trails v1 (71.3). Both effects are reported; v3 itself is kept exactly as the brief specifies.
2. **Is the refit the source of the seed variance?** Yes. Train-only vs refit on test: v1 60.0 ± 2.0 vs 62.4 ± 4.1; v4 52.6 ± 0.9 vs 53.3 ± 0.4. The refit replays the epoch count but can't restore the best epoch. The protocol is **kept** (all models are treated identically; changing it after seeing test scores would be leakage) and the ≈ 0.7 MW cost to v4 is documented.
3. **"Train loss > val loss because of dropout": verified, not assumed.** v1 training loss is 0.0138 with dropout on vs 0.0033 with it off (val 0.0088).

**Fixes:** (a) `use_deterministic_algorithms(True)` made training 2.5× slower; reproducibility was proven without it and a test now guards it. (b) Crowded log-axis labels in F25.

**Expectations:** 1 ✘ (v2's capacity didn't help: 60.6 vs 56.0), 2 ◐ (slower ✔, but worse; partly the scheduler), 3 ✔ (Δ-target: 56.0 → 52.1, seed spread ±4.1 → ±0.4).

**Key finding for the report:** three unrelated model families (RF 51.9, XGBoost 51.7, MLP 52.1) converge on ≈ 52 MW once they predict the hourly change. That suggests a floor set by the information in the features rather than by the algorithm. Whether the 0.4 MW differences are significant is tested in Phase 7 (Diebold–Mariano).

Tests: 41/41 pass.

## Phase 7 — Unified evaluation & error analysis — PASS after 6 caption/figure fixes (2026-10-03)

| Check | Result |
|---|---|
| `compare_all.csv` reproducible from saved predictions; all models scored on identical test rows | PASS (tests) |
| DM test calibrated under autocorrelated errors (simulation) | PASS: 4.5–5.5% false positives at 5% (a naive t-test gives 12–14%); power 97% for a 3% MAE gap |
| Holm correction for 15 pairwise tests; moving-block (168 h) bootstrap CIs | PASS (tests) |
| Ablation "full set" reproduces the Phase 5 XGBoost (51.50 vs 51.73, seed averaging) | PASS (test) |
| Walk-forward uses only data before each month | PASS (test) |
| SHAP additivity (base + Σ SHAP = model output) | PASS (max error 0.001 MW) |
| Palette validated (dataviz script); CVD warning → direct labels + marker shapes on every line chart | PASS |

**Headline results (test, 4,762 h)**

| | MAE (95% block-bootstrap CI) |
|---|---|
| XGBoost | 51.7 [48.1, 54.4] |
| Random Forest | 51.9 [48.3, 55.0] |
| MLP v4 Δ-target | 52.1 [48.1, 55.9] |
| Lasso + hour×month | 64.9 [61.0, 68.6] |
| Step baseline | 74.3 [70.3, 78.2] |
| Persistence | 204.7 [185.3, 218.4] |

- **Three-way tie:** RF vs XGBoost vs MLP have Holm p = 1.0, and all gap CIs contain 0. Every other adjacent step is significant (p < 0.001).
- **Breakdowns:** hardest hours 06, 09 and 14 h (scheduled steps / ramp); November easiest for all models. Nonlinear error rises with demand (46 → 50 → 57 MW). Festival hours +17% (n = 336, noisy).
- **Peaks** (≥ 6,137 MW, n = 588): nonlinear MAE ≈ 53–54, bias +4 to +8 MW, F1 ≈ 0.96. Lasso bias +24.6.
- **Walk-forward:** the ranking is stable every month. Monthly retraining improves the nonlinear models by 1.6–2.5% (XGBoost 51.7 → 50.9).
- **Ablation:** weather **hurts** on Val (+0.29) and Test (+0.26, p = 0.03), even perfect weather at h (+0.34, p = 0.0002). The festival flag helps on Test (−0.26, p < 0.001) but not on Val (+0.09). Load + calendar alone = full model (51.5).
- **SHAP:** momentum (`load_ramp_1`, 75 MW mean |SHAP|), then the hour encodings (scheduled steps); `load_lag_3` acts as mean reversion.
- **Residuals:** ACF at lag 1 0.48 → 0.16 vs the step baseline. The remaining peak is at **lag 24 (0.22)**, a future-work lead.

**Fixes during the gate**

1. **F34 misleading:** bars measured against set A made "+festival" look harmful. Redrawn as incremental steps with DM p-values.
2. **F31 overclaims:** "error grows with demand" holds only for the nonlinear models; "October hardest" only for baseline/Lasso (nonlinear: July/January); "monsoon onset" was speculation, removed; June flagged as only 5 test days.
3. **F36 incomplete:** the caption missed that the residual structure peaks at lag 24.
4. **F28/F33/F31 cosmetics:** broken marker, stray log-axis label, overlapping label, over-long caption.

**Not acted on (by design):** dropping weather would be test-informed now. It's recorded in `FEATURES.md` as a recommendation to confirm by CV.

Tests: 49/49 pass.
