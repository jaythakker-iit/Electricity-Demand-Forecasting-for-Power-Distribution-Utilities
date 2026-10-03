# Project Roadmap — Electricity Demand Forecasting (EM 630, IIT Gandhinagar)

**Goal:** Forecast hourly electricity demand (MW) **one hour ahead** from past load, calendar, weather and festival data. Compare regularised linear models, tree ensembles and a neural network against naive baselines. Ship the work as a reproducible repo, a report, a slide deck and a demo app.

**Scope (agreed):** a hybrid. We keep the existing Indian load dataset and the XGBoost work, and add Ridge/Lasso and a PyTorch MLP so the project covers the course syllabus: regularisation, convex optimisation, gradient descent and hyperparameter tuning.

**How we work:** one phase at a time. Each phase ends with an **Evaluator Gate**, a checklist that has to pass before the next phase starts. The gate result is recorded in `results/evaluator_log.md`.

---

## The pipeline (end-to-end)

```
data/raw/load_data.csv
   │  Phase 1  src/data.py      parse dates → hourly grid → gap handling → flags
   │           src/features.py  calendar + cyclic + lags + rolling + exogenous (all causal)
   │           src/split.py     chronological Train 70% | Val 10% | Test 20%
   ▼
data/processed/features.parquet  ──► tests/ (leakage + integrity checks)
   │  Phase 2  EDA figures + findings
   │  Phase 3  baselines      naive-1, naive-24, naive-168
   │  Phase 4  linear         OLS, RidgeCV, LassoCV (TimeSeriesSplit)
   │  Phase 5  trees          DT, RF, XGBoost (randomised search + early stopping)
   │  Phase 6  neural         EnergyMLP v1/v2/v3 (PyTorch, early stopping on Val)
   ▼
results/tables/*.csv, results/figures/*.png, results/models/*
   │  Phase 7  compare_all + error analysis + significance test
   ▼
Phase 8 report (.docx) · Phase 9 deck · Phase 10 Streamlit app · Phase 11 final QA
```

**Golden rules (every phase):**

1. **Causality.** A feature for target hour *h* may only use information available at *h−1*. Calendar facts about hour *h* (hour of day, holiday, festival) are allowed because they are known in advance.
2. **No shuffling.** Splits and cross-validation folds always run forward in time.
3. **Fit on train only.** Scalers, imputers and tuning never see the test set. The test set is scored **once**, at the end.
4. **One source of truth.** Every number in the report or deck comes from a CSV in `results/tables/`.
5. **Reproducible.** Fixed seeds, pinned requirements, and `python run_all.py` rebuilds everything.

---

## Phases

| # | Phase | Why we do it | Key outputs | Evaluator gate (must pass) |
|---|---|---|---|---|
| 0 | Roadmap & scaffold ✅ | A modular repo lets three members work in parallel and lets any result be regenerated | Folder layout, `config.py`, `requirements.txt` | Imports work; paths resolve |
| 1 | Data & features ✅ | Bad timestamps, gaps or leakage would invalidate every later number | `data.py`, `features.py`, `split.py`, processed dataset, tests | Every hour is unique and on the hourly grid; no imputed targets are scored; perturbing future values leaves past features unchanged; splits are strictly ordered |
| 2 | EDA ✅ | Shows *why* each feature exists (daily/weekly/seasonal cycles, temperature effect) | ~10 figures + a findings table | Every figure has an axis label, units and a one-line *specific* insight |
| 3 | Baselines ✅ | A model is only "good" relative to a trivial forecast. This phase also fixes the off-by-one bug in the old seasonal naive | `baselines.csv` | naive-24 uses the load at *h−24* (old code used *h−25*) |
| 4 | Linear models ✅ | Course core: L2 vs L1 regularisation, bias–variance, sparsity | OLS/Ridge/Lasso metrics, α-CV curve, Lasso path, coefficients | α chosen by time-series CV; scaler inside the pipeline; nonzero Lasso coefficients reported |
| 5 | Tree models ✅ | Non-linear interactions (hour × temperature × weekday) | DT/RF/XGB metrics, best params, importances | Tuning uses train only; search space wider than in v1; early stopping on Val |
| 6 | MLP ✅ | Course core: gradient descent, Adam, regularisation through dropout and weight decay, learning-rate scheduling | 3 experiments, loss curves, `mlp_*.pt` | Val loss curve plateaus; best epoch restored; results stable across seeds |
| 7 | Evaluation ✅ | Shows *where* and *when* the models fail, beyond one overall number | `compare_all.csv`, error by hour/weekday/level/peak, walk-forward, feature-group ablation, SHAP, Diebold–Mariano test | Same test set for every model; peak threshold taken from train; claims backed by tables |
| 8 | Report | The graded deliverable | `reports/Final_Report.docx` | Every number traceable to a CSV; sections: Abstract → Conclusion → References |
| 9 | Deck | Viva / presentation | Slide deck | Content matches the report; ~12–15 slides |
| 10 | Demo app | Shows a working product | `app/streamlit_app.py` | Runs locally; uses the saved models; no retraining in the app |
| 11 | Final QA | Proves the project is "top-notch" | Clean run of `run_all.py`, `pytest` all green | Fresh run reproduces the report numbers to 2 decimal places |

---

## The bar (set in Phase 3)

Every model in Phases 4–6 is judged by its **skill against the strongest baseline**, "Naive-1 + hourly step": **test MAE 74.3 MW, MAPE 1.77%**. Beating plain persistence (204.7 MW) is not enough. For reference, v1's tuned XGBoost (74.4 MW) did not clear this bar.

**Updated after Phase 4:** the best linear model, Lasso + hour×month, reaches **64.9 MW**. XGBoost (Phase 5) and the MLP (Phase 6) must beat **64.9 MW** to justify their extra complexity.

**Updated after Phase 5:** XGBoost **51.7 MW** and Random Forest **51.9 MW** (both predicting the hourly change). The MLP (Phase 6) is judged against **≈ 51.7 MW**.

**Updated after Phase 6:** MLP v4 (Δ-target, 3-seed ensemble) reaches **52.1 MW**. Three model families converge on ≈ 52 MW; Phase 7 tests whether the differences are significant.

**Final verdict (Phase 7):** Random Forest 51.9, XGBoost 51.7, MLP v4 52.1 MW form a **statistical three-way tie** (Diebold–Mariano, Holm p = 1.0). All are significantly better than Lasso + hour×month (64.9), the step baseline (74.3) and persistence (204.7). XGBoost is used as the demo-app model (smallest file, fastest, tied best).

## Known issues carried over from v1 (and where they get fixed)

| Issue in `legacy/electricity_demand_forecasting_v1.py` | Fixed in |
|---|---|
| Seasonal naive used `Lag_24` (load at *h−25*) instead of *h−24* | Phase 1 (feature naming relative to the target hour) + Phase 3 |
| Imputed (forward-filled) loads could become targets and be scored | Phase 1 (`is_imputed` flag, those targets dropped) |
| Weather forward-filled with no limit across long gaps | Phase 1 (limit = 3 h) |
| Weather at the target hour is not known in real time | Phase 1 (weather lagged by 1 h) + Phase 7 ("perfect weather forecast" what-if experiment) |
| Peak threshold taken from the **test** set in one analysis | Phase 7 (train-derived threshold everywhere) |
| XGBoost search tried 5 of 12 combinations | Phase 5 |
| Conclusion claimed "Pipeline prevents leakage" for all models | Phase 8 wording |
| Figures report had no methodology/discussion; Figures 9 and 14 were missing | Phase 8 |

## Team mapping (suggested)

- **Member 1:** Phases 1–3 (data, EDA, baselines)
- **Member 2:** Phases 4–5 (linear and tree models)
- **Member 3:** Phase 6 (MLP) and the app
- **All:** Phase 7 analysis, report, deck
