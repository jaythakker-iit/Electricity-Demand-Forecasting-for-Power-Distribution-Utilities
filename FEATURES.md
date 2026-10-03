# Feature Register

This is the single list of every feature: what is used, what was left out, and **why**. Use it to bring features back after the main pipeline is finished. Every entry says how to re-include it and what to check when you do.

The feature definitions live in `src/features.py`. All features are named relative to the **target hour h** (`load_lag_1` = load at h−1).

---

## 1. In use

| Group | Features | Tree models (28) | Linear / MLP (47) |
|---|---|---|---|
| Load history | `load_lag_1, _2, _3, _24, _48, _168`, `load_roll_mean_24`, `load_roll_std_24`, `load_roll_mean_168` | ✔ | ✔ |
| Load momentum | `load_ramp_1` (= lag1 − lag2) | ✔ | — (see 2a) |
| Calendar (raw integers) | `hour, dow, month, dayofyear` | ✔ | — (a line through 0…23 is meaningless) |
| Calendar (cyclic) | `hour_sin/cos, dow_sin/cos` | ✔ | — (see 2a) |
| Calendar (cyclic) | `month_sin/cos` | ✔ | ✔ |
| Calendar (one-hot) | `hour_1…hour_23`, `dow_1…dow_6` | — (trees use raw ints) | ✔ |
| Day type | `is_festival` | ✔ | ✔ |
| Day type | `is_weekend` | ✔ | — (see 2a) |
| Weather at h−1 | `temp_lag_1, humidity_lag_1, temp_roll_mean_24`, `weather_lag_1_{cloudy, rain, storm}` | ✔ | ✔ |

---

## 2. Left out, and how to bring each back

### 2a. Removed from the linear/MLP set because they are exact duplicates (no information lost)

These are exact linear combinations of other columns in the linear set. Keeping them makes the design matrix rank-deficient (cond ≈ 10¹⁶), so OLS has no unique solution. The information is **still present** through the equivalent columns. *Found in Phase 2b (VIF check).*

| Feature | Exactly equal to | To re-include |
|---|---|---|
| `load_ramp_1` | `load_lag_1 − load_lag_2` | Don't, for linear models; it adds nothing. |
| `is_weekend` | `dow_5 + dow_6` | Swap it **in place of** `dow_1…dow_6` (a simpler weekday model). |
| `hour_sin`, `hour_cos` | a fixed weighting of `hour_1…hour_23` | Swap **in place of** the hour one-hots (2 smooth columns instead of 23 flexible ones). EDA Fig 2 suggests one-hot fits better. |
| `dow_sin`, `dow_cos` | a fixed weighting of `dow_1…dow_6` | Same idea as hour. |

✅ The check `test_linear_feature_set_is_full_rank` fails if a duplicate creeps back in.

### 2b. Held out on purpose (realism / leakage)

| Feature | Why held out | Where it is used |
|---|---|---|
| `fx_temp`, `fx_humidity`, `fx_weather_*` (weather **at** hour h) | Not known in real time at forecast time; it would be "perfect weather forecast" information | Phase 7 what-if ablation ("D_+weather_at_h"). Re-include only if the report states the assumption that a weather forecast is available. |

### 2c. Raw information not turned into features (yet)

| Raw column / idea | Current handling | Why | To re-include |
|---|---|---|---|
| `festival_name` (e.g. Diwali, Holi) | Collapsed to the binary `is_festival` | Each festival covers only ≈ 72 hours in 3 years, too sparse for one column per festival | Add one-hot for the 3–5 biggest festivals, or a festival-group category. Check the effect on **festival hours only** (Phase 7). |
| `holiday_type = "Weekend"` | Not used | Same information as `is_weekend` / `dow` | — |
| Weather label "Clear" | Reference level (dropped dummy) | Avoids the dummy-variable trap | — |
| Days before/after a festival | Not built | Possible pre/post-festival effect | `days_to_festival`, `is_festival_eve`. Must use the known calendar only. |
| Hour × month interaction | **Built in Phase 4** (`features.linear_design(df, interactions=True)`: +11 month one-hots, +253 hour×month dummies, 309 columns) | The daily shape flips with season (EDA Fig 3) | Already in use for the best linear model (64.9 MW vs 92.4 without). Trees and the MLP learn interactions on their own. |
| Cooling degree-hours, temp² | Not built | EDA showed temperature has no effect beyond the calendar (r ≈ 0.000) | Only worth trying if better weather data becomes available. |
| Short rolling windows (3 h, 6 h), rolling max/min | Not built | Lags 1–3 already cover the last few hours | Add to `features.py`, then re-run the full-rank test and the Phase 7 ablation. |

### 2d. Rows (not columns) excluded

| Rows | Count | Why |
|---|---|---|
| Imputed target hours | 455 | Never score a model on a value we made up |
| Warm-up (first 168 h) | 168 | `load_lag_168` / `load_roll_mean_168` undefined |
| ±3σ "outliers" | **0 removed** (11 hours kept) | They are real heatwave peaks (EDA Fig 11) |

---

## 2e. Evidence from the Phase 7 ablation (XGBoost, Δ-target, 3 seeds)

| Change | Val MAE | Test MAE | Test DM p | Recommendation |
|---|---|---|---|---|
| + weather at h−1 (`temp_lag_1`, `humidity_lag_1`, `temp_roll_mean_24`, `weather_lag_1_*`) | +0.29 | +0.26 | 0.03 | **Candidate for removal**: hurts on both splits |
| + `is_festival` | +0.09 | −0.26 | < 0.001 | Keep for now; benefit uncertain (Val has few festivals) |
| + weather at h (`fx_*`, what-if) | +0.08 | +0.34 | < 0.001 | Don't add: even perfect weather doesn't help |

Load + calendar alone (21 features) matches the full 28-feature model on test (51.5 MW). **Not applied to the final models**, because that would be a test-informed change. To act on it, re-run the Phase 5–6 tuning with weather removed and decide on **CV / Val**, then report the test score once.

## 3. Checklist when re-including a feature

1. Add it in `src/features.py` (named relative to h, built with `shift(≥1)` for anything measured).
2. `pytest -q`. The leakage tests (future perturbation, gap imputation) and the full-rank test must still pass.
3. Re-run the phases from 4 onwards, plus the Phase 7 ablation, and compare against the 74.3 MW bar.
4. Log the result in `results/evaluator_log.md`.
