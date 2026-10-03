"""Builds notebooks/02_baselines.ipynb.
Run:  python notebooks/build_02_baselines.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/02_baselines.ipynb
"""
from pathlib import Path

import nbformat as nbf

C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# 02 — Baselines: the bar every model must beat
**EM 630 · Electricity Demand Forecasting · Phase 3**

**Why baselines?** A forecasting model is only useful if it beats what you could do *without* a model. Load is extremely persistent (Fig 8: autocorrelation at lag 1 = 0.98), so even "next hour = this hour" looks impressive on paper. Without a baseline, an R² of 0.99 tells you nothing.

We build five baselines, none of which involves machine learning:

| Baseline | Forecast for target hour $h$ | Parameters learned |
|---|---|---|
| **Naive-1** (persistence) | $\hat y_h = y_{h-1}$ | 0 |
| **Seasonal naive-24** | $\hat y_h = y_{h-24}$ | 0 |
| **Seasonal naive-168** | $\hat y_h = y_{h-168}$ | 0 |
| **Calendar climatology** | mean load for the same (hour, weekday, month) | 2,016 cell means |
| **Naive-1 + hourly step** | $\hat y_h = y_{h-1} + \overline{\Delta}_{hour(h),\,month(h)}$ | 288 average steps |

All "learned" quantities are fitted on data **strictly before** the period being forecast: Train for scoring Val, Train+Val for scoring Test.
""")

code(r"""
import sys, inspect, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
from src import baselines, config, data, evaluation as ev, features, split
from src.metrics import evaluate
pd.set_option("display.precision", 2)

def show(name, width=900):
    display(Image(filename=str(config.FIG_DIR / f"{name}.png"), width=width))

hourly = data.load_clean()
feats = features.build_features(hourly)
sp = split.chronological_split(feats)
T = config.TARGET
print(f"Val : {sp.val.index.min():%d %b %Y} → {sp.val.index.max():%d %b %Y}  ({len(sp.val):,} h)")
print(f"Test: {sp.test.index.min():%d %b %Y} → {sp.test.index.max():%d %b %Y}  ({len(sp.test):,} h)")
""")

md(r"""
---
## 1. How we score: metrics and the skill score

| Metric | Formula | Reads as |
|---|---|---|
| MAE | $\frac1n\sum|y-\hat y|$ | average miss in MW (our headline metric) |
| RMSE | $\sqrt{\frac1n\sum(y-\hat y)^2}$ | penalises big misses more; RMSE ≫ MAE means occasional large errors |
| MAPE | $\frac{100}{n}\sum\left|\frac{y-\hat y}{y}\right|$ | average miss in %; safe here because load never approaches 0 |
| R² | $1-\frac{\sum(y-\hat y)^2}{\sum(y-\bar y)^2}$ | share of variance explained |
| Bias | $\frac1n\sum(y-\hat y)$ | > 0 means systematic under-forecasting |
| **Skill** | $1-\mathrm{MAE}_{model}/\mathrm{MAE}_{naive\text{-}1}$ | > 0 means the model beats persistence; this is the honest metric |

One definition is shared by every model:
""")
code(r"""
print(inspect.getsource(evaluate))
""")

md(r"""
---
## 2. The three naive forecasts and the v1 off-by-one bug

Because Phase 1 names every feature **relative to the target hour $h$**, a naive forecast is just a column lookup:
""")
code(r"""
print(baselines.NAIVE)
# Independent check: recompute from the raw hourly series instead of the feature table
for name, k in [("Naive-1 (persistence)", 1), ("Seasonal naive-24", 24), ("Seasonal naive-168", 168)]:
    same = np.allclose(baselines.naive_forecasts(sp.test)[name], hourly["load_mw"].shift(k).reindex(sp.test.index))
    print(f"{name:24s} equals y(h-{k}) from raw series: {same}")
""")
md(r"""
**What went wrong in v1.** v1 put features on the row of the *current* hour $t$ but predicted the *next* hour $t+1$. So its column `Lag_24 = MW.shift(24)` was the load at $t-24 = (t+1)-25$. Its "same hour yesterday" was really **one hour off**, and so was its "same hour last week". Reproducing that:
""")
code(r"""
pd.read_csv(config.TABLE_DIR / "phase3_v1_offbyone.csv")
""")
md(r"""
✅ v1 overstated the seasonal-naive error by about **34%** (daily) and about **15%** (weekly), which made its ML models look better *relative to* those baselines than they were. With our naming convention this bug cannot happen.
""")

md(r"""
---
## 3. Calendar climatology: "what's normal for this hour?"
""")
code(r"""
print(inspect.getsource(baselines.CalendarClimatology))
""")
md(r"""
💡 Climatology has **no memory**: it predicts the same value for every Tuesday 10:00 in July, whatever happened yesterday. It cannot follow a heatwave or a year-to-year level change. Watch its **bias** in the results: it is large and positive (it under-forecasts), because the test period sits above the average level of past years.
""")

md(r"""
---
## 4. A discovery: load steps on a schedule

Persistence (naive-1) error is not smooth across the day (Fig 13): it spikes at fixed hours. Let's look at the average change from one hour to the next, by hour of day:
""")
code(r"""
def step_table(part):
    d = part[T] - part["load_lag_1"]
    return d.groupby(part["hour"]).agg(["mean", "std"])

st = pd.concat({"train": step_table(sp.train), "test": step_table(sp.test)}, axis=1).round(0)
st.T
""")
md(r"""
✅ **Finding.** At 09:00, load jumps by about **+410 MW in Train and +460 MW in Test**, with a spread (std) of only about 170 MW. Similar fixed steps occur at 00:00–02:00 (down, about −250 MW each hour), 06:00 (up, about +300 MW) and 23:00 (down). The steps are **large, regular and stable from Train to Test**. This pattern is consistent with scheduled load switching, such as rotating feeder supply schedules.

💡 **Why this matters.** Persistence ignores the clock, so it misses every scheduled jump by its full size. Adding the *average* step for that hour (and month, since the schedule shifts with the season) fixes most of that. It is still not machine learning, just a lookup table of 24 × 12 numbers:
""")
code(r"""
print(inspect.getsource(baselines.PersistencePlusStep))
""")

md(r"""
---
## 5. Results
""")
code(r"""
tab = pd.read_csv(config.TABLE_DIR / "phase3_baselines.csv")
cols = ["model", "MAE", "RMSE", "MAPE", "R2", "Bias", "skill_vs_naive1"]
for s in ["val", "test"]:
    print(f"\n{s.upper()} — all hours")
    display(tab[(tab.split == s) & (tab.subset == "all")][cols].sort_values("MAE").reset_index(drop=True))
print("TEST — peak hours (load ≥ train 90th percentile)")
display(tab[(tab.split == "test") & (tab.subset == "peak")][["model", "n", "MAE", "MAPE", "Bias"]].sort_values("MAE").reset_index(drop=True))
""")
code(r"""
show("F12_baselines_mae")
""")
code(r"""
show("F13_baselines_by_hour")
""")
code(r"""
show("F14_baselines_week", 1000)
""")
md(r"""
✅ **Findings**
1. **Naive-1 + hourly step is by far the strongest baseline:** test MAE ≈ 74 MW (1.8%), 64% better than persistence, and also the best on peak hours.
2. **Persistence beats both seasonal naives**, and naive-24 beats naive-168. Both match the EDA expectations (Fig 8: lag 1 > lag 24 > lag 168).
3. **R² is misleading here.** Even plain persistence scores R² ≈ 0.96, so any model looks excellent on R². That is why we report MAE and **skill vs persistence**.
4. Climatology under-forecasts (Bias > 0) because it can't track the current level, which is the point of memory-based features.

⚠️ **The evaluator's most important note.** v1's **tuned XGBoost** reported **MAE 74.4 MW, MAPE 1.75%**, essentially *the same* as this 288-number lookup table. On roughly the same test period, v1's best model did not clearly beat a simple non-ML baseline. In the report, ML only counts as successful if it beats **74 MW**, not 205 MW.
""")

md(r"""
---
## 6. Is anything left for ML to learn?
If the step baseline's errors were pure noise, no model could do better. We test this by looking for structure in its residuals:
""")
code(r"""
from statsmodels.tsa.stattools import acf
m = baselines.PersistencePlusStep().fit(sp.dev)
resid = sp.test[T] - m.predict(sp.test)
full = resid.reindex(pd.date_range(resid.index.min(), resid.index.max(), freq="h")).interpolate()
r_acf = acf(full, nlags=24, fft=True)
print(f"Residual autocorrelation: lag1 = {r_acf[1]:.2f}, lag2 = {r_acf[2]:.2f}, lag24 = {r_acf[24]:.2f}")
pd.Series({c: np.corrcoef(sp.test[c], resid)[0, 1] for c in
           ["load_ramp_1", "load_lag_1", "load_lag_24", "temp_lag_1", "is_festival"]}).round(3).to_frame("corr with residual").T
""")
md(r"""
✅ **Finding.** Yes. The residuals are **autocorrelated** (lag 1 ≈ 0.48, lag 24 ≈ 0.39) and correlate with recent **momentum** (`load_ramp_1`). Errors come in runs, so recent history still holds information the lookup table ignores. Weather and festival flags show almost nothing, consistent with the EDA.

🔧 **What the ML models must do (Phases 4–6):** learn (1) the scheduled steps, and they get these from the hour features, plus (2) the momentum and the persistence of deviations that the lookup table misses.

| Bar | Test MAE | Meaning |
|---|---|---|
| Naive-1 (persistence) | ≈ 205 MW | the minimum; any model must beat this |
| **Naive-1 + hourly step** | **≈ 74 MW** | **the real bar**: an ML model must beat this to justify itself |
| v1 tuned XGBoost (for reference) | 74.4 MW | did not clear the real bar |
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("02_baselines.ipynb")
nbf.write(nb, out)
print("wrote", out)
