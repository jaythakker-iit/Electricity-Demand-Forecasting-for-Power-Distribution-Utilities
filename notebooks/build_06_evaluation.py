"""Builds notebooks/06_evaluation.ipynb (after compare_all.py, run_phase7_experiments.py, run_phase7_figures.py)."""
from pathlib import Path

import nbformat as nbf

C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# 06 — Evaluation: which model is best, where, and is the difference real?
**EM 630 · Electricity Demand Forecasting · Phase 7**

A single MAE per model hides most of the story. This notebook asks:
1. **Is the ranking real?** Significance tests that respect time dependence.
2. **Where and when do models fail?** By hour, month, demand level, day type and peak hours.
3. **Is performance stable?** Month-by-month walk-forward, with retraining.
4. **What actually matters?** Feature-group ablation and SHAP.
5. **Is anything left to learn?** Residual autocorrelation.

Every number here comes from `results/tables/`, produced by `scripts/compare_all.py` and `scripts/run_phase7_experiments.py`.
""")
code(r"""
import sys, inspect, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
from src import config, compare as cp
pd.set_option("display.precision", 3)
TD = config.TABLE_DIR
def show(name, width=1000):
    display(Image(filename=str(config.FIG_DIR / f"{name}.png"), width=width))
print("Final contenders:", cp.FINAL)
""")

md(r"""
---
## 1. The unified table (`compare_all.csv`)
One scoring function, identical test rows for every model (enforced by a test).
""")
code(r"""
tab = pd.read_csv(TD / "compare_all.csv")
t = tab[(tab.split == "test") & (tab.subset == "all") & tab.final_contender]
t[["model", "family", "MAE", "RMSE", "MAPE", "R2", "Bias", "skill_vs_naive1", "skill_vs_step"]].sort_values("MAE").reset_index(drop=True)
""")

md(r"""
---
## 2. Is the ranking real?

💡 **Why ordinary tests fail here.** Forecast errors are autocorrelated: a model that is wrong at 09:00 is likely wrong at 10:00. Treating 4,762 hours as 4,762 independent observations overstates our certainty. We use two methods that respect the time dependence:

**Diebold–Mariano test.** With loss differential $d_t = |e^{(1)}_t| - |e^{(2)}_t|$, test $H_0: \mathbb{E}[d_t]=0$ using
$$DM = \frac{\bar d}{\sqrt{\widehat{\mathrm{LRV}}(d)/T}},$$
where the long-run variance uses a Newey–West (Bartlett) kernel with 24 lags, so the autocorrelation is included in the standard error. Holm's correction controls false positives across the 15 pairwise tests.

**Moving-block bootstrap.** Resample whole weeks (168-hour blocks) to get confidence intervals for MAE and for MAE differences.
""")
code(r"""
print(inspect.getsource(cp.diebold_mariano))
""")
md(r"""
**Checking the test before trusting it.** On simulated data where two models are *equally* good but have autocorrelated errors, a correct 5% test should reject about 5% of the time:
""")
code(r"""
from scipy import stats
rng = np.random.default_rng(0)
def ar(n, phi):
    z = rng.normal(size=n); x = np.empty(n); x[0] = z[0]
    for t_ in range(1, n): x[t_] = phi * x[t_ - 1] + z[t_]
    return x
rej_dm = rej_naive = 0; N = 200
for _ in range(N):
    c = ar(4762, 0.5) * 60
    e1, e2 = c + ar(4762, 0.5) * 20, c + ar(4762, 0.5) * 20
    rej_dm += cp.diebold_mariano(e1, e2)["p_value"] < 0.05
    d = np.abs(e1) - np.abs(e2)
    rej_naive += 2 * stats.t.sf(abs(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d) - 1) < 0.05
print(f"False-positive rate (target 5%):  DM with Newey-West {rej_dm / N:.1%}   |   naive t-test {rej_naive / N:.1%}")
""")
md(r"""
✅ The Newey–West DM test holds its 5% level; the naive test rejects true nulls more than twice as often.
""")
code(r"""
pd.read_csv(TD / "compare_ci.csv")
""")
code(r"""
show("F28_final_ranking_ci")
""")
code(r"""
dm = pd.read_csv(TD / "compare_dm.csv")
dm[["model_a", "model_b", "MAE_a_minus_b", "gap_ci_low", "gap_ci_high", "p_value", "p_holm", "significant_5pct"]]
""")
code(r"""
show("F29_dm_matrix", 850)
""")
md(r"""
✅ **Finding: a three-way tie.** Random Forest (51.9), XGBoost (51.7) and MLP v4 (52.1) are **statistically indistinguishable**: Holm-adjusted p = 1.0, and every bootstrap CI of their pairwise gaps contains 0. Every other step down the ranking is significant (p < 0.001): nonlinear > Lasso + h×m > step baseline > persistence.
The report must therefore say *"three models tie for best"*, not *"XGBoost wins"*.
""")

md(r"""
---
## 3. Where and when do models fail?
""")
code(r"""
show("F30_error_by_hour")
""")
code(r"""
show("F31_error_breakdowns", 1100)
display(pd.read_csv(TD / "breakdown_month.csv", index_col=0).round(1))
""")
md(r"""
✅ **Findings**
- **By hour:** all models are worst at the same hours (06:00, 09:00, 14:00), around the scheduled load steps and the morning ramp. The nonlinear models beat the step baseline at every hour.
- **By month:** November is easiest for every model. June has only 5 test days, so its value is unreliable.
- **By demand level:** for the nonlinear models, error rises with load (46 → 50 → 57 MW).
- **Festival days** are harder (+17% for XGBoost), but with only 336 test hours this is a noisy estimate. That matches the EDA expectation that the largest errors would come on festival days.
""")
code(r"""
show("F32_peak_hours", 1100)
pd.read_csv(TD / "peak_classification.csv")
""")
md(r"""
✅ **Peak hours** (≥ 6,137 MW, train 90th percentile, 588 test hours): the nonlinear models are as accurate on peaks as overall (≈ 53–54 MW) with only a small under-forecast (4–8 MW), and they flag peak hours with F1 ≈ 0.96. Lasso under-forecasts peaks the most among the learned models (+24.6 MW). For a utility, under-forecasting peaks is the costly direction, so this matters more than the overall MAE suggests.
""")

md(r"""
---
## 4. Walk-forward: stability and retraining

For each test month, every model is refit on **all data before that month** with its fixed Phase 4–6 hyperparameters, then scored on the month. This mimics a utility retraining its model monthly.
""")
code(r"""
show("F33_walkforward", 1100)
wf = pd.read_csv(TD / "phase7_walkforward.csv")
wf.pivot(index="month", columns="model", values="MAE_walkforward").round(1)
""")
md(r"""
✅ The ranking holds in every month. **Monthly retraining trims 1.6–2.5%** off the nonlinear models (XGBoost 51.7 → 50.9 MW), a cheap, practical improvement for deployment. The lookup baseline and Lasso barely change.
""")

md(r"""
---
## 5. What actually matters: feature-group ablation

XGBoost (Δ-target, fixed hyperparameters, mean of 3 seeds) is refit with feature groups added one at a time. DM tests check whether each change is real.
""")
code(r"""
display(pd.read_csv(TD / "phase7_ablation.csv").round(2))
pd.read_csv(TD / "phase7_ablation_dm.csv")
""")
code(r"""
show("F34_ablation")
""")
md(r"""
✅ **Findings**
- **Weather makes the model slightly worse**, on both Val (+0.29) and Test (+0.26, p = 0.03). Even *perfect* weather at the forecast hour, information no real forecaster has, does not help. This is the strongest confirmation of the EDA finding (Fig 5) that the weather columns carry no signal beyond the calendar.
- **Festival flag:** helps on Test (−0.26, p < 0.001) but not on Val (+0.09). Its value is uncertain, plausibly because the Val period (Mar–Jun) has few festivals.
- **Load + calendar alone (51.5 MW) matches the full model.**

⚠️ **Not acted on here, on purpose.** Switching the final model to "no weather" now would be a decision informed by the test set. It is recorded in `FEATURES.md` as a recommendation, to be confirmed by cross-validation in future work.
""")

md(r"""
---
## 6. Why does XGBoost predict what it does? (SHAP)

**SHAP values** split each prediction into additive contributions from each feature:
$$\hat d_h = \phi_0 + \sum_j \phi_j(x_h),$$
where $\hat d_h$ is the predicted **hourly change** (the Δ-target). The additivity check below confirms the decomposition is exact.
""")
code(r"""
sv = pd.read_csv(TD / "phase7_shap_values.csv", index_col=0)
pd.read_csv(TD / "phase7_shap_importance.csv", index_col=0).head(12).T
""")
code(r"""
show("F35_shap_summary", 950)
""")
md(r"""
✅ **Reading the SHAP plot**
- **Momentum dominates:** a rising load in the last hour (high `load_ramp_1`, red) pushes the predicted change up, and a falling load pushes it down.
- **Hour features** carry the scheduled steps from Phase 3 (the 09:00 jump and the night drops).
- **`load_lag_3`: high values push the forecast down**, a mean-reversion effect: if load was high three hours ago, the current rise is likely to fade.
- Weather (`temp_roll_mean_24`) contributes about 7 MW on average, mostly acting as a seasonal proxy.
""")

md(r"""
---
## 7. Is anything left to learn?
""")
code(r"""
show("F36_week_and_residual_acf", 1100)
""")
md(r"""
✅ Compared with the step baseline, the best model removes most of the autocorrelation in the errors (lag 1: 0.48 → 0.16). What remains peaks at **lag 24 (0.22)**: a model that errs at 09:00 today tends to err at 09:00 tomorrow. That is a concrete lead for future work, such as feeding yesterday's forecast error at the same hour back in as a feature.

---
## 8. Summary for the report

| Question | Answer |
|---|---|
| Best model? | **Three-way tie**: Random Forest 51.9, XGBoost 51.7, MLP v4 52.1 MW (MAPE ≈ 1.2%); differences not significant |
| Improvement over baselines | 30% vs the step baseline (74.3), 75% vs persistence (204.7); all p < 0.001 |
| Improvement over best linear model | 20% (64.9 → ≈ 52); p < 0.001 |
| Hardest conditions | Scheduled-step hours (06, 09, 14 h), high-demand hours, festival days |
| Peaks | Accurate (≈ 53 MW) with small under-forecast; F1 ≈ 0.96 for flagging peaks |
| Monthly retraining | Helps the nonlinear models by 1.6–2.5% |
| Weather features | Slightly harmful, even "perfect" weather; data limitation |
| Remaining structure | Errors repeat at the same hour on consecutive days (lag-24 ACF 0.22) |
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("06_evaluation.ipynb")
nbf.write(nb, out)
print("wrote", out)
