"""Builds notebooks/03_ridge_lasso.ipynb.
Run:  python notebooks/build_03_ridge_lasso.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/03_ridge_lasso.ipynb
(Expects scripts/run_phase4_linear.py and run_phase4_figures.py to have been run.)
"""
from pathlib import Path

import nbformat as nbf

C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# 03 — Linear models: OLS, Ridge (L2) and Lasso (L1)
**EM 630 · Electricity Demand Forecasting · Phase 4**

**Question.** How far can a *linear* model go, and what does regularisation actually do on this data?

**What we already know**
- EDA: load features are strongly collinear (VIF up to 174, Fig 9); the daily shape **changes with season** (Fig 3); weather has no effect beyond the calendar (Fig 5).
- Baselines: the bar to beat is **74.3 MW** (persistence + average hourly step, Phase 3).

**Expectations written down before running (we check each one at the end)**
1. Ridge ≈ OLS in accuracy but with more stable coefficients.
2. Lasso sets the weather coefficients to zero.
3. The 47-feature linear model can't bend its daily shape by season, so it will struggle against the bar; adding **hour × month interactions** should fix that.
""")

code(r"""
import sys, inspect, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
from src import config, data, features, split, linear_models as lm
pd.set_option("display.precision", 3)
def show(name, width=950):
    display(Image(filename=str(config.FIG_DIR / f"{name}.png"), width=width))

sp = split.chronological_split(features.build_features(data.load_clean()))
T = config.TARGET
X_dev = features.linear_design(sp.dev, False)
print("Base design:", X_dev.shape, "| with hour×month:", features.linear_design(sp.dev, True).shape)
""")

md(r"""
---
## 1. Theory in one table

| | Objective | Solution | Convex? | Effect |
|---|---|---|---|---|
| **OLS** | $\min_\beta \|y-X\beta\|_2^2$ | $\hat\beta=(X^\top X)^{-1}X^\top y$ | yes | unbiased; high variance if $X^\top X$ is near-singular |
| **Ridge** | $\min_\beta \|y-X\beta\|_2^2+\alpha\|\beta\|_2^2$ | $\hat\beta=(X^\top X+\alpha I)^{-1}X^\top y$ | **strictly** (unique solution) | shrinks all coefficients smoothly; stabilises collinear ones |
| **Lasso** | $\min_\beta \frac1{2n}\|y-X\beta\|_2^2+\alpha\|\beta\|_1$ | no closed form → **coordinate descent** | yes | shrinks *and* sets some coefficients exactly to 0 (feature selection) |

**Why standardise first?** The penalty $\|\beta\|$ treats every coefficient equally. If one feature is in MW (thousands) and another is a 0/1 flag, the penalty would hit them unequally. `StandardScaler` puts all features on a common scale, and it sits **inside** the pipeline, so its mean and std come from the training rows only.
""")
code(r"""
print(inspect.getsource(lm.pipe)); print(inspect.getsource(lm.make_model))
""")
md(r"""
**Checking the theory against the code.** Our tests verify that sklearn's OLS and Ridge coefficients equal the closed-form formulas above to 6 decimal places:
""")
code(r"""
X = features.linear_design(sp.train, False).astype(float).values
Xs = (X - X.mean(0)) / X.std(0); y = sp.train[T].values; yc = y - y.mean()
for kind, a in [("ols", None), ("ridge", 50.0)]:
    m = lm.fit_final(features.linear_design(sp.train, False), sp.train[T], kind, a)
    A = Xs.T @ Xs + (0 if a is None else a) * np.eye(Xs.shape[1])
    b = np.linalg.solve(A, Xs.T @ yc)
    print(f"{kind:5s}: max |sklearn − closed form| = {np.abs(m.named_steps['model'].coef_ - b).max():.2e}")
""")

md(r"""
---
## 2. Choosing α with time-series cross-validation

💡 **Hyperparameter tuning without leakage.** α isn't learned by the optimiser; we pick it by **forward-chaining CV**. Each fold trains on the past and validates on the block right after it, never the other way round:

```
fold 1: [train]               [val]
fold 2: [train......]               [val]
...
fold 5: [train...........................]  [val]
```
Two rules are reported:
- **best α**: lowest mean CV error;
- **1-SE α**: the *most regularised* α whose error is within one standard error of the best, the simplest model that is statistically tied.
""")
code(r"""
print(inspect.getsource(lm.cv_curve)); print(inspect.getsource(lm.select_alpha))
""")
code(r"""
pd.read_csv(config.TABLE_DIR / "phase4_alphas.csv")
""")
code(r"""
show("F15_cv_alpha_curves")
""")
md(r"""
✅ **Finding.** The CV curves are **flat** for small α and only rise once α is large. For three models the "best" α is the smallest one tried, but the curve is flat there (the `curve_flat_there` column, also enforced by a test). The honest reading is that **regularisation does not improve accuracy here**.

💡 **Why? Sample size.** The variance of OLS shrinks like $\sigma^2/n$. With $n \approx 19{,}000$ rows and $p = 47$ (or 309) features, OLS variance is already small, so there is little left for shrinkage to remove, and any shrinkage adds bias. Regularisation pays off when $p$ is large relative to $n$. Section 4 shows this directly.
""")

md(r"""
---
## 3. Lasso: sparsity and the order features enter
""")
code(r"""
show("F16_lasso_path")
""")
code(r"""
c = pd.read_csv(config.TABLE_DIR / "phase4_coefficients_std.csv", index_col=0)
rows = []
for m in ["Lasso", "Lasso (1-SE α)", "Lasso + hour×month", "Lasso + hour×month (1-SE α)"]:
    s = c[m].dropna(); nz = s.abs() > 1e-9
    rows.append({"model": m, "non-zero": f"{nz.sum()}/{len(s)}",
                 "weather kept": ", ".join(w for w in features.WEATHER if nz.get(w, False)) or "none"})
pd.DataFrame(rows)
""")
code(r"""
c.loc[features.WEATHER, ["OLS", "Lasso", "Lasso (1-SE α)"]].round(2)
""")
md(r"""
✅ **Findings**
- The first features in (largest α) are `load_lag_1`, `load_lag_24` and `load_lag_168`: persistence, then the daily cycle, then the weekly cycle, the same ranking as the ACF in EDA Fig 8.
- **Expectation 2 only partly holds.** At the CV-best α, Lasso keeps all 6 weather features, but they are tiny (≤ 9 MW per SD, against 1,791 for `load_lag_1`). At the 1-SE α it drops two of them, and `temp_lag_1` **flips sign** (−9 → +22). An unstable sign is itself a sign of no real signal.
- **Confounding, seen inside the model:** `temp_lag_1` enters 4th, before any hour dummy, acting as a stand-in for time of day. When the hour dummies enter it shrinks and flips. That is exactly the EDA finding (Fig 5), that temperature only looked useful because it tracks the clock.
- `load_lag_1` and `load_lag_2` carry large **opposite-signed** coefficients (+1,791 / −828). That is the collinearity signature: together they encode "level + momentum".
""")

md(r"""
---
## 4. The bias–variance trade-off, measured

A first attempt compared coefficient spread across the 5 CV folds and found Ridge *less* stable than OLS. The evaluator rejected that test. CV folds have **growing** sizes, and a fixed α penalises small samples more, so the spread mixed shrinkage differences with variance. The fair test fits on **equal-size 30-day windows**:
""")
code(r"""
print(inspect.getsource(lm.window_bias_variance))
pd.read_csv(config.TABLE_DIR / "phase4_bias_variance.csv")
""")
code(r"""
show("F17_bias_variance")
""")
md(r"""
✅ **Finding: the textbook trade-off, in numbers.** As α grows, **variance falls** (coefficient spread down 44%) while **bias rises** (next-window MAE from 102 to 161 MW). On this data the bias wins: the true model needs a large `load_lag_1` weight, and shrinking it toward zero hurts.
**Expectation 1:** ✔ for stability, ✔ "Ridge ≈ OLS" at CV-best α (tied within 0.01 MW), ✘ if we had hoped Ridge would be *more accurate*.
""")

md(r"""
---
## 5. Results on the test set
""")
code(r"""
tab = pd.read_csv(config.TABLE_DIR / "phase4_linear.csv")
cols = ["model", "MAE", "RMSE", "MAPE", "R2", "Bias", "skill_vs_naive1", "skill_vs_step"]
tab[(tab.split == "test") & (tab.subset == "all")][cols].sort_values("MAE").reset_index(drop=True)
""")
code(r"""
show("F18_linear_results")
""")
md(r"""
✅ **Expectation 3 confirmed.** With 47 features, every linear model reaches only **92 MW**, worse than the 74 MW lookup table, because it applies the *same* hourly adjustment all year. Adding the hour×month interactions (309 features) brings it to **64.9 MW (Lasso), 13% better than the bar** and better than v1's tuned XGBoost (74.4 MW).
""")
code(r"""
show("F19_learned_hour_month", 800)
""")
md(r"""
💡 **What the interactions learned.** The 09:00 step (EDA, Phase 3) shows clearly in every month. The winter 05:00–06:00 bump and the summer late-evening block show that the hourly adjustment depends on the season, which a single additive hour effect can't represent.

### Peak hours
""")
code(r"""
tab[(tab.split == "test") & (tab.subset == "peak")][["model", "n", "MAE", "MAPE", "Bias"]].sort_values("MAE").reset_index(drop=True)
""")

md(r"""
---
## 6. Summary

| Expectation | Result |
|---|---|
| Ridge ≈ OLS in accuracy, more stable coefficients | ✔ tied at CV-best α; stability gain demonstrated on equal windows, but it costs accuracy (bias) |
| Lasso zeroes weather | ◐ only at the 1-SE α, and with a sign flip; at best α weather is kept but negligible |
| Linear needs hour×month to compete | ✔ 92.4 → 64.9 MW; beats the 74.3 MW bar by 13% |

**Takeaways for the report**
1. With $n \gg p$, regularisation isn't needed for **accuracy**; its value here is **interpretability** (Lasso path) and **stability**.
2. **Feature engineering beat algorithm choice:** adding the right interaction cut error by 30%, while switching OLS → Ridge → Lasso changed it by under 1%.
3. The best linear model (**64.9 MW**) is the new reference for Phases 5–6: XGBoost and the MLP must beat it to justify their extra complexity.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("03_ridge_lasso.ipynb")
nbf.write(nb, out)
print("wrote", out)
