"""Builds notebooks/04_tree_models.ipynb.
Run after scripts/run_phase5_trees.py, run_phase5_refine.py, run_phase5_figures.py:
  python notebooks/build_04_tree_models.py
  jupyter nbconvert --to notebook --execute --inplace notebooks/04_tree_models.ipynb
"""
from pathlib import Path

import nbformat as nbf

C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# 04 — Tree models: Decision Tree, Random Forest, XGBoost
**EM 630 · Electricity Demand Forecasting · Phase 5**

**Question.** Can non-linear models beat the best linear model (Lasso + hour×month, **64.9 MW**) and the baseline bar (**74.3 MW**)?

**Why trees?** Phase 4 needed 253 hand-made hour×month interaction columns before a linear model could follow the seasonal change in daily shape. A tree finds interactions **by itself**: a split on `month` followed by a split on `hour` *is* an interaction.

**Expectations written down before running**
1. A single tree overfits easily and will not beat the linear model.
2. Random Forest (bagging) beats a single tree by reducing variance.
3. XGBoost (boosting) is the best tree model and beats 64.9 MW.
""")
code(r"""
import sys, inspect, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
from src import config, data, features, split, tree_models as tm
pd.set_option("display.precision", 3)
def show(name, width=950):
    display(Image(filename=str(config.FIG_DIR / f"{name}.png"), width=width))
sp = split.chronological_split(features.build_features(data.load_clean()))
F = features.FEATURE_SETS["tree"]
print(f"{len(F)} tree features:", F)
""")
md(r"""
Trees use the **raw** calendar integers (`hour`, `month`, …): a tree splits on thresholds such as `hour ≤ 8.5`, so an ordinal integer is perfectly usable and no one-hot encoding or scaling is needed. They also keep `load_ramp_1`, the exact duplicate that had to be removed for linear models: trees don't invert $X^\top X$, so collinearity doesn't break them, and a ready-made difference saves them several splits. (See `FEATURES.md`.)
""")

md(r"""
---
## 1. The three algorithms

| | How it fits | Main hyperparameters | Course link |
|---|---|---|---|
| **Decision tree** | Greedy recursive splits; each split picks the feature/threshold that most reduces squared error | `max_depth`, `min_samples_leaf` | complexity control, bias–variance |
| **Random Forest** | Average of many deep trees, each on a bootstrap sample with a random subset of features per split (**bagging**) | `max_features`, `min_samples_leaf`, depth | averaging de-correlated estimators reduces **variance** |
| **XGBoost** | Trees added one at a time; tree $m$ fits the **negative gradient** of the loss (for squared error, the current residuals): $F_m(x)=F_{m-1}(x)+\eta\, f_m(x)$ | learning rate $\eta$, number of trees, depth, `reg_lambda` (L2), `reg_alpha` (L1), subsampling | **gradient descent in function space**; L1/L2 penalties on leaf weights = Ridge/Lasso ideas |
""")

md(r"""
---
## 2. A design choice made by cross-validation: predict the *level* or the *change*?

A tree's prediction is an **average of training targets** in a leaf, so a "level" tree can never forecast above the highest load it saw in training. Predicting the **hourly change** $d_h = y_h - y_{h-1}$ and adding back `load_lag_1` avoids that ceiling, and each leaf then holds a small, stationary quantity. Rather than choosing by hand, we made `target ∈ {level, delta}` **a hyperparameter**, chosen by CV:
""")
code(r"""
print(inspect.getsource(tm.TargetMode))
""")
md(r"""
A test confirms the ceiling (`test_level_tree_cannot_exceed_training_max`): a level tree fed an artificial heatwave never predicts above the training maximum.
*Honesty note:* in **this** test period the load never exceeds the training maximum (7,610 vs 8,632 MW), so the ceiling is **not** why "delta" wins here. The gain comes from the finer resolution of predicting small steps (Fig 22).
""")

md(r"""
---
## 3. Hyperparameter tuning

Randomised search with the same forward-chaining 5-fold CV as Phase 4, scored by MAE. **Stage 1** searches a broad grid. Where stage 1 chose a value at the **edge** of its grid, **stage 2** searches past that edge. A stage-2 model replaces stage 1 only if its CV error is lower. The test set is never consulted. Stopping rule, fixed in advance: one refinement stage only.
""")
code(r"""
for n, g in tm.SEARCH.items():
    print(n, {k.replace('regressor__', ''): v for k, v in g.items()})
""")
code(r"""
# stage-2 grids (searched past the stage-1 edges); see scripts/run_phase5_refine.py
src = open("../scripts/run_phase5_refine.py").read()
print(src[src.index("STAGE2 = {"):src.index("best1 =")].split("# keep stage-1")[0])
pd.read_csv(config.TABLE_DIR / "phase5_stage2_log.csv")
""")
code(r"""
pd.read_csv(config.TABLE_DIR / "phase5_best_params.csv").set_index(["model", "split"]).T
""")

md(r"""
---
## 4. One tree: watching it overfit
""")
code(r"""
show("F20_tree_depth_sweep")
""")
md(r"""
💡 **Bias–variance for trees.** Shallow trees underfit (high bias): they can't represent the hour × season pattern. Deep trees memorise the training data (training error keeps falling) but generalise worse (validation error rises). Depth plays the same role for trees that α played for Ridge: it controls complexity.
""")

md(r"""
---
## 5. Boosting = gradient descent
""")
code(r"""
show("F21_boosting_curve")
""")
md(r"""
💡 Each boosting round is one gradient step, taken in the space of functions rather than parameters. The **learning rate η** is the step size, exactly as in gradient descent on weights. A small η needs many rounds but reaches a lower validation error; a large η descends fast and plateaus higher. The **number of trees** acts like the number of epochs: past the validation minimum, extra rounds only fit noise (the training curve keeps falling, the validation curve doesn't).
""")

md(r"""
---
## 6. Results on the test set
""")
code(r"""
tab = pd.read_csv(config.TABLE_DIR / "phase5_trees.csv")
cols = ["model", "MAE", "RMSE", "MAPE", "R2", "Bias", "skill_vs_naive1", "skill_vs_step", "skill_vs_lasso"]
tab[(tab.split == "test") & (tab.subset == "all")][cols].sort_values("MAE").reset_index(drop=True)
""")
code(r"""
show("F22_level_vs_delta")
""")
code(r"""
show("F24_results_so_far")
""")
md(r"""
### Peak hours
""")
code(r"""
tab[(tab.split == "test") & (tab.subset == "peak")][["model", "n", "MAE", "MAPE", "Bias"]].sort_values("MAE").reset_index(drop=True)
""")

md(r"""
---
## 7. What drives the predictions
""")
code(r"""
show("F23_xgb_importance", 800)
""")
code(r"""
imp = pd.read_csv(config.TABLE_DIR / "phase5_importance.csv", index_col=0)
grp = lambda c: "Load history" if c in features.LOAD else "Weather" if c in features.WEATHER else "Calendar"
imp.groupby(imp.index.map(grp)).sum().round(3)
""")
md(r"""
💡 **Gain importance** = the total reduction in loss contributed by splits on a feature. It describes *the model*, not causation, and it is spread arbitrarily among correlated features (e.g. the hour encodings). SHAP values in Phase 7 give a per-prediction view.
""")

md(r"""
---
## 8. Summary

| Expectation (written before running) | Result |
|---|---|
| 1. A single tree overfits and won't beat the linear model | ✔ 67.8 vs 64.9 MW; depth sweep (Fig 20) shows the overfitting directly |
| 2. Random Forest beats a single tree (variance reduction) | ✔ 51.9 vs 67.8 MW (−23%) |
| 3. XGBoost is the best tree model and beats 64.9 MW | ◐ beats it (51.7 MW, −20%) but only **ties** Random Forest (0.2 MW apart); significance tested in Phase 7 |

**Further findings**
- **Predicting the change beats predicting the level** for every tree model (XGBoost 64.6 → 51.7 MW), chosen by CV, not by looking at the test set.
- **Peak hours:** the linear model under-forecast peaks by about 25 MW on average; the tree ensembles cut that bias to 4–8 MW, and Random Forest has the lowest peak MAE (53.0 MW).
- **Weather still contributes little** (4–6% of gain in all three models), consistent with the EDA.
- **A lesson about tuning.** Stage 2 improved XGBoost's CV error by 1.4% but its test error got 1.0% *worse*. We still keep the stage-2 model, because choosing by test score would be leakage. Beyond a point, extra tuning chases noise in the CV folds; hence the one-stage stopping rule.

**New reference for Phase 6 (MLP): ≈ 51.7 MW.**
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("04_tree_models.ipynb")
nbf.write(nb, out)
print("wrote", out)
