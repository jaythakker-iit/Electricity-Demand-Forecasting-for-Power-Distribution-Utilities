"""Phase 7: compare_all — the single source of truth for every number in the report.

Outputs (results/tables/):
  compare_all.csv            every model x split x subset (all / peak): MAE RMSE MAPE R2 Bias + skills
  compare_ci.csv             test MAE with 95% moving-block bootstrap CI (final contenders)
  compare_dm.csv             pairwise Diebold-Mariano tests (Holm-adjusted) + bootstrap CI of the MAE gap
  breakdown_*.csv            MAE by hour / month / demand level / day type / peak (test)
  peak_classification.csv    can each model flag peak hours (>= train P90)?
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import compare as cp  # noqa: E402
from src import config, data, evaluation as ev, features, split  # noqa: E402

T = config.TARGET
sp = split.chronological_split(features.build_features(data.load_clean()))
allp = ev.load_all_predictions()
thr = ev.peak_threshold(sp.train[T])

# ---------------------------------------------------------------- 1. unified table
table = ev.add_skill(ev.score(allp, thr))
idx = table.set_index(["split", "subset"]).index
for col, ref in [("skill_vs_step", "Naive-1 + hourly step"), ("skill_vs_xgb", "XGBoost")]:
    table[col] = 1 - table["MAE"] / idx.map(table[table.model == ref].set_index(["split", "subset"])["MAE"])
table["family"] = table.model.map(lambda m: __import__("src.plotstyle", fromlist=["family"]).family(m))
table["final_contender"] = table.model.isin(cp.FINAL)
table.round(4).to_csv(config.TABLE_DIR / "compare_all.csv", index=False)

# ---------------------------------------------------------------- 2. bootstrap CIs
w = cp.wide(allp, "test")
err = w[cp.FINAL].sub(w["y_true"], axis=0)
ae = err.abs()
boot = cp.block_bootstrap(ae, block=168, n_boot=2000)
ci = pd.DataFrame({"model": cp.FINAL, "MAE": ae.mean().values,
                   "ci_low": np.percentile(boot, 2.5, 0), "ci_high": np.percentile(boot, 97.5, 0)})
ci.round(3).to_csv(config.TABLE_DIR / "compare_ci.csv", index=False)

# ---------------------------------------------------------------- 3. pairwise DM + bootstrap gap
rows = []
col = {m: i for i, m in enumerate(cp.FINAL)}
for a, b in itertools.combinations(cp.FINAL, 2):
    r = cp.diebold_mariano(err[a].values, err[b].values, lags=24)
    gap = boot[:, col[a]] - boot[:, col[b]]
    rows.append({"model_a": a, "model_b": b, "MAE_a": ae[a].mean(), "MAE_b": ae[b].mean(),
                 "MAE_a_minus_b": r["mean_diff"], "gap_ci_low": np.percentile(gap, 2.5),
                 "gap_ci_high": np.percentile(gap, 97.5), "dm_stat": r["dm_stat"], "p_value": r["p_value"]})
dm = pd.DataFrame(rows)
dm["p_holm"] = cp.holm(dm["p_value"]).values
dm["significant_5pct"] = dm["p_holm"] < 0.05
dm.round(5).to_csv(config.TABLE_DIR / "compare_dm.csv", index=False)

# ---------------------------------------------------------------- 4. breakdowns (test)
X = sp.test
dtr = pd.qcut(sp.train[T], 3, retbins=True)[1]              # demand-level cut points from TRAIN
level = pd.cut(X[T], [-np.inf, dtr[1], dtr[2], np.inf], labels=["Low", "Medium", "High"])
daytype = pd.Series(np.select([X.is_festival == 1, X.is_weekend == 1], ["Festival", "Weekend"], "Working day"), index=X.index)
month = pd.Series(X.index.strftime("%Y-%m"), index=X.index)
for name, by in [("hour", X.hour), ("month", month), ("level", level.astype(str)), ("daytype", daytype),
                 ("peak", pd.Series(np.where(X[T] >= thr, "Peak (≥ train P90)", "Non-peak"), index=X.index))]:
    cp.breakdown(w, by).round(2).to_csv(config.TABLE_DIR / f"breakdown_{name}.csv")
# bias per group for peaks
bias_peak = err.groupby(np.where(X[T].reindex(w.index) >= thr, "Peak", "Non-peak")).mean().mul(-1)   # y - yhat
bias_peak.round(2).to_csv(config.TABLE_DIR / "breakdown_peak_bias.csv")

# ---------------------------------------------------------------- 5. peak classification (threshold from TRAIN)
pc = []
actual = w["y_true"] >= thr
for m in cp.FINAL:
    pred = w[m] >= thr
    tp, fp, fn, tn = (actual & pred).sum(), (~actual & pred).sum(), (actual & ~pred).sum(), (~actual & ~pred).sum()
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    pc.append({"model": m, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": prec, "recall": rec,
               "F1": 2 * prec * rec / (prec + rec)})
pd.DataFrame(pc).round(4).to_csv(config.TABLE_DIR / "peak_classification.csv", index=False)

print(ci.round(2).to_string(index=False))
print(dm[["model_a", "model_b", "MAE_a_minus_b", "gap_ci_low", "gap_ci_high", "p_value", "p_holm", "significant_5pct"]]
      .round(4).to_string(index=False))
print(pd.DataFrame(pc).round(3).to_string(index=False))
