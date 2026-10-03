"""Phase 7 experiments (fixed hyperparameters from Phases 4-6; no new tuning).

E1 Walk-forward: for each calendar month of the test period, refit on ALL data before that month
   and forecast the month. Compared with the static models (fit once on Train+Val). Answers: does
   performance hold month by month, and does monthly retraining help?
E2 Feature-group ablation with XGBoost (delta target, 3 seeds): which feature groups matter?
   'C_+festival' is the full tree feature set = the Phase 5 model (consistency check).
E3 SHAP values for the Phase 5 XGBoost on 2,000 random test hours.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from src import baselines, config, data, evaluation as ev, features, split  # noqa: E402
from src import compare as cp  # noqa: E402
from src import linear_models as lm  # noqa: E402
from src import mlp  # noqa: E402
from src import tree_models as tm  # noqa: E402

T = config.TARGET
FT, FL = features.FEATURE_SETS["tree"], features.FEATURE_SETS["mlp"]
feats = features.build_features(data.load_clean())
sp = split.chronological_split(feats)
t0 = time.time()

best = pd.read_csv(config.TABLE_DIR / "phase5_best_params.csv")
alphas = pd.read_csv(config.TABLE_DIR / "phase4_alphas.csv")
meta6 = json.load(open(config.TABLE_DIR / "phase6_meta.json"))
hist6 = pd.read_csv(config.TABLE_DIR / "phase6_histories.csv")
seeds6 = pd.read_csv(config.TABLE_DIR / "phase6_seed_results.csv")
V4 = "MLP v4 [64,32] Δ-target"


def xgb_params():
    r = best[(best.model == "XGBoost") & (best.split == "test")].iloc[0]
    return dict(n_estimators=int(r.n_estimators), learning_rate=float(r.learning_rate), max_depth=int(r.max_depth),
                min_child_weight=float(r.min_child_weight), subsample=float(r.subsample),
                colsample_bytree=float(r.colsample_bytree), reg_lambda=float(r.reg_lambda), reg_alpha=float(r.reg_alpha))


def rf_params():
    r = best[(best.model == "Random Forest") & (best.split == "test")].iloc[0]
    md = None if str(r.max_depth) in ("None", "nan") else int(float(r.max_depth))
    return dict(n_estimators=200, max_depth=md, min_samples_leaf=int(float(r.min_samples_leaf)),
                max_features=float(r.max_features))


def make_xgb(seed=config.SEED):
    return tm.TargetMode(XGBRegressor(tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0, **xgb_params()), "delta")


def make_rf():
    return tm.TargetMode(RandomForestRegressor(n_jobs=-1, random_state=config.SEED, **rf_params()), "delta")


lasso_alpha = float(alphas[(alphas.model == "Lasso + hour×month") & (alphas.split == "test")].alpha_best.iloc[0])
v4cfg = mlp.MLPConfig(**{k: v for k, v in meta6["configs"][V4].items() if k in mlp.MLPConfig.__dataclass_fields__})
v4_sched = {s: hist6[(hist6.model == V4) & (hist6.seed == s)].sort_values("epoch").lr.iloc[
    :int(seeds6[(seeds6.model == V4) & (seeds6.seed == s)].best_epoch.iloc[0])].tolist() for s in [42, 7, 2026]}

# ================================================================ E1 walk-forward
months = pd.period_range(sp.test.index.min(), sp.test.index.max(), freq="M")
wf = []
for mth in months:
    start = mth.start_time
    fit = feats[feats.index < start]
    tgt = sp.test[(sp.test.index >= start) & (sp.test.index <= mth.end_time)]
    if len(tgt) == 0:
        continue
    y = tgt[T]
    out = {"Naive-1 + hourly step": baselines.PersistencePlusStep().fit(fit).predict(tgt)}
    lm_m = lm.fit_final(features.linear_design(fit, True), fit[T], "lasso", lasso_alpha)
    out["Lasso + hour×month"] = lm_m.predict(features.linear_design(tgt, True))
    out["XGBoost"] = make_xgb().fit(fit[FT], fit[T]).predict(tgt[FT])
    out["Random Forest"] = make_rf().fit(fit[FT], fit[T]).predict(tgt[FT])
    out[V4] = np.mean([mlp.MLPForecaster(v4cfg, FL, seed=s).refit_replay(fit, v4_sched[s]).predict(tgt)
                       for s in v4_sched], 0)
    for m, p in out.items():
        wf.append(ev.to_long(m, "test", y, p).assign(month=str(mth), fit_rows=len(fit)))
    print(f"walk-forward {mth}: fit rows {len(fit):,}, target hours {len(tgt)}  ({time.time() - t0:.0f}s)", flush=True)
wf = pd.concat(wf, ignore_index=True)
wf.to_csv(config.TABLE_DIR / "phase7_walkforward_predictions.csv", index=False)

static = ev.load_all_predictions()
static = static[(static.split == "test") & static.model.isin(wf.model.unique())].copy()
static["month"] = static.timestamp.dt.to_period("M").astype(str)
rows = []
for (m, mth), g in wf.groupby(["model", "month"]):
    s = static[(static.model == m) & (static.month == mth)]
    rows.append({"model": m, "month": mth, "n": len(g),
                 "MAE_walkforward": np.mean(np.abs(g.y_true - g.y_pred)),
                 "MAE_static": np.mean(np.abs(s.y_true - s.y_pred))})
wft = pd.DataFrame(rows)
wft.round(3).to_csv(config.TABLE_DIR / "phase7_walkforward.csv", index=False)
print(wft.pivot(index="month", columns="model", values="MAE_walkforward").round(1))

# ================================================================ E2 ablation
abl_rows, abl_preds = [], []
for setname, cols in features.ABLATION.items():
    for split_name, fit_part, ev_part in [("val", sp.train, sp.val), ("test", sp.dev, sp.test)]:
        ps_ = [make_xgb(s).fit(fit_part[cols], fit_part[T]).predict(ev_part[cols]) for s in (42, 7, 2026)]
        p = np.mean(ps_, 0)
        maes = [np.mean(np.abs(ev_part[T] - q)) for q in ps_]
        abl_rows.append({"feature_set": setname, "split": split_name, "n_features": len(cols),
                         "MAE_ensemble": np.mean(np.abs(ev_part[T] - p)), "MAE_seed_mean": np.mean(maes),
                         "MAE_seed_std": np.std(maes, ddof=1)})
        if split_name == "test":
            abl_preds.append(pd.DataFrame({"timestamp": ev_part.index, "set": setname, "y_true": ev_part[T].values, "y_pred": p}))
    print(f"ablation {setname} done ({time.time() - t0:.0f}s)", flush=True)
abl = pd.DataFrame(abl_rows)
ap = pd.concat(abl_preds).pivot(index="timestamp", columns="set", values="y_pred")
yt = pd.concat(abl_preds).groupby("timestamp").y_true.first()
dm_rows = []
names = list(features.ABLATION)
for a, b in zip(names[:-1], names[1:]):
    r = cp.diebold_mariano((yt - ap[a]).values, (yt - ap[b]).values)
    dm_rows.append({"from": a, "to": b, "MAE_change": -r["mean_diff"], "p_value": r["p_value"]})
pd.DataFrame(dm_rows).round(5).to_csv(config.TABLE_DIR / "phase7_ablation_dm.csv", index=False)
abl.round(3).to_csv(config.TABLE_DIR / "phase7_ablation.csv", index=False)
print(abl.round(2).to_string(index=False))
print(pd.DataFrame(dm_rows).round(4).to_string(index=False))

# ================================================================ E3 SHAP
model = joblib.load(config.MODEL_DIR / "XGBoost.joblib")
sample = sp.test[FT].sample(2000, random_state=config.SEED)
expl = shap.TreeExplainer(model.regressor_)
sv = expl.shap_values(sample)
pd.DataFrame(sv, columns=FT, index=sample.index).to_csv(config.TABLE_DIR / "phase7_shap_values.csv")
sample.to_csv(config.TABLE_DIR / "phase7_shap_sample.csv")
mabs = pd.Series(np.abs(sv).mean(0), index=FT).sort_values(ascending=False)
mabs.round(3).to_csv(config.TABLE_DIR / "phase7_shap_importance.csv", header=["mean_abs_shap_MW"])
# additivity check: base value + sum of SHAP = model's predicted change
recon = expl.expected_value + sv.sum(1)
raw = model.regressor_.predict(sample)
print("SHAP additivity max error (MW):", float(np.abs(recon - raw).max()))
print(mabs.head(12).round(2))
print(f"done ({time.time() - t0:.0f}s)")
