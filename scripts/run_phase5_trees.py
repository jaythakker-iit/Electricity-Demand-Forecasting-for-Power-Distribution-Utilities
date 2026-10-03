"""Phase 5 runner: Decision Tree, Random Forest, XGBoost.

Protocol (same as Phase 4):
  Val  score: hyperparameters chosen by forward-chaining CV on TRAIN, refit on TRAIN.
  Test score: hyperparameters chosen by forward-chaining CV on TRAIN+VAL, refit on TRAIN+VAL.
Target mode ('level' vs 'delta') is one of the tuned hyperparameters.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, data, evaluation as ev, features, split  # noqa: E402
from src import tree_models as tm  # noqa: E402

T = config.TARGET
F = features.FEATURE_SETS["tree"]
sp = split.chronological_split(features.build_features(data.load_clean()))
VAL_ITER = {"Decision Tree": 30, "Random Forest": 8, "XGBoost": 20}   # cheaper search for the secondary val score

preds, best_rows, imps = [], [], {}
t0 = time.time()
for name in ["Decision Tree", "Random Forest", "XGBoost"]:
    for split_name, fit_part, ev_part, n_iter in [("val", sp.train, sp.val, VAL_ITER[name]),
                                                   ("test", sp.dev, sp.test, None)]:
        search = tm.tune(name, fit_part[F], fit_part[T], n_iter)
        model = search.best_estimator_
        preds.append(ev.to_long(name, split_name, ev_part[T], model.predict(ev_part[F])))
        bp = {k.replace("regressor__", ""): v for k, v in search.best_params_.items()}
        best_rows.append({"model": name, "split": split_name, "cv_mae": -search.best_score_,
                          "n_iter": search.n_iter, **{k: (v if v is not None else "None") for k, v in bp.items()}})
        if split_name == "test":
            tm.cv_table(search).to_csv(config.TABLE_DIR / f"phase5_cv_{name.replace(' ', '_')}.csv", index=False)
            imps[name] = tm.importance(model)
            joblib.dump(model, config.MODEL_DIR / f"{name.replace(' ', '_')}.joblib")
        print(f"{name:14s} {split_name:4s} cv_mae={-search.best_score_:7.2f}  {bp}  ({time.time() - t0:.0f}s)", flush=True)

    # Diagnostic (NOT used for selection): same tuned hyperparameters, the other target mode, on test.
    best = joblib.load(config.MODEL_DIR / f"{name.replace(' ', '_')}.joblib")
    other = "level" if best.target == "delta" else "delta"
    alt = tm.TargetMode(best.regressor, other).fit(sp.dev[F], sp.dev[T])
    preds.append(ev.to_long(f"{name} [{other} target]", "test", sp.test[T], alt.predict(sp.test[F])))

preds = pd.concat(preds, ignore_index=True)
ev.save_predictions(preds, "phase5_trees")
pd.DataFrame(best_rows).to_csv(config.TABLE_DIR / "phase5_best_params.csv", index=False)
pd.DataFrame(imps).to_csv(config.TABLE_DIR / "phase5_importance.csv")

thr = ev.peak_threshold(sp.train[T])
allp = ev.load_all_predictions()
keep = ["Naive-1 (persistence)", "Naive-1 + hourly step", "Lasso + hour×month"]
table = ev.add_skill(ev.score(pd.concat([allp[allp.model.isin(keep)], preds]), thr))
ref = table[table.model == "Lasso + hour×month"].set_index(["split", "subset"])["MAE"]
step = table[table.model == "Naive-1 + hourly step"].set_index(["split", "subset"])["MAE"]
idx = table.set_index(["split", "subset"]).index
table["skill_vs_step"] = 1 - table["MAE"] / idx.map(step)
table["skill_vs_lasso"] = 1 - table["MAE"] / idx.map(ref)
table.round(4).to_csv(config.TABLE_DIR / "phase5_trees.csv", index=False)
print(table[table.subset == "all"][["split", "model", "MAE", "RMSE", "MAPE", "R2", "Bias", "skill_vs_step", "skill_vs_lasso"]]
      .round(3).to_string(index=False))
print(pd.DataFrame(best_rows).to_string(index=False))
json.dump({"seconds": round(time.time() - t0)}, open(config.TABLE_DIR / "phase5_runtime.json", "w"))
