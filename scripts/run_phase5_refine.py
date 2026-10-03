"""Phase 5, stage 2: refine the search where stage 1 picked values on the EDGE of its grid.

Stopping rule (fixed before running): ONE refinement stage only. Further edge-chasing
would start to overfit the CV folds themselves; any edges that remain are documented.

Selection is still by forward-chaining CV on past data only (Train for the val score,
Train+Val for the test score). A stage-2 model replaces the stage-1 model only if its
CV MAE is lower; the test set plays no part in that decision.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit  # noqa: E402

from src import config, data, evaluation as ev, features, split  # noqa: E402
from src import tree_models as tm  # noqa: E402

T, F = config.TARGET, features.FEATURE_SETS["tree"]
sp = split.chronological_split(features.build_features(data.load_clean()))

STAGE2 = {
    "Random Forest": ({
        "regressor__max_features": [0.15, 0.2, 0.25, 0.33],
        "regressor__min_samples_leaf": [1],
        "regressor__max_depth": [18, 24],
        "target": ["delta"],
    }, 8),
    "XGBoost": ({
        "regressor__learning_rate": [0.01, 0.02],
        "regressor__n_estimators": [600, 1000, 1500, 2500],
        "regressor__max_depth": [10, 12, 14],
        "regressor__min_child_weight": [20, 50, 100],
        "regressor__colsample_bytree": [0.4, 0.5, 0.6],
        "regressor__reg_lambda": [0.01, 0.1],
        "regressor__subsample": [0.85],
        "regressor__reg_alpha": [0],
        "target": ["delta"],
    }, 24),
}

# keep stage-1 results for the record (outside the predictions folder, so they are not re-scored)
import shutil  # noqa: E402
for f in ["phase5_best_params.csv", "phase5_trees.csv"]:
    shutil.copy(config.TABLE_DIR / f, config.TABLE_DIR / f.replace(".csv", "_stage1.csv"))
shutil.copy(ev.PRED_DIR / "predictions_phase5_trees.csv", config.TABLE_DIR / "phase5_predictions_stage1.csv")
best1 = pd.read_csv(config.TABLE_DIR / "phase5_best_params.csv")
best1["n_iter"] = best1["n_iter"].astype(str)          # will hold e.g. "40+24 (stage 2)"
best1 = best1.astype({c: object for c in best1.columns if c not in ("model", "split", "cv_mae")})
preds = pd.read_csv(ev.PRED_DIR / "predictions_phase5_trees.csv", parse_dates=["timestamp"])
log = []
t0 = time.time()
for name, (grid, n_iter) in STAGE2.items():
    for split_name, fit_part, ev_part in [("val", sp.train, sp.val), ("test", sp.dev, sp.test)]:
        cv1 = best1[(best1.model == name) & (best1.split == split_name)].cv_mae.iloc[0]
        search = RandomizedSearchCV(tm.TargetMode(tm.base_models()[name]), grid, n_iter=n_iter,
                                    cv=TimeSeriesSplit(n_splits=tm.N_SPLITS), scoring="neg_mean_absolute_error",
                                    random_state=config.SEED, n_jobs=1, refit=True, return_train_score=True)
        search.fit(fit_part[F], fit_part[T])
        cv2 = -search.best_score_
        improved = cv2 < cv1
        bp = {k.replace("regressor__", ""): v for k, v in search.best_params_.items()}
        log.append({"model": name, "split": split_name, "cv_mae_stage1": cv1, "cv_mae_stage2": cv2,
                    "replaced": improved, **bp})
        print(f"{name:14s} {split_name:4s} stage1 {cv1:.2f} -> stage2 {cv2:.2f}  replaced={improved}  {bp}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if improved:
            m = search.best_estimator_
            mask = (preds.model == name) & (preds.split == split_name)
            preds.loc[mask, "y_pred"] = m.predict(ev_part[F].loc[preds.loc[mask, "timestamp"]])
            if split_name == "test":
                joblib.dump(m, config.MODEL_DIR / f"{name.replace(' ', '_')}.joblib")
                tm.cv_table(search).to_csv(config.TABLE_DIR / f"phase5_cv_{name.replace(' ', '_')}_stage2.csv", index=False)
                imp = pd.read_csv(config.TABLE_DIR / "phase5_importance.csv", index_col=0)
                imp[name] = tm.importance(m)
                imp.to_csv(config.TABLE_DIR / "phase5_importance.csv")
                # refresh the [other target] diagnostic with the new hyperparameters
                alt_name = f"{name} [level target]"
                alt = tm.TargetMode(m.regressor, "level").fit(sp.dev[F], sp.dev[T])
                amask = (preds.model == alt_name) & (preds.split == "test")
                preds.loc[amask, "y_pred"] = alt.predict(sp.test[F].loc[preds.loc[amask, "timestamp"]])
            row = best1[(best1.model == name) & (best1.split == split_name)].index[0]
            best1.loc[row, "cv_mae"] = cv2
            best1.loc[row, "n_iter"] = f"{best1.loc[row, 'n_iter']}+{n_iter} (stage 2)"
            for k, v in bp.items():
                best1.loc[row, k] = v

ev.save_predictions(preds, "phase5_trees")
best1.to_csv(config.TABLE_DIR / "phase5_best_params.csv", index=False)
pd.DataFrame(log).to_csv(config.TABLE_DIR / "phase5_stage2_log.csv", index=False)

# re-score
thr = ev.peak_threshold(sp.train[T])
allp = ev.load_all_predictions()
keep = ["Naive-1 (persistence)", "Naive-1 + hourly step", "Lasso + hour×month"]
table = ev.add_skill(ev.score(pd.concat([allp[allp.model.isin(keep)], preds]), thr))
idx = table.set_index(["split", "subset"]).index
for col, ref in [("skill_vs_step", "Naive-1 + hourly step"), ("skill_vs_lasso", "Lasso + hour×month")]:
    table[col] = 1 - table["MAE"] / idx.map(table[table.model == ref].set_index(["split", "subset"])["MAE"])
table.round(4).to_csv(config.TABLE_DIR / "phase5_trees.csv", index=False)
print(table[(table.subset == "all")][["split", "model", "MAE", "MAPE", "skill_vs_step", "skill_vs_lasso"]].round(3).to_string(index=False))
