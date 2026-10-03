"""Phase 4 runner: OLS / Ridge / Lasso, with and without hour x month interactions.

Protocol (identical for every model):
  * Val score : alpha chosen by 5-fold forward-chaining CV on TRAIN, model refit on TRAIN.
  * Test score: alpha chosen by 5-fold forward-chaining CV on TRAIN+VAL, refit on TRAIN+VAL.
  The test set is used exactly once, for the final numbers.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, data, evaluation as ev, features, split  # noqa: E402
from src import linear_models as lm  # noqa: E402

T = config.TARGET
hourly = data.load_clean()
feats = features.build_features(hourly)
sp = split.chronological_split(feats)

DESIGNS = {"": False, " + hour×month": True}
preds, curves, chosen, coefs = [], [], [], {}
t0 = time.time()
for suffix, inter in DESIGNS.items():
    for split_name, fit_part, eval_part in [("val", sp.train, sp.val), ("test", sp.dev, sp.test)]:
        X_fit, y_fit = features.linear_design(fit_part, inter), fit_part[T]
        X_ev, y_ev = features.linear_design(eval_part, inter), eval_part[T]
        for kind in ("ols", "ridge", "lasso"):
            name = {"ols": "OLS", "ridge": "Ridge", "lasso": "Lasso"}[kind] + suffix
            alpha = None
            if kind != "ols":
                grid = lm.RIDGE_ALPHAS if kind == "ridge" else lm.lasso_alphas(X_fit, y_fit)
                curve = lm.cv_curve(X_fit, y_fit, kind, grid)
                sel = lm.select_alpha(curve)
                alpha = sel["best"]
                curves.append(curve.assign(model=name, split=split_name))
                chosen.append({"model": name, "split": split_name, "alpha_best": sel["best"],
                               "alpha_1se": sel["one_se"], "n_features": X_fit.shape[1],
                               "best_at_low_edge": sel["at_low_edge"], "curve_flat_there": sel["flat_at_low_edge"],
                               "grid_min": min(grid), "grid_max": max(grid)})
                # 1-SE variant: most regularised model statistically tied with the best
                m1 = lm.fit_final(X_fit, y_fit, kind, sel["one_se"])
                preds.append(ev.to_long(f"{name} (1-SE α)", split_name, y_ev, m1.predict(X_ev)))
                if split_name == "test":
                    coefs[f"{name} (1-SE α)"] = pd.Series(m1.named_steps["model"].coef_, index=X_fit.columns)
            model = lm.fit_final(X_fit, y_fit, kind, alpha)
            preds.append(ev.to_long(name, split_name, y_ev, model.predict(X_ev)))
            if split_name == "test":
                coefs[name] = pd.Series(model.named_steps["model"].coef_, index=X_fit.columns)
                joblib.dump(model, config.MODEL_DIR / f"{name.replace(' ', '_').replace('×', 'x')}.joblib")
            a_txt = "—" if alpha is None else f"{alpha:.4g}"
            print(f"{name:22s} {split_name:4s} alpha={a_txt:>10}  ({time.time() - t0:.0f}s)", flush=True)

preds = pd.concat(preds, ignore_index=True)
ev.save_predictions(preds, "phase4_linear")
pd.concat(curves).to_csv(config.TABLE_DIR / "phase4_cv_curves.csv", index=False)
pd.DataFrame(chosen).to_csv(config.TABLE_DIR / "phase4_alphas.csv", index=False)
pd.DataFrame(coefs).to_csv(config.TABLE_DIR / "phase4_coefficients_std.csv")

thr = ev.peak_threshold(sp.train[T])
base = ev.load_all_predictions()
base = base[base.model.isin(["Naive-1 (persistence)", "Naive-1 + hourly step"])]
table = ev.add_skill(ev.score(pd.concat([base, preds]), thr))
table["skill_vs_step"] = 1 - table["MAE"] / table.set_index(["split", "subset"]).index.map(
    table[table.model == "Naive-1 + hourly step"].set_index(["split", "subset"])["MAE"])
table.round(4).to_csv(config.TABLE_DIR / "phase4_linear.csv", index=False)
print(table[table.subset == "all"][["split", "model", "MAE", "RMSE", "MAPE", "R2", "skill_vs_naive1", "skill_vs_step"]]
      .round(3).to_string(index=False))
print(pd.DataFrame(chosen).to_string(index=False))
json.dump({"seconds": round(time.time() - t0)}, open(config.TABLE_DIR / "phase4_runtime.json", "w"))
