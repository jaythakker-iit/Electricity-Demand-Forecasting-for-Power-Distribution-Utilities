"""Phase 6 runner: EnergyMLP experiments v1-v3 (project brief) + v4 (delta target), 3 seeds each.

Protocol
  * fit on TRAIN with early stopping + ReduceLROnPlateau on VAL  -> val predictions, epoch count, LR schedule
  * refit from scratch on TRAIN+VAL replaying that exact schedule -> test predictions
  * per configuration, the reported model is the 3-seed ENSEMBLE (mean of the seeds' predictions);
    per-seed scores are kept to show seed variance
  * the configuration carried forward is chosen on VAL (never on test)
"""
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import config, data, evaluation as ev, features, split  # noqa: E402
from src import mlp  # noqa: E402
from src.metrics import evaluate  # noqa: E402

T, F = config.TARGET, features.FEATURE_SETS["mlp"]
sp = split.chronological_split(features.build_features(data.load_clean()))
SEEDS = [42, 7, 2026]

CONFIGS = [
    mlp.MLPConfig("MLP v1 [64,32]", [64, 32], dropout=0.1, lr=1e-3, batch_size=256,
                  notes="brief v1: small network"),
    mlp.MLPConfig("MLP v2 [256,128,64]", [256, 128, 64], dropout=0.3, lr=1e-3, batch_size=256,
                  notes="brief v2: wider/deeper + dropout 0.3"),
    mlp.MLPConfig("MLP v3 [256,128,64] lr1e-4", [256, 128, 64], dropout=0.3, lr=1e-4, batch_size=512,
                  max_epochs=400, notes="brief v3: lr 1e-4, batch 512"),
]

seed_rows, hist_rows, preds, meta = [], [], [], {}
t0 = time.time()


def run(cfg):
    vals, tests = [], []
    for s in SEEDS:
        m = mlp.MLPForecaster(cfg, F, seed=s).fit(sp.train, sp.val)
        pv = m.predict(sp.val)
        hist_rows.append(m.history_.assign(model=cfg.name, seed=s))
        sched = m.lr_schedule_
        best_ep, n_par, secs = m.best_epoch_, m.n_params(), m.fit_seconds_
        m.refit_replay(sp.dev, sched)
        pt = m.predict(sp.test)
        torch.save(m.model_.state_dict(), config.MODEL_DIR / f"{cfg.name.split(' [')[0].replace(' ', '_')}_seed{s}.pt")
        vals.append(pv)
        tests.append(pt)
        seed_rows.append({"model": cfg.name, "seed": s, "best_epoch": best_ep, "params": n_par,
                          "final_lr": sched[-1], "fit_seconds": round(secs, 1),
                          "val_MAE": evaluate(sp.val[T], pv)["MAE"], "test_MAE": evaluate(sp.test[T], pt)["MAE"]})
        print(f"{cfg.name:30s} seed {s:4d}: epochs {best_ep:3d}  val {seed_rows[-1]['val_MAE']:6.1f}  "
              f"test {seed_rows[-1]['test_MAE']:6.1f}  ({time.time() - t0:.0f}s)", flush=True)
    pv, pt = np.mean(vals, 0), np.mean(tests, 0)
    preds.append(ev.to_long(cfg.name, "val", sp.val[T], pv))
    preds.append(ev.to_long(cfg.name, "test", sp.test[T], pt))
    meta[cfg.name] = {**asdict(cfg), "val_MAE_ensemble": evaluate(sp.val[T], pv)["MAE"]}
    return meta[cfg.name]["val_MAE_ensemble"]


val_scores = {c.name: run(c) for c in CONFIGS}

# v4: our experiment — same architecture/training as the best brief config (chosen on VAL), delta target
best_brief = min(val_scores, key=val_scores.get)
base = next(c for c in CONFIGS if c.name == best_brief)
v4 = replace(base, name=f"MLP v4 {base.name.split(' ', 2)[2]} Δ-target", target="delta",
             notes=f"ours: {best_brief} with the hourly-change target (Phase 5 finding)")
val_scores[v4.name] = run(v4)
selected = min(val_scores, key=val_scores.get)

preds = pd.concat(preds, ignore_index=True)
ev.save_predictions(preds, "phase6_mlp")
pd.DataFrame(seed_rows).to_csv(config.TABLE_DIR / "phase6_seed_results.csv", index=False)
pd.concat(hist_rows).to_csv(config.TABLE_DIR / "phase6_histories.csv", index=False)
json.dump({"configs": meta, "best_brief_config": best_brief, "selected_on_val": selected,
           "seeds": SEEDS, "seconds": round(time.time() - t0)},
          open(config.TABLE_DIR / "phase6_meta.json", "w"), indent=2, default=str)

thr = ev.peak_threshold(sp.train[T])
allp = ev.load_all_predictions()
keep = ["Naive-1 (persistence)", "Naive-1 + hourly step", "Lasso + hour×month", "Random Forest", "XGBoost"]
table = ev.add_skill(ev.score(pd.concat([allp[allp.model.isin(keep)], preds]), thr))
idx = table.set_index(["split", "subset"]).index
for col, ref in [("skill_vs_step", "Naive-1 + hourly step"), ("skill_vs_xgb", "XGBoost")]:
    table[col] = 1 - table["MAE"] / idx.map(table[table.model == ref].set_index(["split", "subset"])["MAE"])
table.round(4).to_csv(config.TABLE_DIR / "phase6_mlp.csv", index=False)
print(table[table.subset == "all"][["split", "model", "MAE", "RMSE", "MAPE", "R2", "Bias", "skill_vs_step", "skill_vs_xgb"]]
      .round(3).to_string(index=False))
print(pd.DataFrame(seed_rows).groupby("model")[["val_MAE", "test_MAE", "best_epoch"]].agg(["mean", "std"]).round(1))
print("best brief config (val):", best_brief, "| selected overall (val):", selected)
