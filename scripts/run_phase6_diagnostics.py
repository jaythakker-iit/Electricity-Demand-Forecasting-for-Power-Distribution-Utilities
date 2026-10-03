"""Phase 6 diagnostics (evaluator follow-ups; NOT used to select the final model unless it wins on VAL).

D1  v3b = brief v3 (lr 1e-4, batch 512) with a PATIENT scheduler (lr_patience 15, stop patience 40,
    600 epochs). Tests whether v3 lost because of lr=1e-4 itself or because ReduceLROnPlateau halved
    the rate to ~1e-6 before convergence.
D2  Refit variance: score the TRAIN-only early-stopped models on test, next to the Train+Val refits,
    to measure how much seed spread the schedule-replay refit adds.
"""
import json
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, data, evaluation as ev, features, split  # noqa: E402
from src import mlp  # noqa: E402
from src.metrics import evaluate  # noqa: E402

T, F = config.TARGET, features.FEATURE_SETS["mlp"]
sp = split.chronological_split(features.build_features(data.load_clean()))
SEEDS = [42, 7, 2026]
t0 = time.time()

# ---------------------------------------------------------------- D1
v3b = mlp.MLPConfig("MLP v3b [256,128,64] lr1e-4 patient", [256, 128, 64], dropout=0.3, lr=1e-4,
                    batch_size=512, max_epochs=600, patience=40, lr_patience=15,
                    notes="diagnostic: brief v3 with a patient LR scheduler")
vals, tests, rows = [], [], []
for s in SEEDS:
    m = mlp.MLPForecaster(v3b, F, seed=s).fit(sp.train, sp.val)
    pv = m.predict(sp.val)
    sched = m.lr_schedule_
    m.refit_replay(sp.dev, sched)
    pt = m.predict(sp.test)
    vals.append(pv); tests.append(pt)
    rows.append({"model": v3b.name, "seed": s, "best_epoch": len(sched), "final_lr": sched[-1],
                 "val_MAE": evaluate(sp.val[T], pv)["MAE"], "test_MAE": evaluate(sp.test[T], pt)["MAE"]})
    print(rows[-1], f"({time.time() - t0:.0f}s)", flush=True)
pv, pt = np.mean(vals, 0), np.mean(tests, 0)
d1 = pd.concat([ev.to_long(v3b.name, "val", sp.val[T], pv), ev.to_long(v3b.name, "test", sp.test[T], pt)])
ev.save_predictions(d1, "phase6b_mlp_diag")
meta = json.load(open(config.TABLE_DIR / "phase6_meta.json"))
meta["configs"][v3b.name] = {**v3b.__dict__, "val_MAE_ensemble": evaluate(sp.val[T], pv)["MAE"]}
vals_all = {k: v["val_MAE_ensemble"] for k, v in meta["configs"].items()}
meta["selected_on_val"] = min(vals_all, key=vals_all.get)
json.dump(meta, open(config.TABLE_DIR / "phase6_meta.json", "w"), indent=2, default=str)
seed_tab = pd.read_csv(config.TABLE_DIR / "phase6_seed_results.csv")
seed_tab = pd.concat([seed_tab[seed_tab.model != v3b.name], pd.DataFrame(rows)], ignore_index=True)
seed_tab.to_csv(config.TABLE_DIR / "phase6_seed_results.csv", index=False)

# ---------------------------------------------------------------- D2
cfgs = {c: meta["configs"][c] for c in ["MLP v1 [64,32]", "MLP v4 [64,32] Δ-target"]}
rows2 = []
for name, c in cfgs.items():
    cfg = mlp.MLPConfig(**{k: v for k, v in c.items() if k in mlp.MLPConfig.__dataclass_fields__})
    for s in SEEDS:
        m = mlp.MLPForecaster(cfg, F, seed=s).fit(sp.train, sp.val)
        rows2.append({"model": name, "seed": s, "test_MAE_train_only": evaluate(sp.test[T], m.predict(sp.test))["MAE"]})
    print(name, "done", f"({time.time() - t0:.0f}s)", flush=True)
d2 = pd.DataFrame(rows2).merge(seed_tab[["model", "seed", "test_MAE"]].rename(columns={"test_MAE": "test_MAE_refit"}),
                               on=["model", "seed"])
d2.to_csv(config.TABLE_DIR / "phase6_refit_variance.csv", index=False)
print(d2.round(1).to_string(index=False))
print(d2.groupby("model")[["test_MAE_train_only", "test_MAE_refit"]].agg(["mean", "std"]).round(1))
print("selected on val:", meta["selected_on_val"])
