"""Phase 5 figures (+ two small diagnostic experiments: tree-depth sweep, boosting curve)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from src import config, data, features, split  # noqa: E402
from src import plotstyle as ps  # noqa: E402
from src import tree_models as tm  # noqa: E402

ps.apply()
T, F = config.TARGET, features.FEATURE_SETS["tree"]
sp = split.chronological_split(features.build_features(data.load_clean()))
table = pd.read_csv(config.TABLE_DIR / "phase5_trees.csv")
best = pd.read_csv(config.TABLE_DIR / "phase5_best_params.csv")
tall = table[(table.split == "test") & (table.subset == "all")].set_index("model")
STEP, LASSO = tall.loc["Naive-1 + hourly step", "MAE"], tall.loc["Lasso + hour×month", "MAE"]


def titled(ax, title, sub):
    n = ps.subtitle(ax, sub)
    ax.set_title(title, pad=12 + 12 * n)


# ================================================================ F20 decision-tree depth sweep (CV on dev)
dt_best = best[(best.model == "Decision Tree") & (best.split == "test")].iloc[0]
depths = list(range(2, 26))
rows = []
for leaf in (1, int(dt_best.min_samples_leaf)):
    for d in depths:
        tr_err, va_err = [], []
        for tr, va in TimeSeriesSplit(n_splits=5).split(sp.dev):
            m = tm.TargetMode(DecisionTreeRegressor(max_depth=d, min_samples_leaf=leaf, random_state=config.SEED),
                              dt_best.target)
            a, b = sp.dev.iloc[tr], sp.dev.iloc[va]
            m.fit(a[F], a[T])
            tr_err.append(np.mean(np.abs(a[T] - m.predict(a[F]))))
            va_err.append(np.mean(np.abs(b[T] - m.predict(b[F]))))
        rows.append({"min_samples_leaf": leaf, "max_depth": d, "train_mae": np.mean(tr_err), "cv_mae": np.mean(va_err)})
dsw = pd.DataFrame(rows)
dsw.to_csv(config.TABLE_DIR / "phase5_tree_depth_sweep.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 4.8))
summary = {}
for i, leaf in enumerate(sorted(dsw.min_samples_leaf.unique())):
    g = dsw[dsw.min_samples_leaf == leaf]
    col = ps.C[i]
    ax.plot(g.max_depth, g.train_mae, color=col, ls="--", lw=1.5)
    ax.plot(g.max_depth, g.cv_mae, color=col, lw=2.2, marker="o", ms=3.5)
    k = g.cv_mae.idxmin()
    ax.plot(g.loc[k, "max_depth"], g.loc[k, "cv_mae"], "o", ms=8, color=col, mec=ps.SURFACE, mew=1.5, zorder=4)
    summary[leaf] = (int(g.loc[k, "max_depth"]), g.loc[k, "cv_mae"], g.cv_mae.iloc[-1], g.train_mae.iloc[-1])
handles = [Line2D([], [], color=ps.C[i], lw=2.2, label=f"min_samples_leaf = {leaf}")
           for i, leaf in enumerate(sorted(summary))] + [
    Line2D([], [], color=ps.TEXT_2, lw=2.2, label="validation (5-fold forward CV)"),
    Line2D([], [], color=ps.TEXT_2, lw=1.5, ls="--", label="training")]
ax.legend(handles=handles, loc="upper right", ncol=2)
ax.set_xlabel("Maximum tree depth (model complexity)")
ax.set_ylabel("MAE (MW)")
l1, lb = summary[1], summary[int(dt_best.min_samples_leaf)]
titled(ax, "Fig 20. One decision tree: underfitting → overfitting, and how leaf size controls it",
       f"Unconstrained leaves (1 h): validation error is lowest at depth {l1[0]} ({l1[1]:.0f} MW), then rises to "
       f"{l1[2]:.0f} MW as the tree memorises the training data (training error → {l1[3]:.0f}). "
       f"Requiring ≥{int(dt_best.min_samples_leaf)} h per leaf caps that: validation error plateaus at ~{lb[2]:.0f} MW.")
ps.save(fig, "F20_tree_depth_sweep")

# ================================================================ F21 boosting = gradient descent in function space
xb = best[(best.model == "XGBoost") & (best.split == "test")].iloc[0]
curves = {}
for lr in sorted({0.02, 0.1, float(xb.learning_rate)}):
    reg = XGBRegressor(tree_method="hist", n_jobs=-1, random_state=config.SEED, verbosity=0,
                       n_estimators=1500, learning_rate=lr, max_depth=int(xb.max_depth),
                       min_child_weight=float(xb.min_child_weight), subsample=float(xb.subsample),
                       colsample_bytree=float(xb.colsample_bytree), reg_lambda=float(xb.reg_lambda),
                       reg_alpha=float(xb.reg_alpha), eval_metric="mae")
    off_tr = sp.train["load_lag_1"] if xb.target == "delta" else 0
    off_va = sp.val["load_lag_1"] if xb.target == "delta" else 0
    reg.fit(sp.train[F], sp.train[T] - off_tr,
            eval_set=[(sp.train[F], sp.train[T] - off_tr), (sp.val[F], sp.val[T] - off_va)], verbose=False)
    r = reg.evals_result()
    curves[lr] = (np.array(r["validation_0"]["mae"]), np.array(r["validation_1"]["mae"]))
fig, ax = plt.subplots(figsize=(10, 4.8))
for i, (lr, (trc, vac)) in enumerate(sorted(curves.items())):
    col = ps.C[i]
    ax.plot(np.arange(1, len(trc) + 1), trc, color=col, ls="--", lw=1.4)
    ax.plot(np.arange(1, len(vac) + 1), vac, color=col, lw=2)
    k = int(np.argmin(vac))
    ax.plot(k + 1, vac[k], "o", ms=7, color=col, mec=ps.SURFACE, mew=1.5, zorder=4)
ax.set_yscale("log")
ax.set_ylim(top=min(400, max(v[1][0] for v in curves.values())))
ax.set_xlabel("Boosting rounds (number of trees) = gradient-descent iterations")
ax.set_ylabel("MAE (MW), log scale")
bests = {lr: (float(v.min()), int(v.argmin()) + 1, float(v[-1])) for lr, (_, v) in curves.items()}
handles = [Line2D([], [], color=ps.C[i], lw=2, label=f"η = {lr:g}: best {bests[lr][0]:.1f} MW @ {bests[lr][1]} trees")
           for i, lr in enumerate(sorted(curves))] + [
    Line2D([], [], color=ps.TEXT_2, lw=2, label="validation (Val split)"),
    Line2D([], [], color=ps.TEXT_2, lw=1.4, ls="--", label="training")]
ax.legend(handles=handles, loc="upper right")
lo, hi = min(bests), max(bests)
titled(ax, "Fig 21. XGBoost learns by gradient descent in function space",
       f"Each round adds a tree fitted to the current residuals, scaled by the step size η. η = {hi:g} reaches its best in "
       f"{bests[hi][1]} rounds but at a higher error ({bests[hi][0]:.1f} MW) and then overfits (→ {bests[hi][2]:.1f}); "
       f"η = {lo:g} needs {bests[lo][1]} rounds and gets lower ({bests[lo][0]:.1f}). Training error keeps falling either way.")
ps.save(fig, "F21_boosting_curve")
pd.DataFrame({f"lr_{lr}_{s}": c for lr, (a, b) in curves.items() for s, c in [("train", a), ("val", b)]}).to_csv(
    config.TABLE_DIR / "phase5_boosting_curves.csv", index_label="round")

# ================================================================ F22 level vs delta target
names = ["Decision Tree", "Random Forest", "XGBoost"]
chosen = {n: best[(best.model == n) & (best.split == "test")].iloc[0].target for n in names}
other = {n: ("level" if chosen[n] == "delta" else "delta") for n in names}
fig, ax = plt.subplots(figsize=(9, 4))
yy = np.arange(len(names))
h = 0.36
for j, mode in enumerate(["level", "delta"]):
    vals = [tall.loc[n, "MAE"] if chosen[n] == mode else tall.loc[f"{n} [{mode} target]", "MAE"] for n in names]
    ax.barh(yy + (j - 0.5) * h, vals, height=h, color=ps.C[j], label=f"predict {'the level  y(h)' if mode == 'level' else 'the change  y(h) − y(h−1)'}")
    for i, v in enumerate(vals):
        tag = "  ← chosen by CV" if chosen[names[i]] == mode else ""
        ax.text(v + 1, i + (j - 0.5) * h, f"{v:.1f}{tag}", va="center", fontsize=9, color=ps.TEXT)
ax.set_yticks(yy, names)
ax.invert_yaxis()
ax.set_xlabel("Test MAE (MW)")
ax.set_xlim(0, max(tall.loc[[f"{n} [{other[n]} target]" for n in names], "MAE"].max(), tall.loc[names, "MAE"].max()) * 1.35)
ax.legend(loc="lower right")
titled(ax, "Fig 22. Same model, two targets: predicting the level vs the hourly change",
       "The target was chosen by cross-validation on Train+Val; the other variant is shown only as a diagnostic. "
       "Predicting the change gives each leaf a small, stationary quantity instead of a whole load level.")
ps.save(fig, "F22_level_vs_delta")

# ================================================================ F23 XGBoost importance
imp = pd.read_csv(config.TABLE_DIR / "phase5_importance.csv", index_col=0)["XGBoost"].sort_values(ascending=False)
top = imp.head(15)[::-1]
grp = {c: ("Load history" if c in features.LOAD else "Weather" if c in features.WEATHER else "Calendar") for c in imp.index}
gcol = {"Load history": ps.C[0], "Calendar": ps.C[1], "Weather": ps.C[2]}
fig, ax = plt.subplots(figsize=(9, 5.8))
ax.barh(range(len(top)), top.values * 100, color=[gcol[grp[c]] for c in top.index], height=0.7)
for i, v in enumerate(top.values):
    ax.text(v * 100 + 0.3, i, f"{v:.1%}", va="center", fontsize=8.5, color=ps.TEXT)
ax.set_yticks(range(len(top)), top.index, fontsize=9)
ax.set_xlabel("Share of total gain (%)")
ax.legend(handles=[Patch(color=c, label=g) for g, c in gcol.items()], loc="lower right")
wshare = imp[[c for c in imp.index if grp[c] == "Weather"]].sum()
lshare = imp[[c for c in imp.index if grp[c] == "Load history"]].sum()
hshare = imp[["hour", "hour_sin", "hour_cos"]].sum()
titled(ax, "Fig 23. What XGBoost relies on (gain importance, top 15 of 28)",
       f"Load history carries {lshare:.0%} of the gain, led by momentum (load_ramp_1, {imp['load_ramp_1']:.0%}). The three "
       f"hour encodings together carry {hshare:.0%}: the scheduled steps found in Phase 3. Weather: {wshare:.1%}.")
ps.save(fig, "F23_xgb_importance")

# ================================================================ F24 all models so far
order = ["Naive-1 (persistence)", "Naive-1 + hourly step", "Lasso + hour×month", "Decision Tree", "Random Forest", "XGBoost"]
t = tall.loc[order]
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.barh(range(len(order)), t["MAE"], color=[ps.FAMILY_COLORS[ps.family(m)] for m in order], height=0.6)
for i, m in enumerate(order):
    ax.text(t.loc[m, "MAE"] + 2, i, f"{t.loc[m, 'MAE']:.1f} MW · {t.loc[m, 'MAPE']:.2f}%", va="center", fontsize=9, color=ps.TEXT)
for x, lab, ha in [(STEP, f" baseline bar {STEP:.1f}", "left"), (LASSO, f"best linear {LASSO:.1f} ", "right")]:
    ax.axvline(x, color=ps.TEXT_2, ls="--", lw=1)
    ax.text(x, -0.8, lab, ha=ha, fontsize=8.5, color=ps.TEXT_2)
ax.legend(handles=[Patch(color=c, label=f) for f, c in ps.FAMILY_COLORS.items() if f != "Neural"], loc="lower right")
ax.set_yticks(range(len(order)), order)
ax.set_ylim(len(order) - 0.4, -1.1)
ax.set_xlim(0, 240)
ax.set_xlabel("Test MAE (MW)")
bt = t.loc[["Decision Tree", "Random Forest", "XGBoost"], "MAE"].idxmin()
titled(ax, "Fig 24. Test MAE: baselines, best linear model and tree models",
       f"Best tree model: {bt} at {t.loc[bt, 'MAE']:.1f} MW, "
       f"{1 - t.loc[bt, 'MAE'] / LASSO:+.0%} vs the best linear model and {1 - t.loc[bt, 'MAE'] / STEP:+.0%} vs the baseline bar.")
ps.save(fig, "F24_results_so_far")
print(dsw.pivot(index='max_depth', columns='min_samples_leaf', values=['train_mae','cv_mae']).round(1).to_string())
print({lr: (round(float(v.min()), 1), int(v.argmin()) + 1) for lr, (_, v) in curves.items()})
