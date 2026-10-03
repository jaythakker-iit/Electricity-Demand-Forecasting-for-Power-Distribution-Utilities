"""Phase 6 figures + final MLP table (includes the v3b diagnostic)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from src import config, data, evaluation as ev, features, split  # noqa: E402
from src import plotstyle as ps  # noqa: E402

ps.apply()
T = config.TARGET
sp = split.chronological_split(features.build_features(data.load_clean()))
meta = json.load(open(config.TABLE_DIR / "phase6_meta.json"))
SEL = meta["selected_on_val"]
seeds = pd.read_csv(config.TABLE_DIR / "phase6_seed_results.csv")
hist = pd.read_csv(config.TABLE_DIR / "phase6_histories.csv")


def titled(ax, title, sub):
    n = ps.subtitle(ax, sub)
    ax.set_title(title, pad=12 + 12 * n)


# ---------------------------------------------------------------- final table (all MLPs incl. diagnostic)
allp = ev.load_all_predictions()
keep = ["Naive-1 (persistence)", "Naive-1 + hourly step", "Lasso + hour×month", "Random Forest", "XGBoost"]
mlps = [m for m in allp.model.unique() if m.startswith("MLP")]
thr = ev.peak_threshold(sp.train[T])
table = ev.add_skill(ev.score(allp[allp.model.isin(keep + mlps)], thr))
idx = table.set_index(["split", "subset"]).index
for col, ref in [("skill_vs_step", "Naive-1 + hourly step"), ("skill_vs_xgb", "XGBoost")]:
    table[col] = 1 - table["MAE"] / idx.map(table[table.model == ref].set_index(["split", "subset"])["MAE"])
table["selected_on_val"] = table.model == SEL
table.round(4).to_csv(config.TABLE_DIR / "phase6_mlp.csv", index=False)
tall = table[(table.split == "test") & (table.subset == "all")].set_index("model")
tval = table[(table.split == "val") & (table.subset == "all")].set_index("model")

ORDER = ["MLP v1 [64,32]", "MLP v2 [256,128,64]", "MLP v3 [256,128,64] lr1e-4",
         "MLP v3b [256,128,64] lr1e-4 patient", "MLP v4 [64,32] Δ-target"]
SHORT = {m: m.replace("MLP ", "").replace(" [256,128,64]", "").replace(" [64,32]", "") for m in ORDER}

# ================================================================ F25 training curves (seed 42)
fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
for ax, m in zip(axes.flat, ORDER):
    g = hist[(hist.model == m) & (hist.seed == 42)]
    if g.empty:      # v3b histories are not stored by the diagnostic script
        ax.axis("off")
        continue
    ax.plot(g.epoch, g.train_loss, color=ps.C[0], lw=1.4, ls="--", label="training")
    ax.plot(g.epoch, g.val_loss, color=ps.C[1], lw=2, label="validation")
    b = g.loc[g.val_loss.idxmin()]
    ax.plot(b.epoch, b.val_loss, "o", ms=7, color=ps.C[1], mec=ps.SURFACE, mew=1.5, zorder=4)
    drops = g.epoch[g.lr.diff() < 0]
    for e in drops:
        ax.axvline(e, color=ps.GRID, lw=1, zorder=0)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f"{SHORT[m]} — best epoch {int(b.epoch)}, {len(drops)} LR halvings", fontsize=10, pad=6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardised target)")
# last panel: LR schedules
ax = axes.flat[5]
for i, m in enumerate([o for o in ORDER if not hist[hist.model == o].empty]):
    g = hist[(hist.model == m) & (hist.seed == 42)]
    ax.plot(g.epoch, g.lr, color=ps.C[i], lw=1.8, label=SHORT[m], drawstyle="steps-post")
ax.set_yscale("log")
ax.set_title("Learning rate (ReduceLROnPlateau)", fontsize=10, pad=6)
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning rate")
ax.legend(fontsize=8, loc="lower left")
axes.flat[0].legend(loc="upper right")
# panel 4 (v3b) has no stored history: annotate instead
axv = axes.flat[3]
axv.axis("on")
axv.set_axis_off()
v3, v3b = seeds[seeds.model == ORDER[2]], seeds[seeds.model == ORDER[3]]
axv.text(0.02, 0.95, "v3 vs v3b (same lr 1e-4, batch 512)", fontsize=10, fontweight="bold", va="top", transform=axv.transAxes)
axv.text(0.02, 0.80,
         f"v3  (brief): scheduler patience 6\n   final LR ≈ {v3.final_lr.median():.0e}, val MAE {v3.val_MAE.mean():.1f}\n\n"
         f"v3b (diagnostic): patience 15, stop 40\n   final LR ≈ {v3b.final_lr.median():.0e}, val MAE {v3b.val_MAE.mean():.1f}\n\n"
         "With small steps, validation loss improves\nslowly and noisily; an impatient scheduler\n"
         "reads the noise as a plateau and keeps\nhalving the LR until training stalls.",
         fontsize=9, color=ps.TEXT, va="top", transform=axv.transAxes, family="DejaVu Sans")
fig.suptitle("Fig 25. MLP training: gradient descent, early stopping and learning-rate decay (seed 42)", x=0.01,
             ha="left", fontweight="bold", fontsize=12, color=ps.TEXT)
fig.text(0.01, 0.935, "Dots = epoch restored by early stopping; grey verticals = LR halvings. Training loss sits above "
         "validation loss for the level models only because dropout is active while training: measured for v1, training "
         "loss is 0.0138 with dropout on but 0.0033 with it off (validation 0.0088).", fontsize=9, color=ps.TEXT_2)
fig.tight_layout(rect=(0, 0, 1, 0.92))
ps.save(fig, "F25_mlp_training_curves")

# ================================================================ F26 seed variance + ensembling
fig, ax = plt.subplots(figsize=(10, 4.6))
for i, m in enumerate(ORDER):
    g = seeds[seeds.model == m]
    ax.scatter(g.test_MAE, [i] * len(g), s=55, color=ps.C[0], alpha=0.8, zorder=3,
               label="single seed" if i == 0 else None, edgecolor=ps.SURFACE, linewidth=1)
    ax.scatter(tall.loc[m, "MAE"], i, marker="D", s=70, color=ps.C[1], zorder=4,
               label="3-seed ensemble" if i == 0 else None, edgecolor=ps.SURFACE, linewidth=1)
    ax.text(max(g.test_MAE.max(), tall.loc[m, "MAE"]) + 1.5, i,
            f"ensemble {tall.loc[m, 'MAE']:.1f}   seeds {g.test_MAE.mean():.1f} ± {g.test_MAE.std():.1f}",
            va="center", fontsize=9, color=ps.TEXT)
xgb = tall.loc["XGBoost", "MAE"]
ax.axvline(xgb, color=ps.TEXT_2, ls="--", lw=1)
ax.text(xgb, -0.85, f"XGBoost {xgb:.1f}", ha="center", fontsize=8.5, color=ps.TEXT_2)
ax.set_yticks(range(len(ORDER)), [SHORT[m] + ("  ★ selected on Val" if m == SEL else "") for m in ORDER])
ax.set_ylim(len(ORDER) - 0.4, -1.2)
ax.set_xlim(45, 140)
ax.set_xlabel("Test MAE (MW)")
ax.legend(loc="lower right")
v1 = seeds[seeds.model == ORDER[0]]
titled(ax, "Fig 26. Seed-to-seed variation and the effect of averaging seeds",
       f"Level-target MLPs swing by up to {v1.test_MAE.max() - v1.test_MAE.min():.0f} MW between random seeds; averaging 3 seeds "
       f"recovers most of it. The Δ-target MLP is both better and far more stable (±{seeds[seeds.model == ORDER[4]].test_MAE.std():.1f} MW).")
ps.save(fig, "F26_mlp_seed_variance")

# ================================================================ F27 results so far
order = ["Naive-1 (persistence)", "Naive-1 + hourly step", "Lasso + hour×month", "Random Forest", "XGBoost",
         "MLP v1 [64,32]", SEL]
order = list(dict.fromkeys(order))
t = tall.loc[order]
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.barh(range(len(order)), t["MAE"], color=[ps.FAMILY_COLORS[ps.family(m)] for m in order], height=0.6)
for i, m in enumerate(order):
    ax.text(t.loc[m, "MAE"] + 2, i, f"{t.loc[m, 'MAE']:.1f} MW · {t.loc[m, 'MAPE']:.2f}%", va="center", fontsize=9, color=ps.TEXT)
ax.set_yticks(range(len(order)), [m + ("  (selected)" if m == SEL else "") for m in order])
ax.invert_yaxis()
ax.set_xlim(0, 240)
ax.set_xlabel("Test MAE (MW)")
ax.legend(handles=[Patch(color=c, label=f) for f, c in ps.FAMILY_COLORS.items()], loc="lower right")
titled(ax, "Fig 27. Test MAE after Phase 6: every model family",
       f"The selected MLP ({tall.loc[SEL, 'MAE']:.1f} MW) lands within {tall.loc[SEL, 'MAE'] / xgb - 1:+.1%} of XGBoost and Random Forest. "
       "Three very different model families converge on ≈52 MW once they predict the hourly change.")
ps.save(fig, "F27_results_after_mlp")

print(table[table.subset == "all"][["split", "model", "MAE", "MAPE", "Bias", "skill_vs_step", "skill_vs_xgb"]].round(3).to_string(index=False))
print(table[(table.subset == "peak") & (table.split == "test")][["model", "MAE", "Bias"]].round(1).to_string(index=False))
