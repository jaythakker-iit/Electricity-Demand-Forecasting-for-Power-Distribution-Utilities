"""Phase 3 runner: score naive baselines on Val and Test, reproduce the v1 off-by-one bug,
save predictions (long format) + tables + figures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

from src import baselines, config, data, evaluation as ev, features, split  # noqa: E402
from src import plotstyle as ps  # noqa: E402
from src.metrics import evaluate  # noqa: E402

ps.apply()


def titled(ax, title, sub):
    n = ps.subtitle(ax, sub)
    ax.set_title(title, pad=12 + 12 * n)


hourly = data.load_clean()
feats = features.build_features(hourly)
sp = split.chronological_split(feats)
T = config.TARGET

# ------------------------------------------------------------- predictions
rows = []
for split_name, part, fit_part in [("val", sp.val, sp.train), ("test", sp.test, sp.dev)]:
    y = part[T]
    for name, pred in baselines.naive_forecasts(part).items():
        rows.append(ev.to_long(name, split_name, y, pred))
    # climatology: fitted only on data strictly before this split (train for val, train+val for test)
    clim = baselines.CalendarClimatology().fit(fit_part)
    assert clim.fit_end_ < part.index.min(), "climatology saw the future"
    rows.append(ev.to_long(baselines.CLIM_NAME, split_name, y, clim.predict(part)))
    step = baselines.PersistencePlusStep().fit(fit_part)
    assert step.fit_end_ < part.index.min(), "step model saw the future"
    rows.append(ev.to_long(baselines.STEP_NAME, split_name, y, step.predict(part)))
preds = pd.concat(rows, ignore_index=True)
ev.save_predictions(preds, "phase3_baselines")

thr = ev.peak_threshold(sp.train[T])
table = ev.add_skill(ev.score(preds, thr))
table.round(4).to_csv(config.TABLE_DIR / "phase3_baselines.csv", index=False)

# ------------------------------------------------------------- v1 off-by-one reproduction
# v1 built Lag_24 = MW.shift(24) on the row of hour t but predicted Target = MW(t+1),
# so its "same hour yesterday" was really load at h-25.
y_test = sp.test[T]
bug_rows = []
for k in (24, 168):
    v1_pred = hourly["load_mw"].shift(k + 1).reindex(y_test.index)     # what v1 actually used
    mae_v1 = evaluate(y_test, v1_pred)["MAE"]
    mae_fixed = evaluate(y_test, sp.test[f"load_lag_{k}"])["MAE"]
    bug_rows.append({"baseline": f"naive-{k}", "v1 used": f"h-{k + 1}", "v1 MAE": mae_v1,
                     "fixed MAE": mae_fixed,
                     "v1 overstated error by": f"{mae_v1 / mae_fixed - 1:+.1%}"})
bug = pd.DataFrame(bug_rows)
bug.round(2).to_csv(config.TABLE_DIR / "phase3_v1_offbyone.csv", index=False)

# ------------------------------------------------------------- F12 bar chart
order = ["Naive-1 + hourly step", "Naive-1 (persistence)", "Seasonal naive-24", "Seasonal naive-168", "Calendar climatology"]
t_all = table[(table.split == "test") & (table.subset == "all")].set_index("model").loc[order]
t_pk = table[(table.split == "test") & (table.subset == "peak")].set_index("model").loc[order]
fig, ax = plt.subplots(figsize=(9, 4.2))
yy = np.arange(len(order))
h = 0.36
ax.barh(yy - h / 2, t_all["MAE"], height=h, color=ps.C[0], label="All test hours")
ax.barh(yy + h / 2, t_pk["MAE"], height=h, color=ps.C[1], label=f"Peak hours (≥ {thr:,.0f} MW, train P90)")
for i, m in enumerate(order):
    ax.text(t_all.loc[m, "MAE"] + 8, i - h / 2, f"{t_all.loc[m, 'MAE']:.0f} MW · {t_all.loc[m, 'MAPE']:.1f}%",
            va="center", fontsize=9, color=ps.TEXT)
    ax.text(t_pk.loc[m, "MAE"] + 8, i + h / 2, f"{t_pk.loc[m, 'MAE']:.0f} MW", va="center", fontsize=9, color=ps.TEXT)
ax.set_yticks(yy, order)
ax.invert_yaxis()
ax.set_xlabel("Mean absolute error on the test set (MW)")
ax.set_xlim(0, max(t_all["MAE"].max(), t_pk["MAE"].max()) * 1.32)
ax.legend(loc="upper right")
best = t_all["MAE"].idxmin()
titled(ax, "Fig 12. Baselines: the bar every model must beat",
       f"Best baseline: {best} (MAE {t_all.loc[best, 'MAE']:.0f} MW, MAPE {t_all.loc[best, 'MAPE']:.2f}%). "
       f"Test period {sp.test.index.min():%b %Y}–{sp.test.index.max():%b %Y}, n = {len(sp.test):,} hours.")
ps.save(fig, "F12_baselines_mae")

# ------------------------------------------------------------- F13 error by hour of day
test_p = preds[preds.split == "test"].copy()
test_p["hour"] = test_p["timestamp"].dt.hour
test_p["abs_err"] = (test_p["y_true"] - test_p["y_pred"]).abs()
by_hour = test_p.pivot_table(index="hour", columns="model", values="abs_err", aggfunc="mean")
fig, ax = plt.subplots(figsize=(9.5, 4.8))
f13 = ["Naive-1 (persistence)", "Naive-1 + hourly step", "Seasonal naive-24"]
short = {"Naive-1 (persistence)": "Naive-1", "Naive-1 + hourly step": "Naive-1 + step", "Seasonal naive-24": "Naive-24"}
for m in f13:
    ax.plot(by_hour.index, by_hour[m], color=ps.MODEL_COLORS[m], marker="o", ms=3, label=m)
    ax.annotate(short[m], (23, by_hour[m].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=9, va="center", color=ps.TEXT)
ax.set_xlim(0, 26)
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean absolute error (MW)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
worst_h = by_hour["Naive-1 (persistence)"].idxmax()
gain_9 = 1 - by_hour.loc[worst_h, "Naive-1 + hourly step"] / by_hour.loc[worst_h, "Naive-1 (persistence)"]
titled(ax, "Fig 13. Where each baseline fails: error by hour of day (test)",
       f"Persistence spikes at fixed hours where load steps up or down on a schedule (worst {worst_h}:00). "
       f"Adding the average step for that hour cuts the {worst_h}:00 error by {gain_9:.0%}: "
       f"the jumps are predictable from the clock.")
ps.save(fig, "F13_baselines_by_hour")

# ------------------------------------------------------------- F14 one test week
wk_start = sp.test.index.min().normalize() + pd.Timedelta(days=14)
wk = slice(wk_start, wk_start + pd.Timedelta(days=7))
fig, ax = plt.subplots(figsize=(11, 4))
yt = sp.test.loc[wk, T]
ax.plot(yt.index, yt, color=ps.TEXT, lw=2.2, label="Actual")
ax.plot(yt.index, sp.test.loc[wk, "load_lag_1"], color=ps.MODEL_COLORS["Naive-1 (persistence)"], lw=1.4, label="Naive-1")
ax.plot(yt.index, sp.test.loc[wk, "load_lag_24"], color=ps.MODEL_COLORS["Seasonal naive-24"], lw=1.4, ls="--", label="Naive-24")
ax.set_ylabel("Load (MW)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
ax.legend(loc="upper left", ncol=3)
titled(ax, f"Fig 14. One test week ({wk_start:%d %b %Y}): what the baselines actually do",
       "Naive-1 is the actual curve shifted right by one hour; naive-24 copies yesterday's shape. "
       "A good model must combine both: today's level with the right daily shape.")
ps.save(fig, "F14_baselines_week")

# ------------------------------------------------------------- print
show = table[table.subset == "all"][["split", "model", "MAE", "RMSE", "MAPE", "R2", "Bias", "skill_vs_naive1"]]
print(show.round(3).to_string(index=False))
print("\nPeak hours (test):")
print(table[(table.subset == "peak") & (table.split == "test")][["model", "n", "MAE", "MAPE", "Bias"]].round(2).to_string(index=False))
print("\nv1 off-by-one reproduction:")
print(bug.round(2).to_string(index=False), flush=True)
print(f"\nPeak threshold (train P90): {thr:.1f} MW")
