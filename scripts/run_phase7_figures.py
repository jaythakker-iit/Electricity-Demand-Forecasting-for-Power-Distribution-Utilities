"""Phase 7 figures F28–F36 (read-only: everything comes from results/tables)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402

from src import compare as cp  # noqa: E402
from src import config, data, evaluation as ev, features, split  # noqa: E402
from src import plotstyle as ps  # noqa: E402

ps.apply()
TD = config.TABLE_DIR
T = config.TARGET
sp = split.chronological_split(features.build_features(data.load_clean()))
S = cp.SHORT
LINE = ["Naive-1 + hourly step", "Lasso + hour×month", "Random Forest", "XGBoost", "MLP v4 [64,32] Δ-target"]
MARK = {"Naive-1 + hourly step": "s", "Lasso + hour×month": "^", "Random Forest": "o", "XGBoost": "D",
        "MLP v4 [64,32] Δ-target": "v", "Naive-1 (persistence)": "X"}
col = ps.MODEL_COLORS


def titled(ax, title, sub):
    n = ps.subtitle(ax, sub)
    ax.set_title(title, pad=12 + 12 * n)


def end_labels(ax, x_last, series: dict, dx=0.4, min_gap=None):
    """Direct labels at the right end of lines, nudged apart so they never overlap."""
    items = sorted(series.items(), key=lambda kv: kv[1])
    lo, hi = ax.get_ylim()
    gap = min_gap or (hi - lo) * 0.045
    placed = []
    for m, y in items:
        y2 = max(y, placed[-1] + gap) if placed else y
        placed.append(y2)
        ax.text(x_last + dx, y2, S[m], fontsize=8.5, color=ps.TEXT, va="center")


ci = pd.read_csv(TD / "compare_ci.csv")
dm = pd.read_csv(TD / "compare_dm.csv")

# ================================================================ F28 final ranking with CIs
c = ci.set_index("model").loc[cp.FINAL[::-1]]
fig, ax = plt.subplots(figsize=(10, 4.4))
for i, (m, r) in enumerate(c.iterrows()):
    ax.plot([r.ci_low, r.ci_high], [i, i], color=ps.FAMILY_COLORS[ps.family(m)], lw=3, solid_capstyle="round")
    ax.plot(r.MAE, i, MARK[m], ms=9, color=ps.FAMILY_COLORS[ps.family(m)], mec=ps.SURFACE, mew=1.5, zorder=3)
    ax.text(r.ci_high + 3, i, f"{r.MAE:.1f}  [{r.ci_low:.1f}, {r.ci_high:.1f}]", va="center", fontsize=9, color=ps.TEXT)
ax.set_yticks(range(len(c)), [S[m] for m in c.index])
ax.set_xscale("log")
ax.set_xticks([40, 50, 60, 80, 100, 150, 200, 250], ["40", "50", "60", "80", "100", "150", "200", "250"])
ax.set_xlim(40, 330)
from matplotlib.ticker import NullFormatter  # noqa: E402
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Test MAE (MW), log scale — dot = estimate, bar = 95% block-bootstrap CI")
ax.legend(handles=[Patch(color=v, label=k) for k, v in ps.FAMILY_COLORS.items()], loc="lower right")
top3 = dm[(dm.model_a.isin(["Random Forest", "XGBoost", "MLP v4 [64,32] Δ-target"])) &
          (dm.model_b.isin(["Random Forest", "XGBoost", "MLP v4 [64,32] Δ-target"]))]
titled(ax, "Fig 28. Final ranking on the test set (Jun 2025 – Jan 2026, 4,762 hours)",
       f"Random Forest, XGBoost and the MLP overlap completely (pairwise DM tests p ≥ {top3.p_value.min():.2f}): "
       "a three-way tie. Every other step down the ranking is statistically significant (p < 0.001).")
ps.save(fig, "F28_final_ranking_ci")

# ================================================================ F29 pairwise significance matrix
names = cp.FINAL
M = pd.DataFrame(np.nan, index=names, columns=names)
P = M.copy()
for _, r in dm.iterrows():
    M.loc[r.model_a, r.model_b] = r.MAE_a_minus_b
    M.loc[r.model_b, r.model_a] = -r.MAE_a_minus_b
    P.loc[r.model_a, r.model_b] = P.loc[r.model_b, r.model_a] = r.p_holm
fig, ax = plt.subplots(figsize=(8.8, 6.4))
lim = 30
im = ax.imshow(M.clip(-lim, lim).values, cmap=ps.DIVERGING, vmin=-lim, vmax=lim)
for i in range(len(names)):
    for j in range(len(names)):
        if i == j:
            ax.text(j, i, "—", ha="center", va="center", color=ps.TEXT_2)
            continue
        v, p = M.iloc[i, j], P.iloc[i, j]
        txt = f"{v:+.1f}\n{'p<0.001' if p < 0.001 else f'p={p:.2f}'}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                color="white" if abs(v) > 20 else ps.TEXT, fontweight="bold" if p >= 0.05 else "normal")
ax.set_xticks(range(len(names)), [S[m] for m in names], rotation=30, ha="right")
ax.set_yticks(range(len(names)), [S[m] for m in names])
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8)
cb.set_label("Row MAE − column MAE (MW); clipped at ±30", color=ps.TEXT_2)
cb.outline.set_visible(False)
titled(ax, "Fig 29. Pairwise Diebold–Mariano tests (Holm-adjusted)",
       "Negative (blue) = the row model is more accurate. Bold cells are NOT significant at 5%: only the "
       "Random Forest / XGBoost / MLP block, the three-way tie.")
ps.save(fig, "F29_dm_matrix")

# ================================================================ F30 error by hour
bh = pd.read_csv(TD / "breakdown_hour.csv", index_col=0)
fig, ax = plt.subplots(figsize=(10.5, 4.8))
for m in LINE:
    ax.plot(bh.index, bh[m], color=col[m], marker=MARK[m], ms=4, lw=1.8, label=S[m])
ax.set_xlim(-0.5, 26.5)
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Test MAE (MW)")
end_labels(ax, 23, {m: bh[m].iloc[-1] for m in LINE})
ax.legend(loc="upper left", ncol=3)
best3 = bh[["Random Forest", "XGBoost", "MLP v4 [64,32] Δ-target"]].mean(axis=1)
hard = best3.nlargest(3).index.tolist()
titled(ax, "Fig 30. When are forecasts hardest? Error by hour of day (test)",
       f"All models struggle at the same hours: {', '.join(f'{h}:00' for h in sorted(hard))}, around the scheduled "
       "load steps and the morning ramp. The nonlinear models' advantage over the step baseline holds at every hour.")
ps.save(fig, "F30_error_by_hour")

# ================================================================ F31 month / level / day type
bm = pd.read_csv(TD / "breakdown_month.csv", index_col=0)
bl = pd.read_csv(TD / "breakdown_level.csv", index_col=0).loc[["Low", "Medium", "High"]]
bd = pd.read_csv(TD / "breakdown_daytype.csv", index_col=0).loc[["Working day", "Weekend", "Festival"]]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.6, 1, 1]})
ax = axes[0]
xm = np.arange(len(bm))
for m in LINE:
    ax.plot(xm, bm[m], color=col[m], marker=MARK[m], ms=5, lw=1.8, label=S[m])
ax.set_xticks(xm, [("Jun*" if i == 0 else pd.Period(p).strftime("%b")) + ("\n2025" if i == 0 else "\n2026" if i == len(bm) - 1 else "") for i, p in enumerate(bm.index)])
ax.set_ylabel("Test MAE (MW)")
ax.set_title("By month (*Jun = last 5 days only)", fontsize=10, pad=6)
ax.legend(loc="upper right", fontsize=8)
w = 0.15
for ax, tb, ttl in [(axes[1], bl, "By demand level (train terciles)"), (axes[2], bd, "By day type")]:
    xx = np.arange(len(tb))
    for k, m in enumerate(LINE):
        ax.bar(xx + (k - 2) * w, tb[m], width=w, color=col[m], label=S[m])
    ax.set_xticks(xx, [f"{i}\n(n={int(n):,})" for i, n in zip(tb.index, tb.n)])
    ax.set_title(ttl, fontsize=10, pad=6)
fig.suptitle("Fig 31. Where the error comes from: season, demand level and day type (test)", x=0.01, ha="left",
             fontweight="bold", fontsize=12, color=ps.TEXT)
fest_gap = (bd.loc["Festival", "XGBoost"] / bd.loc["Working day", "XGBoost"] - 1)
nl = bl[["Random Forest", "XGBoost", "MLP v4 [64,32] Δ-target"]].mean(axis=1)
n_jun = int(bm.n.iloc[0])
full = bm.iloc[1:]
hard_nl = full[["Random Forest", "XGBoost", "MLP v4 [64,32] Δ-target"]].mean(axis=1).nlargest(2).index
fig.text(0.01, 0.885, f"June: only {n_jun} test hours. Among full months, November is easiest for every model; the hardest is October for "
         f"the step baseline and Lasso but {' and '.join(pd.Period(p).strftime('%B') for p in sorted(hard_nl))} for the nonlinear models. "
         f"For the nonlinear models error rises with demand ({nl['Low']:.0f} → {nl['Medium']:.0f} → {nl['High']:.0f} MW); "
         f"festival hours are {fest_gap:+.0%} harder for XGBoost, but n = {int(bd.loc['Festival', 'n'])}, so that estimate is noisy.",
         fontsize=9, color=ps.TEXT_2, wrap=True)
fig.tight_layout(rect=(0, 0, 1, 0.86))
ps.save(fig, "F31_error_breakdowns")

# ================================================================ F32 peak hours
bp = pd.read_csv(TD / "breakdown_peak.csv", index_col=0)
bb = pd.read_csv(TD / "breakdown_peak_bias.csv", index_col=0)
pc = pd.read_csv(TD / "peak_classification.csv").set_index("model")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
names5 = ["Naive-1 (persistence)"] + LINE
for ax, vals, ttl, xl in [(axes[0], bp.loc["Peak (≥ train P90)", names5], "MAE on peak hours", "MAE (MW)"),
                          (axes[1], bb.loc["Peak", names5], "Bias on peak hours (+ = under-forecast)", "Mean error y − ŷ (MW)"),
                          (axes[2], pc.loc[names5, "F1"], "Flagging peak hours (F1)", "F1 score")]:
    ax.barh(range(len(names5)), vals, color=[col[m] for m in names5], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + (0.01 if xl == "F1 score" else 2), i, f"{v:.2f}" if xl == "F1 score" else f"{v:.1f}", va="center", fontsize=8.5, color=ps.TEXT)
    ax.set_yticks(range(len(names5)), [S[m] for m in names5])
    ax.invert_yaxis()
    ax.set_title(ttl, fontsize=10, pad=6)
    ax.set_xlabel(xl)
axes[2].set_xlim(0.75, 1.0)
axes[1].axvline(0, color=ps.TEXT_2, lw=0.8)
thr = ev.peak_threshold(sp.train[T])
fig.suptitle(f"Fig 32. Peak hours: load ≥ {thr:,.0f} MW (train 90th percentile), {int(bp.loc['Peak (≥ train P90)', 'n'])} test hours",
             x=0.01, ha="left", fontweight="bold", fontsize=12, color=ps.TEXT)
fig.text(0.01, 0.9, "The nonlinear models are as accurate on peaks as overall, with a small positive bias (slight "
         "under-forecast). The Lasso model under-forecasts peaks the most among the learned models.",
         fontsize=9, color=ps.TEXT_2)
fig.tight_layout(rect=(0, 0, 1, 0.88))
ps.save(fig, "F32_peak_hours")

# ================================================================ F33 walk-forward
wf = pd.read_csv(TD / "phase7_walkforward.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
months = sorted(wf.month.unique())
xm = np.arange(len(months))
for m in LINE:
    g = wf[wf.model == m].set_index("month").loc[months]
    ax.plot(xm, g.MAE_walkforward, color=col[m], marker=MARK[m], ms=5, lw=1.8, label=S[m])
ax.set_xticks(xm, [pd.Period(p).strftime("%b") for p in months])
ax.set_ylabel("MAE (MW), model retrained before each month")
ax.set_title("Monthly MAE with monthly retraining", fontsize=10, pad=6)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax = axes[1]
tot = wf.assign(ws=wf.MAE_walkforward * wf.n, ss=wf.MAE_static * wf.n).groupby("model")[["ws", "ss", "n"]].sum()
tot = (tot[["ws", "ss"]].div(tot.n, axis=0)).loc[LINE]
gain = (1 - tot.ws / tot.ss) * 100
ax.barh(range(len(LINE)), gain, color=[col[m] for m in LINE], height=0.6)
for i, (m, g) in enumerate(gain.items()):
    ax.text(max(g, 0) + 0.05, i, f"{g:+.1f}%  ({tot.loc[m, 'ss']:.1f} → {tot.loc[m, 'ws']:.1f})",
            va="center", ha="left", fontsize=8.5, color=ps.TEXT)
ax.axvline(0, color=ps.TEXT_2, lw=0.8)
ax.set_yticks(range(len(LINE)), [S[m] for m in LINE])
ax.invert_yaxis()
ax.set_xlim(-1, 5)
ax.set_xlabel("MAE reduction from monthly retraining (%)")
ax.set_title("Retrain monthly vs fit once", fontsize=10, pad=6)
fig.suptitle("Fig 33. Walk-forward evaluation: is performance stable, and does retraining help?", x=0.01, ha="left",
             fontweight="bold", fontsize=12, color=ps.TEXT)
fig.text(0.01, 0.9, "The ranking holds in every month. Retraining each month on all data so far trims 1.6–2.5% "
         "off the nonlinear models; the lookup baseline and Lasso barely change.", fontsize=9, color=ps.TEXT_2)
fig.tight_layout(rect=(0, 0, 1, 0.88))
ps.save(fig, "F33_walkforward")

# ================================================================ F34 ablation (incremental steps)
ab = pd.read_csv(TD / "phase7_ablation.csv")
adm = pd.read_csv(TD / "phase7_ablation_dm.csv").set_index("to")
steps = [("B_+weather", "A_load_calendar", "+ weather (h−1)"), ("C_+festival", "B_+weather", "+ festival flag"),
         ("D_+weather_at_h (what-if)", "C_+festival", "+ weather at hour h\n(what-if: perfect forecast)")]
fig, ax = plt.subplots(figsize=(10, 4.2))
for k, (sp_name, cc) in enumerate([("val", ps.C[0]), ("test", ps.C[1])]):
    g = ab[ab.split == sp_name].set_index("feature_set").MAE_ensemble
    d = [g[to] - g[fr] for to, fr, _ in steps]
    yy = np.arange(len(steps)) + (k - 0.5) * 0.36
    ax.barh(yy, d, height=0.36, color=cc, label=sp_name.capitalize())
    for i, v in enumerate(d):
        ptxt = f"  (p = {adm.loc[steps[i][0], 'p_value']:.4f})" if sp_name == "test" else ""
        ax.text(v + (0.01 if v >= 0 else -0.01), yy[i], f"{v:+.2f}{ptxt}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=ps.TEXT)
ax.axvline(0, color=ps.TEXT_2, lw=0.8)
ax.set_yticks(range(len(steps)), [s_[2] for s_ in steps])
ax.invert_yaxis()
ax.set_xlim(-0.7, 0.8)
ax.set_xlabel("MAE change when the group is ADDED (MW); + = worse.  XGBoost, Δ-target, mean of 3 seeds; p = test DM")
ax.legend(loc="lower right")
gA = ab[(ab.split == "test")].set_index("feature_set").MAE_ensemble
titled(ax, "Fig 34. Feature-group ablation: what do weather and festivals add?",
       "Weather makes the model slightly worse on both Val and Test, even perfect weather at the forecast hour. "
       f"The festival flag helps on Test but hurts slightly on Val. Load + calendar alone ({gA['A_load_calendar']:.1f} MW) "
       f"matches the full model ({gA['C_+festival']:.1f} MW).")
ps.save(fig, "F34_ablation")

# ================================================================ F35 SHAP
sv = pd.read_csv(TD / "phase7_shap_values.csv", index_col=0)
Xs = pd.read_csv(TD / "phase7_shap_sample.csv", index_col=0)
imp = sv.abs().mean().sort_values(ascending=False)
top = imp.index[:12][::-1]
fig, ax = plt.subplots(figsize=(10, 6))
rng = np.random.default_rng(0)
for i, f in enumerate(top):
    x = sv[f].values
    v = Xs[f].values.astype(float)
    vn = (v - np.nanpercentile(v, 5)) / (np.nanpercentile(v, 95) - np.nanpercentile(v, 5) + 1e-9)
    ax.scatter(x, i + rng.uniform(-0.28, 0.28, len(x)), c=np.clip(vn, 0, 1), cmap=ps.DIVERGING, s=5, alpha=0.6, lw=0)
ax.axvline(0, color=ps.TEXT_2, lw=0.8)
ax.set_yticks(range(len(top)), [f"{f}  ({imp[f]:.0f} MW)" for f in top])
ax.set_xlabel("SHAP value: contribution to the predicted hourly CHANGE (MW)")
sm = plt.cm.ScalarMappable(cmap=ps.DIVERGING)
cb = fig.colorbar(sm, ax=ax, shrink=0.6, ticks=[0, 1])
cb.ax.set_yticklabels(["low\nfeature value", "high\nfeature value"])
cb.outline.set_visible(False)
titled(ax, "Fig 35. SHAP: how each feature moves XGBoost's forecast (2,000 test hours)",
       f"Mean |SHAP| in brackets. Momentum (load_ramp_1, {imp['load_ramp_1']:.0f} MW) dominates: a rising load (red) "
       "pushes the forecast up. The hour features encode the scheduled steps. Weather sits near zero.")
ps.save(fig, "F35_shap_summary")

# ================================================================ F36 one week + residual ACF
w = cp.wide(ev.load_all_predictions(), "test")
start = pd.Timestamp("2025-10-13")
wk = w.loc[start:start + pd.Timedelta(days=7)]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
ax.plot(wk.index, wk.y_true, color=ps.TEXT, lw=2.4, label="Actual")
for m in ["Naive-1 + hourly step", "XGBoost"]:
    ax.plot(wk.index, wk[m], color=col[m], lw=1.4, ls="--" if "step" in m else "-", label=S[m])
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
ax.set_ylabel("Load (MW)")
ax.legend(loc="upper left", ncol=3)
ax.set_title(f"One test week from {start:%d %b %Y}", fontsize=10, pad=6)
ax = axes[1]
res = (w.y_true - w.XGBoost).reindex(pd.date_range(w.index.min(), w.index.max(), freq="h")).interpolate()
ra = acf(res, nlags=48, fft=True)
ax.vlines(range(len(ra)), 0, ra, color=ps.C[6], lw=1.2)
ax.axhline(1.96 / np.sqrt(len(res)), color=ps.TEXT_2, ls="--", lw=0.8)
ax.axhline(-1.96 / np.sqrt(len(res)), color=ps.TEXT_2, ls="--", lw=0.8)
ax.set_xlabel("Lag (hours)")
ax.set_ylabel("Autocorrelation")
ax.set_title(f"XGBoost residual ACF: lag 1 = {ra[1]:.2f}, lag 24 = {ra[24]:.2f}", fontsize=10, pad=6)
step_res = (w.y_true - w["Naive-1 + hourly step"]).reindex(res.index).interpolate()
rs = acf(step_res, nlags=24, fft=True)
fig.suptitle("Fig 36. What the best model does, and what it leaves behind", x=0.01, ha="left", fontweight="bold",
             fontsize=12, color=ps.TEXT)
fig.text(0.01, 0.9, f"Residual autocorrelation fell from {rs[1]:.2f} (step baseline) to {ra[1]:.2f} at lag 1 and from "
         f"{rs[24]:.2f} to {ra[24]:.2f} at lag 24. What remains peaks at lag 24: errors repeat at the same hour on consecutive "
         "days, a lead for future work (feed yesterday's forecast error back as a feature).", fontsize=9, color=ps.TEXT_2)
fig.tight_layout(rect=(0, 0, 1, 0.88))
ps.save(fig, "F36_week_and_residual_acf")
pd.DataFrame({"lag": range(len(ra)), "acf_xgb": ra}).to_csv(TD / "phase7_residual_acf.csv", index=False)
print("hard hours:", hard, " residual acf xgb lag1/24:", round(ra[1], 3), round(ra[24], 3), " step:", round(rs[1], 3), round(rs[24], 3))
print(bd.round(1).to_string()); print(bl.round(1).to_string())
