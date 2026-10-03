"""Phase 4 figures + analysis tables (reads saved results; refits only the small diagnostic models)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src import config, data, features, split  # noqa: E402
from src import linear_models as lm  # noqa: E402
from src import plotstyle as ps  # noqa: E402

ps.apply()
T = config.TARGET


def titled(ax, title, sub):
    n = ps.subtitle(ax, sub)
    ax.set_title(title, pad=12 + 12 * n)


feats = features.build_features(data.load_clean())
sp = split.chronological_split(feats)
curves = pd.read_csv(config.TABLE_DIR / "phase4_cv_curves.csv")
alphas = pd.read_csv(config.TABLE_DIR / "phase4_alphas.csv")
table = pd.read_csv(config.TABLE_DIR / "phase4_linear.csv")
coefs = pd.read_csv(config.TABLE_DIR / "phase4_coefficients_std.csv", index_col=0)
BAR = table[(table.split == "test") & (table.subset == "all") & (table.model == "Naive-1 + hourly step")].MAE.iloc[0]
PERSIST = table[(table.split == "test") & (table.subset == "all") & (table.model == "Naive-1 (persistence)")].MAE.iloc[0]

# ================================================================ F15 CV curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, fam in zip(axes, ["Ridge", "Lasso"]):
    for i, (suffix, lab) in enumerate([("", "47 features"), (" + hour×month", "309 features (+ hour×month)")]):
        name = fam + suffix
        c = curves[(curves.model == name) & (curves.split == "test")]
        g = c.groupby("alpha")["mse"].agg(["mean", "std", "count"])
        rmse = np.sqrt(g["mean"])
        se = (g["std"] / np.sqrt(g["count"])) / (2 * rmse)          # delta-method SE on the RMSE scale
        col = ps.C[i]
        ax.plot(rmse.index, rmse, color=col, label=lab)
        ax.fill_between(rmse.index, rmse - se, rmse + se, color=col, alpha=0.15, lw=0)
        a = alphas[(alphas.model == name) & (alphas.split == "test")].iloc[0]
        for key, mk, fc in [("alpha_best", "o", col), ("alpha_1se", "o", ps.SURFACE)]:
            x = a[key]
            yv = np.interp(np.log10(x), np.log10(rmse.index), rmse.values)
            ax.plot(x, yv, mk, ms=8, mfc=fc, mec=col, mew=2, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Regularisation strength α (log scale)")
    ax.set_title(fam + (" (L2)" if fam == "Ridge" else " (L1)"), fontsize=11, pad=6)
    ax.set_ylim(80, 400)
axes[0].set_ylabel("CV RMSE (MW), mean ± 1 SE over 5 folds")
handles = [Line2D([], [], color=ps.C[0], label="47 features"), Line2D([], [], color=ps.C[1], label="309 features (+ hour×month)"),
           Line2D([], [], ls="", marker="o", ms=8, mfc=ps.TEXT_2, mec=ps.TEXT_2, label="best α (min CV error)"),
           Line2D([], [], ls="", marker="o", ms=8, mfc=ps.SURFACE, mec=ps.TEXT_2, mew=2, label="1-SE α (simplest within 1 SE)")]
axes[1].legend(handles=handles, loc="upper left")
fig.suptitle("Fig 15. Choosing α by forward-chaining cross-validation (Train+Val)", x=0.01, ha="left",
             fontweight="bold", fontsize=12, color=ps.TEXT)
fig.text(0.01, 0.905, "The curves are flat for small α: with ~19,000 rows, extra shrinkage buys no accuracy. "
         "Error climbs steeply once α is large enough to bias the coefficients. "
         "Interactions lower the whole curve.", fontsize=9, color=ps.TEXT_2)
fig.tight_layout(rect=(0, 0, 1, 0.89))
ps.save(fig, "F15_cv_alpha_curves")

# ================================================================ F16 Lasso path
X = features.linear_design(sp.dev, False)
y = sp.dev[T]
grid = lm.lasso_alphas(X, y, n=60)
path = lm.lasso_path(X, y, grid)
path.to_csv(config.TABLE_DIR / "phase4_lasso_path.csv")
group = {c: ("Load history" if c in features.LOAD_LINEAR else "Weather" if c in features.WEATHER
             else "Festival" if c == "is_festival" else "Calendar") for c in X.columns}
gcol = {"Load history": ps.C[0], "Calendar": ps.MUTED, "Weather": ps.C[2], "Festival": ps.C[1]}
fig, ax = plt.subplots(figsize=(11, 5.6))
for c in X.columns:
    g = group[c]
    ax.plot(path.index, path[c], color=gcol[g], lw=2 if g != "Calendar" else 0.9,
            alpha=1 if g != "Calendar" else 0.55, zorder=3 if g != "Calendar" else 2)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=10)
ax.invert_xaxis()
ax.axhline(0, color=ps.TEXT_2, lw=0.8)
a = alphas[(alphas.model == "Lasso") & (alphas.split == "test")].iloc[0]
for key, lab in [("alpha_1se", "1-SE α"), ("alpha_best", "best α")]:
    ax.axvline(a[key], color=ps.TEXT_2, ls="--", lw=1)
    ax.text(a[key], 0.06, f"{lab} ", transform=ax.get_xaxis_transform(), fontsize=9, color=ps.TEXT_2,
            va="bottom", ha="right", bbox=dict(fc=ps.SURFACE, ec="none", pad=1.5))
for c in ["load_lag_1", "load_lag_2", "load_roll_mean_24", "load_roll_mean_168", "temp_lag_1"]:
    v = path[c].iloc[-1]
    ax.annotate(c, (path.index[-1], v), xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                color=ps.TEXT, va="center")
ax.set_xlim(path.index.max() * 1.2, path.index.min() / 30)
ax.set_xlabel("α (log scale, decreasing →: less regularisation)")
ax.set_ylabel("Standardised coefficient (MW per 1 SD), symlog scale")
ax.legend(handles=[Line2D([], [], color=v, lw=2, label=k) for k, v in gcol.items()], loc="upper left")
nz_order = (path.abs() > 1e-9).idxmax()       # first (largest) alpha at which each feature becomes non-zero
first = nz_order.sort_values(ascending=False).index[:3].tolist()
t_peak = path["temp_lag_1"].max()
t_final = path["temp_lag_1"].iloc[-1]
rank_t = list(nz_order.sort_values(ascending=False).index).index("temp_lag_1") + 1
titled(ax, "Fig 16. Lasso path: the order in which features enter the model",
       f"Reading left to right (less regularisation), the first in are {', '.join(first)}. temp_lag_1 enters "
       f"{rank_t}th, before any hour dummy, as a stand-in for time of day (up to {t_peak:+.0f}), then shrinks and flips "
       f"to {t_final:+.0f} once the calendar enters: the confounding of Fig 5, seen inside the model.")
ps.save(fig, "F16_lasso_path")

# ================================================================ F17 bias-variance (equal windows)
X47 = features.linear_design(sp.dev, False)
bv = lm.window_bias_variance(X47, y, [0, 0.1, 0.3, 1, 3, 10, 30, 100], features.LOAD_LINEAR, days=30)
bv.to_csv(config.TABLE_DIR / "phase4_bias_variance.csv", index=False)
xs = np.arange(len(bv))
labels = ["0\n(OLS)" if a == 0 else f"{a:g}" for a in bv.alpha]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].plot(xs, bv.coef_spread, color=ps.C[0], marker="o", ms=6)
axes[0].set_ylabel("Σ std of load coefficients across windows")
axes[0].set_title("Variance ↓  (coefficients more stable)", fontsize=10, pad=6)
axes[1].plot(xs, bv.next_window_mae, color=ps.C[1], marker="o", ms=6)
axes[1].set_ylabel("MAE on the next 30-day window (MW)")
axes[1].set_title("Bias ↑  (forecasts get worse)", fontsize=10, pad=6)
for ax in axes:
    ax.set_xticks(xs, labels)
    ax.set_xlabel("Ridge α")
fig.suptitle("Fig 17. The bias–variance trade-off, measured: Ridge on 25 equal 30-day windows", x=0.01, ha="left",
             fontweight="bold", fontsize=12, color=ps.TEXT)
fig.text(0.01, 0.9, f"Raising α cuts coefficient variance by {1 - bv.coef_spread.iloc[-1] / bv.coef_spread.iloc[0]:.0%} "
         f"but raises forecast error from {bv.next_window_mae.iloc[0]:.0f} to {bv.next_window_mae.iloc[-1]:.0f} MW. "
         "Here, shrinking the dominant lag-1 coefficient costs more than the variance it removes.",
         fontsize=9, color=ps.TEXT_2)
fig.tight_layout(rect=(0, 0, 1, 0.87))
ps.save(fig, "F17_bias_variance")

# ================================================================ F18 results vs bar
order = ["OLS", "Ridge", "Lasso", "OLS + hour×month", "Ridge + hour×month", "Lasso + hour×month"]
t = table[(table.split == "test") & (table.subset == "all")].set_index("model").loc[order]
fig, ax = plt.subplots(figsize=(10, 4.4))
cols = [ps.C[0]] * 3 + [ps.C[1]] * 3
ax.barh(range(len(order)), t["MAE"], color=cols, height=0.6)
for i, m in enumerate(order):
    ax.text(t.loc[m, "MAE"] + 1.5, i, f"{t.loc[m, 'MAE']:.1f} MW · {t.loc[m, 'MAPE']:.2f}%  (skill vs bar {t.loc[m, 'skill_vs_step']:+.0%})",
            va="center", fontsize=9, color=ps.TEXT)
ax.axvline(BAR, color=ps.MODEL_COLORS["Naive-1 + hourly step"], lw=2, ls="--")
ax.text(BAR, -0.75, f"bar: Naive-1 + hourly step {BAR:.1f}", fontsize=9, color=ps.TEXT, ha="center")
ax.set_yticks(range(len(order)), order)
ax.invert_yaxis()
ax.set_xlim(0, 135)
ax.set_ylim(len(order) - 0.4, -1.1)
ax.set_xlabel(f"Test MAE (MW)   ·   persistence = {PERSIST:.0f} MW (off scale)")
best = t["MAE"].idxmin()
titled(ax, "Fig 18. Linear models on the test set",
       f"Without hour×month terms every linear model loses to the 74 MW bar. With them, {best} reaches "
       f"{t.loc[best, 'MAE']:.1f} MW ({1 - t.loc[best, 'MAE'] / BAR:.0%} better than the bar). OLS, Ridge and Lasso are tied within 1 MW.")
ps.save(fig, "F18_linear_results")

# ================================================================ F19 learned hour x month surface
m = __import__("joblib").load(config.MODEL_DIR / "Lasso_+_hourxmonth.joblib")
Xi = features.linear_design(sp.dev, True)
coef = pd.Series(m.named_steps["model"].coef_ / m.named_steps["scaler"].scale_, index=Xi.columns)   # MW per unit
surf = np.zeros((24, 12))
for h in range(24):
    for mo in range(1, 13):
        v = 0.0
        v += coef.get(f"hour_{h}", 0.0) if h > 0 else 0.0
        v += coef.get(f"month_{mo}", 0.0) if mo > 1 else 0.0
        v += coef.get(f"h{h}_m{mo}", 0.0) if (h > 0 and mo > 1) else 0.0
        surf[h, mo - 1] = v
surf -= surf.mean(axis=0, keepdims=True)           # show the shape WITHIN each month
fig, ax = plt.subplots(figsize=(9, 5.6))
lim = np.abs(surf).max()
im = ax.imshow(surf, aspect="auto", origin="lower", cmap=ps.DIVERGING, vmin=-lim, vmax=lim)
ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_yticks(range(0, 24, 3))
ax.set_xlabel("Month")
ax.set_ylabel("Hour of day")
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("Hour effect within month (MW, centred)", color=ps.TEXT_2)
cb.outline.set_visible(False)
titled(ax, "Fig 19. What the interaction terms learned (Lasso + hour×month)",
       "Calendar adjustment the model adds on top of the load lags. Its pattern changes by season, "
       "which is why a single additive hour effect (47-feature model) fell short.")
ps.save(fig, "F19_learned_hour_month")

print(bv.round(1).to_string(index=False))
print("first-in features:", first)
