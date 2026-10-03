"""EDA figures (Phase 2). One function per figure; each saves a PNG and returns
a dict {figure, question, finding, modelling_decision}.

Used by scripts/run_phase2_eda.py (pipeline) and notebooks/01_EDA.ipynb (narrative).
Statistics that drive decisions (ACF, correlations, adjusted effects) are computed
on the TRAIN period only, so the test set never shapes the design.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from statsmodels.tsa.stattools import acf

from . import config, data, features, split
from . import plotstyle as ps

ps.apply()
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ORDER = ["Working day", "Weekend", "Festival"]


def note(fig_id, question, finding, decision):
    return {"figure": fig_id, "question": question, "finding": finding, "modelling_decision": decision}


def titled(ax, title, sub):
    n = ps.subtitle(ax, sub)
    ax.set_title(title, pad=12 + 12 * n)


def calendar_residual(df, keys=("hour", "dow", "month")):
    """Load minus the mean load of the same calendar cell (e.g. same hour, weekday, month).
    What remains is the part of demand the calendar can NOT explain."""
    cell_mean = df.groupby(list(keys))["load_mw"].transform("mean")
    return df["load_mw"] - cell_mean


def load_context():
    """Everything the figures need, built once."""
    raw = data.load_raw()
    hourly = data.load_clean()
    feats = features.build_features(hourly)
    sp = split.chronological_split(feats)
    obs = hourly[~hourly["is_missing_raw"]].copy()          # observed values only
    obs["hour"], obs["dow"], obs["month"] = obs.index.hour, obs.index.dayofweek, obs.index.month
    train_obs = obs[obs.index <= sp.train.index.max()].copy()
    train_obs["resid"] = calendar_residual(train_obs)
    return SimpleNamespace(raw=raw, hourly=hourly, feats=feats, sp=sp, obs=obs, train_obs=train_obs,
                           TRAIN_END=sp.train.index.max(), VAL_END=sp.val.index.max())

def f01_timeline(ctx):
    """F01."""
    obs, TRAIN_END, VAL_END = ctx.obs, ctx.TRAIN_END, ctx.VAL_END
    daily = obs["load_mw"].resample("D").mean()
    roll = daily.rolling(30, center=True, min_periods=15).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    spans = [("train", daily.index.min(), TRAIN_END), ("val", TRAIN_END, VAL_END), ("test", VAL_END, daily.index.max())]
    for name, a, b in spans:
        ax.axvspan(a, b, color=ps.SPLIT_FILL[name], zorder=0, lw=0)
        ax.text(a + (b - a) / 2, 0.03, name.capitalize(), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9, color=ps.TEXT_2, fontweight="bold")
    ax.plot(daily.index, daily, color=ps.BLUE_RAMP[3], lw=0.8, label="Daily mean")
    ax.plot(roll.index, roll, color=ps.C[0], lw=2, label="30-day rolling mean")
    ax.set_ylabel("Load (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, 0.98))
    peak_months = obs["load_mw"].resample("ME").mean()
    pk = peak_months.idxmax()
    titled(ax, "Fig 1. Hourly load, Apr 2023 – Jan 2026",
           f"Strong annual cycle: summer peak (max monthly mean {peak_months.max():,.0f} MW, {pk:%b %Y}) "
           f"vs winter trough (~{peak_months.min():,.0f} MW). Level stepped up in 2024, then plateaued in 2025.")
    ps.save(fig, "F01_timeline_with_split")
    fy = obs.assign(fy=np.where(obs.index.month >= 4, obs.index.year, obs.index.year - 1))
    fy_mean = fy[fy.index < "2025-04-01"].groupby("fy")["load_mw"].mean()
    return note("F01", "How does load evolve and where do the splits fall?",
         f"Annual cycle dominates (monthly mean {peak_months.min():,.0f}–{peak_months.max():,.0f} MW). "
         f"Full fiscal-year means: FY23-24 {fy_mean.get(2023, np.nan):,.0f} MW, FY24-25 {fy_mean.get(2024, np.nan):,.0f} MW. "
         f"Val (Mar–Jun 2025) sits in the hot season, which is why its mean is higher than Train's.",
         "Correction to the Phase 1 note: the Train/Val gap is seasonal, not a growth trend. "
         "Month/seasonal features are needed; trees should cope because Train contains two summers.")


def f02_daily_profile(ctx):
    """F02."""
    obs = ctx.obs
    prof = obs.assign(day_type=np.select(
        [obs["holiday_type"] == "Festival", obs["dow"] >= 5], ["Festival", "Weekend"], "Working day"))
    prof_tbl = prof.groupby(["day_type", "hour"])["load_mw"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = ORDER
    for i, col in enumerate(order):
        ax.plot(prof_tbl.index, prof_tbl[col], color=ps.C[i], label=col, marker="o", ms=3)
        ax.annotate(col, (23, prof_tbl[col].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    color=ps.TEXT, fontsize=9, va="center")
    ax.set_xlim(0, 25.5)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean load (MW)")
    ax.legend(loc="upper left")
    wd = prof_tbl["Working day"]
    peak_h, trough_h = wd.idxmax(), wd.idxmin()
    we_gap = (1 - prof_tbl["Weekend"].mean() / wd.mean()) * 100
    titled(ax, "Fig 2. Average daily load profile",
           f"Working-day peak at {peak_h}:00, trough at {trough_h}:00 (peak/trough = {wd.max() / wd.min():.2f}×). "
           f"Weekends run {we_gap:.1f}% lower on average.")
    ps.save(fig, "F02_daily_profile")
    return note("F02", "What does a typical day look like?",
         f"Peak at {peak_h}:00, trough at {trough_h}:00, peak/trough ratio {wd.max() / wd.min():.2f}. "
         f"Weekend {we_gap:.1f}% below working days; festival days below both.",
         "Hour-of-day must be encoded richly: one-hot + sin/cos for linear/MLP, raw hour for trees. Keep is_weekend.")


def f03_heatmap_hour_month(ctx):
    """F03."""
    obs = ctx.obs
    hm = obs.pivot_table(index="hour", columns="month", values="load_mw", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(hm.values, aspect="auto", cmap=ps.SEQ, origin="lower")
    ax.set_xticks(range(12), MONTHS)
    ax.set_yticks(range(0, 24, 3))
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Mean load (MW)", color=ps.TEXT_2)
    cb.outline.set_visible(False)
    peak_hour_by_month = hm.idxmax()
    titled(ax, "Fig 3. Mean load by hour and month",
           f"The daily shape flips with season: winter peaks at {peak_hour_by_month[1]}:00, Apr–May at "
           f"{peak_hour_by_month[5]}:00, Jun–Sep late at night ({peak_hour_by_month[6:10].min()}–{peak_hour_by_month[6:10].max()}:00).")
    ps.save(fig, "F03_heatmap_hour_month")
    return note("F03", "Does the daily shape change with season?",
         f"Yes, strongly. Peak hour by month: {', '.join(f'{MONTHS[m-1]} {h}:00' for m, h in peak_hour_by_month.items())}. "
         f"Winter = morning peak, Jun–Sep = late-night peak. Fig 2's pooled profile hides this.",
         "Hour×month interaction: an additive linear model can't capture it, but trees and the MLP can. "
         "Expect a clear gap between linear and non-linear models.")


def f04_year_on_year(ctx):
    """F04."""
    obs = ctx.obs
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mm = obs.groupby([obs.index.year, "month"])["load_mw"].mean().unstack(0)
    for i, yr in enumerate([2023, 2024, 2025]):
        s = mm[yr].dropna()
        ax.plot(s.index, s.values, color=ps.C[i], marker="o", ms=4, label=str(yr))
        ax.annotate(str(yr), (s.index[-1], s.values[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=ps.TEXT, va="center")
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_xlim(0.6, 12.9)
    ax.set_ylabel("Monthly mean load (MW)")
    ax.legend(loc="upper left")
    c1 = mm[[2023, 2024]].dropna()
    yoy1 = (c1[2024] / c1[2023] - 1).mean() * 100
    c2 = mm[[2024, 2025]].dropna()
    yoy = (c2[2025] / c2[2024] - 1).mean() * 100
    titled(ax, "Fig 4. Monthly mean load by year",
           f"Same seasonal shape every year. 2024 ran {yoy1:+.1f}% above 2023 (Apr–Dec), "
           f"then 2025 was flat ({yoy:+.1f}% vs 2024): a step up, not a steady trend.")
    ps.save(fig, "F04_year_on_year")
    return note("F04", "Is there a growth trend the models must extrapolate?",
         f"2024 vs 2023: {yoy1:+.1f}% (Apr–Dec); 2025 vs 2024: {yoy:+.1f}%. A level shift in 2024, then a plateau.",
         "No de-trending step: Train already contains the higher 2024 level, and the lags and rolling means "
         "carry the recent level into each forecast. Test (H2 2025) sits on the plateau.")


def f05_weather_confounding(ctx):
    """F05."""
    train_obs = ctx.train_obs
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex="row")
    for r, (var, unit) in enumerate([("temp", "°C"), ("humidity", "%")]):
        bins = pd.cut(train_obs[var], 20)
        counts = bins.value_counts()
        keep = train_obs[bins.isin(counts[counts >= 100].index)]          # drop sparse, noisy bins
        kb = bins[keep.index].cat.remove_unused_categories()
        g_raw = keep.groupby(kb, observed=True)["load_mw"]
        g_res = keep.groupby(kb, observed=True)["resid"]
        centers = [b.mid for b in g_raw.mean().index]
        for c, (g, lab) in enumerate([(g_raw, "Load (MW)"), (g_res, "Calendar-adjusted load (MW)")]):
            ax = axes[r, c]
            q25, q75, m = g.quantile(0.25), g.quantile(0.75), g.mean()
            ax.fill_between(centers, q25.values, q75.values, color=ps.BLUE_RAMP[1], lw=0, label="Interquartile range")
            ax.plot(centers, m.values, color=ps.C[0], marker="o", ms=4, label="Mean")
            if c == 1:
                ax.axhline(0, color=ps.MUTED, lw=1, ls="--")
            ax.set_ylabel(lab)
            if True:
                ax.set_xlabel(f"{'Temperature' if var == 'temp' else 'Relative humidity'} ({unit})")
            rho = train_obs[var].corr(train_obs["load_mw" if c == 0 else "resid"])
            ax.set_title(f"{'Temperature' if var == 'temp' else 'Humidity'} vs "
                         f"{'raw load' if c == 0 else 'adjusted load'}   (r = {rho:+.3f})", fontsize=10, pad=6)
    axes[0, 0].legend(loc="upper left")
    r_t_raw = train_obs["temp"].corr(train_obs["load_mw"])
    r_t_res = train_obs["temp"].corr(train_obs["resid"])
    r_h_raw = train_obs["humidity"].corr(train_obs["load_mw"])
    r_h_res = train_obs["humidity"].corr(train_obs["resid"])
    fig.suptitle("Fig 5. Weather vs load: raw (left) and after removing the hour × weekday × month pattern (right)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12, color=ps.TEXT)
    fig.text(0.01, 0.935, f"The raw correlation (temperature r = {r_t_raw:+.2f}) disappears once the calendar is removed "
             f"(r = {r_t_res:+.3f}): temperature only proxies for time of day and season.  Train period only.",
             fontsize=9, color=ps.TEXT_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    ps.save(fig, "F05_weather_confounding")
    return note("F05", "Does weather drive load beyond the calendar?",
         f"Raw r: temperature {r_t_raw:+.2f}, humidity {r_h_raw:+.2f}. After calendar adjustment: "
         f"{r_t_res:+.3f} and {r_h_res:+.3f}, i.e. no independent effect.",
         "Keep the weather features (the ablation will test them), but expect ≈0 gain. "
         "Consistent with v1, where adding weather did not improve MAE. Report it as a data limitation.")


def f06_weather_label(ctx):
    """F06."""
    raw, train_obs = ctx.raw, ctx.train_obs
    w = raw["weather"]
    levels = features.WEATHER_LEVELS
    trans = pd.crosstab(w.shift(1), w, normalize="index").reindex(index=levels, columns=levels)
    marg = w.value_counts(normalize=True).reindex(levels)
    p_same = (w == w.shift(1)).mean()
    p_indep = (marg ** 2).sum()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.imshow(trans.values, cmap=ps.SEQ, vmin=0, vmax=0.6)
    for i in range(4):
        for j in range(4):
            v = trans.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.35 else ps.TEXT)
    ax.set_xticks(range(4), levels)
    ax.set_yticks(range(4), levels)
    ax.set_xlabel("Weather at hour h")
    ax.set_ylabel("Weather at hour h−1")
    ax.grid(False)
    ax.set_title("P(next | current): rows look alike", fontsize=10, pad=6)
    ax = axes[1]
    eff = train_obs.groupby("weather")["resid"].agg(["mean", "sem"]).reindex(levels)
    ax.bar(range(4), eff["mean"], yerr=1.96 * eff["sem"], color=ps.C[0], width=0.6,
           error_kw={"ecolor": ps.TEXT_2, "lw": 1, "capsize": 3})
    ax.axhline(0, color=ps.MUTED, lw=1)
    ax.set_xticks(range(4), [f"{l}\n{v:+.1f} MW" for l, v in zip(levels, eff["mean"].round(1) + 0.0)])
    ax.set_ylabel("Calendar-adjusted load (MW)")
    ax.set_title("Effect on load beyond calendar (95% CI)", fontsize=10, pad=6)
    fig.suptitle("Fig 6. The weather-condition label", x=0.01, ha="left", fontweight="bold", fontsize=12, color=ps.TEXT)
    fig.text(0.01, 0.9, f"Label repeats from one hour to the next {p_same:.0%} of the time vs {p_indep:.0%} if random; "
             f"its load effect is at most {eff['mean'].abs().max():.1f} MW (≈{eff['mean'].abs().max() / train_obs['load_mw'].mean():.1%} of mean load).",
             fontsize=9, color=ps.TEXT_2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    ps.save(fig, "F06_weather_label")
    return note("F06", "Is the weather label informative?",
         f"Hour-to-hour persistence {p_same:.1%} vs {p_indep:.1%} expected if independent; adjusted effect between "
         f"{eff['mean'].min():+.1f} and {eff['mean'].max():+.1f} MW.",
         "Only seasonal information (more rain in Jun–Sep). Lagged one-hot kept for completeness; Lasso should zero it out.")


def f07_day_type_effect(ctx):
    """F07."""
    train_obs = ctx.train_obs
    order = ORDER
    fig, ax = plt.subplots(figsize=(8, 4))
    tr_dt = train_obs.assign(day_type=np.select(
        [train_obs["holiday_type"] == "Festival", train_obs["dow"] >= 5], ["Festival", "Weekend"], "Working day"))
    # For this figure adjust by hour x month only (not weekday) so the weekend effect stays visible.
    hm_mean = tr_dt.groupby(["hour", "month"])["load_mw"].transform("mean")
    tr_dt["resid_hm"] = tr_dt["load_mw"] - hm_mean
    eff = tr_dt.groupby("day_type")["resid_hm"].agg(["mean", "sem", "size"]).reindex(order)
    ax.barh(range(3), eff["mean"], xerr=1.96 * eff["sem"], color=ps.C[0], height=0.55,
            error_kw={"ecolor": ps.TEXT_2, "lw": 1, "capsize": 3})
    ax.axvline(0, color=ps.MUTED, lw=1)
    ax.set_yticks(range(3), [f"{k}\n(n = {n:,} h)" for k, n in zip(order, eff["size"])])
    ax.invert_yaxis()
    ax.set_xlabel("Load relative to same hour & month average (MW)")
    for i, (v, se) in enumerate(zip(eff["mean"], eff["sem"])):
        edge = v + np.sign(v) * (1.96 * se + 6)
        ax.text(edge, i, f"{v:+.0f} MW", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=ps.TEXT)
    ax.set_xlim(eff["mean"].min() * 1.45, max(eff["mean"].max() * 1.9, 60))
    fest_pct = eff.loc["Festival", "mean"] / train_obs["load_mw"].mean() * 100
    titled(ax, "Fig 7. Day-type effect after removing the hour × month pattern",
           f"Festival hours run {eff.loc['Festival', 'mean']:+.0f} MW ({fest_pct:+.1f}%) vs normal; "
           f"weekends {eff.loc['Weekend', 'mean']:+.0f} MW.  Train period only.")
    ps.save(fig, "F07_day_type_effect")
    return note("F07", "Do festivals and weekends change load?",
         f"Festival {eff.loc['Festival', 'mean']:+.0f} MW ({fest_pct:+.1f}%), weekend {eff.loc['Weekend', 'mean']:+.0f} MW, "
         f"working day {eff.loc['Working day', 'mean']:+.0f} MW (hour×month adjusted, train).",
         "Keep is_festival and is_weekend: small but real effects. Festivals are rare (≈7% of hours), "
         "so expect a small overall gain that is larger on festival hours.")


def f08_autocorrelation(ctx):
    """F08."""
    hourly, TRAIN_END = ctx.hourly, ctx.TRAIN_END
    y_tr = hourly.loc[:TRAIN_END, "load_mw"]
    ac = acf(y_tr, nlags=340, fft=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.vlines(range(len(ac)), 0, ac, color=ps.C[0], lw=0.9)
    for k in (1, 24, 168, 336):
        ax.plot(k, ac[k], "o", ms=7, color=ps.C[1], mec=ps.SURFACE, mew=1.5, zorder=3)
        ax.annotate(f"lag {k}\n{ac[k]:.2f}", (k, ac[k]), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8.5, color=ps.TEXT)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_ylim(min(0, ac.min()) - 0.05, 1.18)
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336])
    titled(ax, "Fig 8. Autocorrelation of hourly load (train period)",
           f"Daily cycle dominates (lag 1 = {ac[1]:.2f}, lag 24 = {ac[24]:.2f}). The weekly cycle is weak: daily peaks "
           f"stop decaying between 120 h and 168 h (≈{ac[168]:.2f}) instead of standing out.")
    ps.save(fig, "F08_autocorrelation")
    return note("F08", "Which past hours carry information?",
         f"ACF lag1 {ac[1]:.3f}, lag24 {ac[24]:.3f}, lag144 {ac[144]:.3f}, lag168 {ac[168]:.3f}, lag192 {ac[192]:.3f}. "
         f"Daily cycle strong; weekly cycle weak (168 h barely above 144 h).",
         "Justifies lags 1–3, 24 and 48. Lag 168 is kept but expected to add little; Lasso will tell. "
         "Lag 1 alone should give a strong naive baseline (Phase 3), and naive-24 should beat naive-168.")


def f09_feature_correlation(ctx):
    """F09."""
    sp = ctx.sp
    cols = [c for c in features.FEATURE_SETS["tree"] if c not in ("dayofyear",)]
    corr = sp.train[cols + [config.TARGET]].corr()[config.TARGET].drop(config.TARGET)
    grp = {c: ("Load history" if c in features.LOAD else "Weather" if c in features.WEATHER else "Calendar")
           for c in cols}
    corr = corr.reindex(corr.abs().sort_values().index)
    gcol = {"Load history": ps.C[0], "Calendar": ps.C[1], "Weather": ps.C[2]}
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.barh(range(len(corr)), corr.values, color=[gcol[grp[c]] for c in corr.index], height=0.7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8.5)
    ax.axvline(0, color=ps.MUTED, lw=1)
    ax.set_xlabel("Pearson correlation with target (train)")
    ax.legend(handles=[Patch(color=c, label=g) for g, c in gcol.items()], loc="lower right")
    titled(ax, "Fig 9. Linear correlation of each feature with next-hour load",
           "Load-history features dominate (|r| > 0.8). Weather and calendar r values here are mostly seasonal proxies (see Fig 5).")
    ps.save(fig, "F09_feature_correlation")
    top = corr.abs().sort_values(ascending=False)
    return note("F09", "Which features relate linearly to the target?",
         f"Top: {', '.join(f'{k} ({corr[k]:+.2f})' for k in top.index[:4])}. Lowest: "
         f"{', '.join(top.index[-3:])}.",
         "Strong collinearity among load lags (lag1, lag2, lag3, roll_mean_24): plain OLS coefficients will be "
         "unstable. This motivates Ridge (L2) and Lasso (L1) in Phase 4.")


def f10_missing_hours(ctx):
    """F10."""
    raw, hourly = ctx.raw, ctx.hourly
    gaps = data.gap_report(hourly)
    miss_m = hourly["is_missing_raw"].resample("ME").sum()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(miss_m.index, miss_m.values, width=20, color=ps.C[0])
    ax.set_ylabel("Missing hours")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    long_g = gaps[gaps.hours > config.MAX_FFILL_HOURS]
    titled(ax, "Fig 10. Missing hours per month in the raw data",
           f"{len(raw):,} of {len(hourly):,} hours present ({hourly['is_missing_raw'].mean():.1%} missing) in {len(gaps)} gaps; "
           f"{len(gaps) - len(long_g)} gaps ≤ 3 h, {len(long_g)} longer (max {gaps.hours.max()} h).")
    ps.save(fig, "F10_missing_hours")
    return note("F10", "How complete is the data?",
         f"{hourly['is_missing_raw'].sum()} missing hours ({hourly['is_missing_raw'].mean():.1%}) in {len(gaps)} gaps; "
         f"longest {gaps.hours.max()} h starting {gaps.loc[gaps.hours.idxmax(), 'start']:%d %b %Y}.",
         "Phase 1 policy holds: forward-fill ≤3 h, seasonal fill for long gaps (history only), "
         "imputed hours never scored.")


def f11_load_distribution(ctx):
    """F11."""
    obs, train_obs = ctx.obs, ctx.train_obs
    y = obs["load_mw"]
    mu, sd = train_obs["load_mw"].mean(), train_obs["load_mw"].std()
    n_out = int(((y - mu).abs() > 3 * sd).sum())
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(y, bins=60, color=ps.C[0], edgecolor=ps.SURFACE, linewidth=0.6)
    for k in (-3, 3):
        ax.axvline(mu + k * sd, color=ps.C[1], lw=1.5, ls="--")
    ax.text(mu + 3 * sd, ax.get_ylim()[1] * 0.92, "  +3σ", color=ps.TEXT, fontsize=9)
    ax.text(mu - 3 * sd, ax.get_ylim()[1] * 0.92, "  −3σ", color=ps.TEXT, fontsize=9, ha="left")
    ax.set_xlabel("Hourly load (MW)")
    ax.set_ylabel("Hours")
    titled(ax, "Fig 11. Distribution of hourly load",
           f"Range {y.min():,.0f}–{y.max():,.0f} MW. The small hump near 1,900 MW is winter nights (Nov–Mar, 01–04 h).  Only {n_out} hours fall beyond ±3σ "
           f"(train μ, σ), all on four June heatwave days — real demand, not errors.")
    ps.save(fig, "F11_load_distribution")
    return note("F11", "Are there outliers to remove?",
         f"{n_out} hours beyond ±3σ (μ={mu:,.0f}, σ={sd:,.0f}); all above +3σ, on 18–19 Jun 2024 and 12–13 Jun 2025 (heatwave days).",
         "Second mode near 1,900 MW = winter night hours (a distinct low regime, not an error). Do NOT delete outliers. (1) Deleting rows breaks the hourly grid that the lags rely on. "
         "(2) Peaks are exactly what a utility needs forecast well. Peak accuracy is analysed separately in Phase 7.")


ALL = [f01_timeline, f02_daily_profile, f03_heatmap_hour_month, f04_year_on_year,
       f05_weather_confounding, f06_weather_label, f07_day_type_effect, f08_autocorrelation,
       f09_feature_correlation, f10_missing_hours, f11_load_distribution]


def run_all(ctx=None):
    ctx = ctx or load_context()
    findings = pd.DataFrame([f(ctx) for f in ALL])
    findings.to_csv(config.TABLE_DIR / "phase2_eda_findings.csv", index=False)
    return findings
