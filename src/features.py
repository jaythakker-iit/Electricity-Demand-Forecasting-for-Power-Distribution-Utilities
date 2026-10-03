"""Phase 1b: feature engineering.

Convention (this fixes the v1 off-by-one bug)
---------------------------------------------
Each row is indexed by the TARGET hour h. `target_mw` is the load at h.
Every feature is named relative to h:

    load_lag_1   = load at h-1   (the latest value known when we forecast)
    load_lag_24  = load at h-24  (same hour yesterday)  -> seasonal-naive-24
    load_lag_168 = load at h-168 (same hour last week)  -> seasonal-naive-168

Causality rule: a feature may use load/weather up to h-1 only.
Calendar facts about h (hour, weekday, festival) are allowed because they
are known in advance.
Columns prefixed `fx_` hold weather AT hour h. They are only used in the
'perfect weather forecast' what-if experiment and never in the main models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

WEATHER_LEVELS = ["Clear", "Cloudy", "Rain", "Storm"]  # 'Clear' = reference level

LOAD_LAGS = [1, 2, 3, 24, 48, 168]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: output of data.to_hourly (regular hourly index)."""
    if pd.infer_freq(df.index) != "h":
        raise ValueError("Input must be on a regular hourly grid")

    f = pd.DataFrame(index=df.index)
    idx = df.index

    # ---------------- calendar of the target hour (known in advance)
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["dayofyear"] = idx.dayofyear
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["is_festival"] = (df["holiday_type"] == "Festival").astype(int)
    for name, period in [("hour", 24), ("dow", 7), ("month", 12)]:
        f[f"{name}_sin"] = np.sin(2 * np.pi * f[name] / period)
        f[f"{name}_cos"] = np.cos(2 * np.pi * f[name] / period)
    for h in range(1, 24):
        f[f"hour_{h}"] = (f["hour"] == h).astype(int)
    for d in range(1, 7):
        f[f"dow_{d}"] = (f["dow"] == d).astype(int)

    # ---------------- load history (strictly <= h-1)
    load = df["load_mw"]
    for k in LOAD_LAGS:
        f[f"load_lag_{k}"] = load.shift(k)
    past = load.shift(1)
    f["load_roll_mean_24"] = past.rolling(24).mean()
    f["load_roll_std_24"] = past.rolling(24).std()
    f["load_roll_mean_168"] = past.rolling(168).mean()
    f["load_ramp_1"] = load.shift(1) - load.shift(2)

    # ---------------- weather observed at h-1 (causal)
    f["temp_lag_1"] = df["temp"].shift(1)
    f["humidity_lag_1"] = df["humidity"].shift(1)
    f["temp_roll_mean_24"] = df["temp"].shift(1).rolling(24).mean()
    w_prev = df["weather"].shift(1)
    for lvl in WEATHER_LEVELS[1:]:
        f[f"weather_lag_1_{lvl.lower()}"] = (w_prev == lvl).astype(int).where(w_prev.notna())

    # ---------------- weather AT h (what-if experiment only)
    f["fx_temp"] = df["temp"]
    f["fx_humidity"] = df["humidity"]
    for lvl in WEATHER_LEVELS[1:]:
        f[f"fx_weather_{lvl.lower()}"] = (df["weather"] == lvl).astype(int).where(df["weather"].notna())

    # ---------------- target and bookkeeping
    f[config.TARGET] = load
    f["target_is_imputed"] = df["is_imputed"].astype(bool)

    # Drop rows we cannot use: missing target, imputed target, or any missing feature
    # (the first 168 h and hours next to long gaps).
    n0 = len(f)
    f = f[f[config.TARGET].notna() & ~f["target_is_imputed"]]
    f = f.dropna()
    f.attrs["rows_dropped"] = n0 - len(f)
    return f.drop(columns=["target_is_imputed"])


# ------------------------------------------------------------ feature sets
CALENDAR_RAW = ["hour", "dow", "month", "dayofyear"]
CALENDAR_CYCLIC = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
CALENDAR_ONEHOT = [f"hour_{h}" for h in range(1, 24)] + [f"dow_{d}" for d in range(1, 7)]
FLAGS = ["is_weekend"]
FESTIVAL = ["is_festival"]
LOAD = [f"load_lag_{k}" for k in LOAD_LAGS] + [
    "load_roll_mean_24", "load_roll_std_24", "load_roll_mean_168", "load_ramp_1"]
WEATHER = ["temp_lag_1", "humidity_lag_1", "temp_roll_mean_24"] + [
    f"weather_lag_1_{l.lower()}" for l in WEATHER_LEVELS[1:]]
WEATHER_AT_H = ["fx_temp", "fx_humidity"] + [f"fx_weather_{l.lower()}" for l in WEATHER_LEVELS[1:]]

# load_ramp_1 = load_lag_1 - load_lag_2 is an EXACT linear combination of two other
# columns. Useful for trees (they cannot form differences themselves), but for linear
# models it makes X rank-deficient (cond(X) ~ 1e16), so OLS has no unique solution.
# Found by the Phase 2 evaluator (VIF check in notebooks/01_EDA.ipynb).
LOAD_LINEAR = [c for c in LOAD if c != "load_ramp_1"]

FEATURE_SETS = {
    # Trees split on raw integers fine; cyclic features added for smooth boundaries.
    "tree": CALENDAR_RAW + CALENDAR_CYCLIC + FLAGS + FESTIVAL + LOAD + WEATHER,
    # Linear models and MLP: no raw integer calendar (a line through hour 0..23 is meaningless).
    # Hour & weekday as one-hot (most flexible); month as sin/cos. Exact dependencies removed
    # (evaluator full-rank test): hour/dow sin-cos are linear combinations of their one-hots
    # (given an intercept), and is_weekend = dow_5 + dow_6.
    "linear": ["month_sin", "month_cos"] + CALENDAR_ONEHOT + FESTIVAL + LOAD_LINEAR + WEATHER,
}
FEATURE_SETS["mlp"] = FEATURE_SETS["linear"]

# Ablation groups for Phase 7 (added cumulatively)
ABLATION = {
    "A_load_calendar": CALENDAR_RAW + CALENDAR_CYCLIC + FLAGS + LOAD,
    "B_+weather": CALENDAR_RAW + CALENDAR_CYCLIC + FLAGS + LOAD + WEATHER,
    "C_+festival": CALENDAR_RAW + CALENDAR_CYCLIC + FLAGS + LOAD + WEATHER + FESTIVAL,
    "D_+weather_at_h (what-if)": CALENDAR_RAW + CALENDAR_CYCLIC + FLAGS + LOAD + WEATHER + FESTIVAL + WEATHER_AT_H,
}


# ------------------------------------------------------------ hour x month interactions (Phase 4)
def linear_design(df: pd.DataFrame, interactions: bool = False) -> pd.DataFrame:
    """Design matrix for linear models.

    interactions=False -> FEATURE_SETS['linear'] (47 columns).
    interactions=True  -> month as one-hot (instead of sin/cos) plus hour x month
                          interaction dummies, so a linear model can give EACH
                          (hour, month) cell its own level, i.e. let the daily shape
                          change with season (EDA Fig 3). Reference levels (hour 0,
                          month 1) are dropped so the matrix stays full rank.
    """
    base = df[FEATURE_SETS["linear"]]
    if not interactions:
        return base
    base = base.drop(columns=["month_sin", "month_cos"])
    month_oh = pd.DataFrame({f"month_{m}": (df["month"] == m).astype(int) for m in range(2, 13)},
                            index=df.index)
    inter = pd.DataFrame({f"h{h}_m{m}": ((df["hour"] == h) & (df["month"] == m)).astype(int)
                          for h in range(1, 24) for m in range(2, 13)}, index=df.index)
    return pd.concat([base, month_oh, inter], axis=1)
