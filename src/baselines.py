"""Phase 3: naive baselines. No learning, so they define the minimum bar.

All baselines read the feature table, whose columns are defined relative to the
target hour h (load_lag_k = load at h-k). That convention makes the v1
off-by-one bug (using h-25 for 'yesterday') impossible by construction.
"""
from __future__ import annotations

import pandas as pd

from . import config

NAIVE = {
    "Naive-1 (persistence)": "load_lag_1",      # y_hat(h) = y(h-1)
    "Seasonal naive-24": "load_lag_24",         # same hour yesterday
    "Seasonal naive-168": "load_lag_168",       # same hour last week
}
CLIM_NAME = "Calendar climatology"
STEP_NAME = "Naive-1 + hourly step"
CLIM_KEYS = ["hour", "dow", "month"]


def naive_forecasts(part: pd.DataFrame) -> dict[str, pd.Series]:
    return {name: part[col] for name, col in NAIVE.items()}


class CalendarClimatology:
    """Average load for the same (hour, weekday, month), learned from past data only.
    Falls back to (hour, month), then (hour,) if a cell was never seen."""

    def fit(self, part: pd.DataFrame):
        y = part[config.TARGET]
        self.levels_ = [CLIM_KEYS, ["hour", "month"], ["hour"]]
        self.tables_ = [y.groupby([part[k] for k in keys]).mean() for keys in self.levels_]
        self.fit_end_ = part.index.max()
        return self

    def predict(self, part: pd.DataFrame) -> pd.Series:
        pred = pd.Series(index=part.index, dtype=float)
        for keys, table in zip(self.levels_, self.tables_):
            missing = pred.isna()
            if not missing.any():
                break
            idx = pd.MultiIndex.from_frame(part.loc[missing, keys]) if len(keys) > 1 else part.loc[missing, keys[0]]
            pred[missing] = table.reindex(idx).values
        return pred


class PersistencePlusStep:
    """y_hat(h) = y(h-1) + average change from hour h-1 to hour h, by (hour, month).

    Captures the scheduled step changes in load (e.g. +~400 MW at 09:00) that pure
    persistence always misses. 24 x 12 numbers learned from past data only; no ML."""

    def fit(self, part: pd.DataFrame):
        step = part[config.TARGET] - part["load_lag_1"]
        self.by_hm_ = step.groupby([part["hour"], part["month"]]).mean()
        self.by_h_ = step.groupby(part["hour"]).mean()
        self.fit_end_ = part.index.max()
        return self

    def predict(self, part: pd.DataFrame) -> pd.Series:
        idx = pd.MultiIndex.from_frame(part[["hour", "month"]])
        step = pd.Series(self.by_hm_.reindex(idx).values, index=part.index)
        step = step.fillna(part["hour"].map(self.by_h_))
        return part["load_lag_1"] + step
