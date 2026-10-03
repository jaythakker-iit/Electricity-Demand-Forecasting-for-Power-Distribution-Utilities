"""Phase 1c: chronological Train / Validation / Test split.

    |------------- Train 70% -------------|-- Val 10% --|------ Test 20% ------|
    past ------------------------------------------------------------> future

* Train: fit model parameters (weights, tree splits).
* Val:   choose things the optimiser can't (early-stopping epoch, MLP config).
* Test:  touched once, at the very end, for the numbers in the report.
Hyperparameter CV (Ridge alpha, XGBoost grid) uses TimeSeriesSplit inside
Train+Val ("development" set), so it never looks at Test either.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def dev(self) -> pd.DataFrame:
        """Train + Val, used for CV-based tuning and final refit of sklearn models."""
        return pd.concat([self.train, self.val])

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "val", "test"):
            part = getattr(self, name)
            rows.append({"split": name, "rows": len(part),
                         "start": part.index.min(), "end": part.index.max(),
                         "mean_mw": part[config.TARGET].mean()})
        return pd.DataFrame(rows)


def chronological_split(df: pd.DataFrame,
                        train_frac: float = config.TRAIN_FRAC,
                        val_frac: float = config.VAL_FRAC) -> Split:
    if not df.index.is_monotonic_increasing:
        raise ValueError("Data must be sorted by time before splitting")
    n = len(df)
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    return Split(df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:])


def xy(part: pd.DataFrame, features: list[str]):
    return part[features], part[config.TARGET]
