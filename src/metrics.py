"""Evaluation metrics shared by every model (one definition = fair comparison)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_true <= 0):
        raise ValueError("MAPE undefined: non-positive actual load")
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": r2_score(y_true, y_pred),
        "Bias": float(np.mean(y_true - y_pred)),   # >0 = under-forecast on average
    }


def metrics_row(model: str, y_true, y_pred, **extra) -> dict:
    return {"Model": model, **evaluate(y_true, y_pred), **extra}


def to_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).round(4)
