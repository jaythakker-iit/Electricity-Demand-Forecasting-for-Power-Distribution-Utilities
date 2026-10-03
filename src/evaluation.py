"""Shared evaluation utilities: every model's predictions are stored in ONE long format
and scored by ONE function, so all models are compared on exactly the same rows.

Long format (one row per model x timestamp):
    timestamp | split ('val'/'test') | model | y_true | y_pred
"""
from __future__ import annotations

import pandas as pd

from . import config
from .metrics import evaluate

PRED_DIR = config.RESULTS_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)


def to_long(model: str, split_name: str, y_true: pd.Series, y_pred) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": y_true.index, "split": split_name, "model": model,
                         "y_true": y_true.values, "y_pred": pd.Series(y_pred, index=y_true.index).values})


def save_predictions(df: pd.DataFrame, phase_tag: str) -> None:
    df.to_csv(PRED_DIR / f"predictions_{phase_tag}.csv", index=False)


def load_all_predictions() -> pd.DataFrame:
    files = sorted(PRED_DIR.glob("predictions_*.csv"))
    return pd.concat([pd.read_csv(f, parse_dates=["timestamp"]) for f in files], ignore_index=True)


def peak_threshold(train_target: pd.Series, q: float = 0.90) -> float:
    """Peak = top 10% of load, threshold from TRAIN only (never from the test set)."""
    return float(train_target.quantile(q))


def score(long_df: pd.DataFrame, peak_thr: float | None = None) -> pd.DataFrame:
    """Metrics per (split, model), optionally also on peak hours only."""
    rows = []
    for (sp, model), g in long_df.groupby(["split", "model"], sort=False):
        rows.append({"split": sp, "model": model, "subset": "all", "n": len(g),
                     **evaluate(g["y_true"], g["y_pred"])})
        if peak_thr is not None:
            gp = g[g["y_true"] >= peak_thr]
            if len(gp):
                rows.append({"split": sp, "model": model, "subset": "peak", "n": len(gp),
                             **evaluate(gp["y_true"], gp["y_pred"])})
    return pd.DataFrame(rows)


def add_skill(table: pd.DataFrame, reference: str = "Naive-1 (persistence)") -> pd.DataFrame:
    """Skill score = 1 - MAE_model / MAE_reference (per split & subset). >0 = beats the reference."""
    t = table.copy()
    ref = t[t["model"] == reference].set_index(["split", "subset"])["MAE"]
    t["skill_vs_naive1"] = 1 - t["MAE"] / t.set_index(["split", "subset"]).index.map(ref)
    return t
