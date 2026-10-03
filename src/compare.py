"""Phase 7: statistical comparison and error breakdowns.

Diebold–Mariano (DM) test
-------------------------
Loss differential d_t = |e1_t| - |e2_t| (absolute-error loss, matching MAE).
H0: E[d_t] = 0 (equal accuracy). DM = mean(d) / sqrt(LRV / T), where LRV is the long-run
variance of d_t estimated with a Newey-West (Bartlett) kernel, because forecast errors are
autocorrelated in time (EDA Fig 8, Phase 3 residual ACF). Harvey-Leybourne-Newbold small-sample
correction; p-value from Student-t with T-1 df.

Moving-block bootstrap
----------------------
Resample whole blocks of consecutive hours (default 168 h = 1 week) so the bootstrap keeps the
daily/weekly dependence. Gives confidence intervals for MAE and for MAE differences without
assuming independence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

FINAL = ["Naive-1 (persistence)", "Naive-1 + hourly step", "Lasso + hour×month",
         "Random Forest", "XGBoost", "MLP v4 [64,32] Δ-target"]
SHORT = {"Naive-1 (persistence)": "Persistence", "Naive-1 + hourly step": "Step baseline",
         "Lasso + hour×month": "Lasso+h×m", "Random Forest": "Random Forest", "XGBoost": "XGBoost",
         "MLP v4 [64,32] Δ-target": "MLP v4", "Decision Tree": "Decision Tree"}


def wide(pred_long: pd.DataFrame, split: str, models=FINAL) -> pd.DataFrame:
    """timestamp x model matrix of predictions + y_true; asserts identical rows for all models."""
    p = pred_long[(pred_long.split == split) & pred_long.model.isin(models)]
    w = p.pivot_table(index="timestamp", columns="model", values="y_pred")
    y = p.groupby("timestamp")["y_true"].first()
    assert w.notna().all().all(), "models are not scored on identical rows"
    w = w[[m for m in models if m in w.columns]]
    w.insert(0, "y_true", y)
    return w.sort_index()


def newey_west_var(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    T = len(d)
    lrv = d @ d / T
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        lrv += 2 * w * (d[k:] @ d[:-k]) / T
    return lrv


def diebold_mariano(e1, e2, lags: int = 24, h: int = 1) -> dict:
    d = np.abs(np.asarray(e1)) - np.abs(np.asarray(e2))
    T = len(d)
    lrv = newey_west_var(d, lags)
    dm = d.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return {"mean_diff": d.mean(), "dm_stat": stat, "p_value": p, "T": T, "lags": lags}


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (controls family-wise error over many pairwise tests)."""
    order = pvals.sort_values().index
    m = len(pvals)
    adj, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return pd.Series(adj)[pvals.index]


def block_bootstrap(abs_err: pd.DataFrame, block: int = 168, n_boot: int = 2000, seed: int = 42) -> np.ndarray:
    """Return an (n_boot x n_models) array of bootstrap MAEs (moving-block resampling of rows)."""
    rng = np.random.default_rng(seed)
    A = abs_err.values
    T = len(A)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n_boot, n_blocks))
    out = np.empty((n_boot, A.shape[1]))
    offs = np.arange(block)
    for b in range(n_boot):
        idx = (starts[b][:, None] + offs).ravel()[:T]
        out[b] = A[idx].mean(0)
    return out


def breakdown(w: pd.DataFrame, by: pd.Series, models=FINAL) -> pd.DataFrame:
    """MAE (and n) of each model within groups defined by `by` (aligned to w.index)."""
    ae = w[models].sub(w["y_true"], axis=0).abs()
    g = ae.groupby(by.reindex(w.index).values)
    t = g.mean()
    t["n"] = g.size()
    return t
