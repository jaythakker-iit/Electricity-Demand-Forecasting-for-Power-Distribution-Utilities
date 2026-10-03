"""Phase 4: OLS, Ridge (L2) and Lasso (L1) with time-series cross-validation.

Course concepts
---------------
OLS    : min ||y - Xb||^2                      closed form b = (X'X)^-1 X'y
Ridge  : min ||y - Xb||^2 + a ||b||_2^2        closed form b = (X'X + aI)^-1 X'y   (strictly convex)
Lasso  : min (1/2n)||y - Xb||^2 + a ||b||_1    no closed form -> coordinate descent  (convex, sparse)

Features are standardised inside the Pipeline (fit on training rows only), so the
penalty treats every feature on the same scale and no scaling statistics leak
from validation/test rows.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_SPLITS = 5
RIDGE_ALPHAS = np.logspace(-4, 5, 60)          # log-spaced; wide enough that the optimum is not on the edge


def pipe(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def lasso_alphas(X: pd.DataFrame, y: pd.Series, n: int = 60, ratio: float = 1e-6) -> np.ndarray:
    """alpha_max = smallest alpha that sets ALL coefficients to zero: max|X's y_c| / n (standardised X)."""
    Xs = StandardScaler().fit_transform(X)
    a_max = np.max(np.abs(Xs.T @ (y.values - y.mean()))) / len(y)
    return np.logspace(np.log10(a_max), np.log10(a_max * ratio), n)


def make_model(kind: str, alpha: float | None = None):
    if kind == "ols":
        return LinearRegression()
    if kind == "ridge":
        return Ridge(alpha=alpha)
    if kind == "lasso":
        return Lasso(alpha=alpha, max_iter=20000, tol=1e-4)
    raise ValueError(kind)


def cv_curve(X: pd.DataFrame, y: pd.Series, kind: str, alphas, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Forward-chaining CV: fold k trains on everything before its validation block.
    Returns one row per (alpha, fold) with validation MSE and MAE."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr, va) in enumerate(tscv.split(X)):
        m = pipe(make_model(kind, alphas[0]))
        if kind == "lasso":                       # large -> small alpha, warm-started (same optimum, faster)
            m.set_params(model__warm_start=True)
            alphas = sorted(alphas, reverse=True)
        for a in alphas:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m.set_params(model__alpha=a).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict(X.iloc[va])
            err = y.iloc[va].values - p
            rows.append({"alpha": a, "fold": fold, "mse": float(np.mean(err ** 2)),
                         "mae": float(np.mean(np.abs(err)))})
    return pd.DataFrame(rows)


def select_alpha(curve: pd.DataFrame) -> dict:
    """Best alpha = lowest mean CV MSE. Also the '1-SE rule' alpha: the most regularised
    alpha whose mean MSE is within one standard error of the best (a simpler model)."""
    g = curve.groupby("alpha")["mse"].agg(["mean", "std", "count"])
    g["se"] = g["std"] / np.sqrt(g["count"])
    best = g["mean"].idxmin()
    limit = g.loc[best, "mean"] + g.loc[best, "se"]
    one_se = g[g["mean"] <= limit].index.max()
    # Boundary diagnostic: if the best alpha is the smallest one tried, is the curve FLAT there
    # (then "no regularisation needed" is the honest conclusion) or still falling (then widen the grid)?
    at_low_edge = best == g.index.min()
    rel_gap = (g["mean"].iloc[:3].max() - g["mean"].min()) / g["mean"].min()   # spread over the 3 smallest alphas
    return {"best": float(best), "one_se": float(one_se), "table": g,
            "at_low_edge": bool(at_low_edge), "flat_at_low_edge": bool(rel_gap < 1e-3)}


def fold_coefficients(X, y, kind, alpha=None, n_splits=N_SPLITS) -> pd.DataFrame:
    """Standardised coefficients fitted on each CV training window: their spread across
    folds measures how STABLE (low-variance) the estimator is."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out = []
    for fold, (tr, _) in enumerate(tscv.split(X)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = pipe(make_model(kind, alpha)).fit(X.iloc[tr], y.iloc[tr])
        out.append(pd.Series(m.named_steps["model"].coef_, index=X.columns, name=fold))
    return pd.DataFrame(out)


def lasso_path(X, y, alphas) -> pd.DataFrame:
    """Coefficient of every feature at every alpha (rows = alpha), standardised scale."""
    m = pipe(Lasso(max_iter=20000, tol=1e-4, warm_start=True))
    rows = []
    for a in sorted(alphas, reverse=True):              # large -> small, warm-started
        m.set_params(model__alpha=a)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m.fit(X, y)
        rows.append(pd.Series(m.named_steps["model"].coef_, index=X.columns, name=a))
    return pd.DataFrame(rows)


def fit_final(X, y, kind, alpha=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return clone(pipe(make_model(kind, alpha))).fit(X, y)


def window_bias_variance(X, y, alphas, cols, days: int = 30) -> pd.DataFrame:
    """Fair stability test: fit on consecutive EQUAL-SIZE windows of `days` days and forecast the
    next window. For each alpha (0 = OLS):
      coef_spread = sum over `cols` of the std of the standardised coefficient across windows (variance)
      next_mae    = mean MAE on the following window (what the variance/bias mix costs in accuracy)
    Equal sizes matter: with expanding windows a fixed alpha shrinks small windows more, which would
    confound shrinkage with variance."""
    n = days * 24
    k = len(X) // n
    idx = [X.columns.get_loc(c) for c in cols]
    rows = []
    for a in alphas:
        kind = "ols" if a == 0 else "ridge"
        co, err = [], []
        for i in range(k - 1):
            tr, te = slice(i * n, (i + 1) * n), slice((i + 1) * n, min((i + 2) * n, len(X)))
            m = fit_final(X.iloc[tr], y.iloc[tr], kind, None if a == 0 else a)
            co.append(m.named_steps["model"].coef_[idx])
            err.append(np.mean(np.abs(y.iloc[te].values - m.predict(X.iloc[te]))))
        rows.append({"alpha": a, "coef_spread": float(np.array(co).std(0).sum()),
                     "next_window_mae": float(np.mean(err)), "windows": k - 1, "window_days": days})
    return pd.DataFrame(rows)
