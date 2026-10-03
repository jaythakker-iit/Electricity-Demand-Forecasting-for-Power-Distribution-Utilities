"""Phase 5: tree ensembles — Decision Tree, Random Forest (bagging), XGBoost (boosting).

Course concepts
---------------
Decision tree : greedy recursive splits minimising squared error in each child; depth / leaf size
                control complexity (bias-variance).
Random forest : bagging — average B trees grown on bootstrap samples with random feature subsets;
                averaging de-correlated trees cuts variance without raising bias much.
XGBoost       : boosting — gradient descent in function space. Tree m is fitted to the negative
                gradient of the loss (for squared error: the current residuals), scaled by the
                learning rate eta:  F_m(x) = F_{m-1}(x) + eta * f_m(x).  L1/L2 penalties on leaf
                weights (reg_alpha, reg_lambda) are the same regularisation ideas as Lasso/Ridge.

Target transform (a hyperparameter chosen by CV)
------------------------------------------------
'level' : predict y_h directly.
'delta' : predict the change  d_h = y_h - y_{h-1}  and add back load_lag_1.
Trees output averages of training targets, so a 'level' tree cannot forecast above the highest
load seen in training; the change is bounded and stationary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from . import config

LAG1 = "load_lag_1"
N_SPLITS = 5


class TargetMode(BaseEstimator, RegressorMixin):
    """Wraps a regressor so that the target can be the level or the hourly change.
    Behaves like a normal sklearn estimator, so `target` can be tuned in a CV search."""

    def __init__(self, regressor=None, target: str = "level"):
        self.regressor = regressor
        self.target = target

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        y_t = y - X[LAG1] if self.target == "delta" else y
        self.regressor_.fit(X, y_t)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def predict(self, X):
        p = self.regressor_.predict(X)
        return p + X[LAG1].values if self.target == "delta" else p


def base_models() -> dict:
    s = config.SEED
    return {
        "Decision Tree": DecisionTreeRegressor(random_state=s),
        "Random Forest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=s),
        "XGBoost": XGBRegressor(tree_method="hist", n_jobs=-1, random_state=s, verbosity=0),
    }


SEARCH = {
    "Decision Tree": {
        "regressor__max_depth": [6, 8, 10, 12, 15, 20, None],
        "regressor__min_samples_leaf": [1, 5, 10, 20, 50, 100],
        "target": ["level", "delta"],
    },
    "Random Forest": {
        "regressor__max_depth": [12, 18, None],
        "regressor__min_samples_leaf": [1, 3, 10],
        "regressor__max_features": [0.33, 0.6, 1.0],
        "target": ["level", "delta"],
    },
    "XGBoost": {
        "regressor__n_estimators": [300, 600, 1000, 1500],
        "regressor__learning_rate": [0.02, 0.05, 0.1],
        "regressor__max_depth": [4, 6, 8, 10],
        "regressor__min_child_weight": [1, 5, 20],
        "regressor__subsample": [0.7, 0.85, 1.0],
        "regressor__colsample_bytree": [0.6, 0.8, 1.0],
        "regressor__reg_lambda": [0.1, 1, 10],
        "regressor__reg_alpha": [0, 1, 10],
        "target": ["level", "delta"],
    },
}
N_ITER = {"Decision Tree": 30, "Random Forest": 14, "XGBoost": 40}


def tune(name: str, X: pd.DataFrame, y: pd.Series, n_iter: int | None = None) -> RandomizedSearchCV:
    """Randomised search, forward-chaining 5-fold CV, scored by MAE (our headline metric)."""
    search = RandomizedSearchCV(
        TargetMode(base_models()[name]), SEARCH[name], n_iter=n_iter or N_ITER[name],
        cv=TimeSeriesSplit(n_splits=N_SPLITS), scoring="neg_mean_absolute_error",
        random_state=config.SEED, n_jobs=1, refit=True, return_train_score=True)
    search.fit(X, y)
    return search


def cv_table(search: RandomizedSearchCV) -> pd.DataFrame:
    r = pd.DataFrame(search.cv_results_)
    keep = [c for c in r.columns if c.startswith("param_")] + [
        "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score", "mean_fit_time"]
    r = r[keep].copy()
    r["cv_mae"] = -r.pop("mean_test_score")
    r["cv_mae_std"] = r.pop("std_test_score")
    r["train_mae"] = -r.pop("mean_train_score")
    return r.sort_values("rank_test_score")


def importance(model: TargetMode) -> pd.Series:
    """Gain-based importance (XGBoost) / impurity importance (sklearn trees)."""
    reg = model.regressor_
    if isinstance(reg, XGBRegressor):
        g = reg.get_booster().get_score(importance_type="gain")
        s = pd.Series({c: g.get(c, 0.0) for c in model.feature_names_in_})
    else:
        s = pd.Series(reg.feature_importances_, index=model.feature_names_in_)
    return (s / s.sum()).sort_values(ascending=False)
