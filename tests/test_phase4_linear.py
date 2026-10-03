"""Evaluator gate for Phase 4: linear models are fitted, tuned and scored correctly."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, evaluation as ev, features, split  # noqa: E402
from src import linear_models as lm  # noqa: E402
from src.metrics import evaluate  # noqa: E402


@pytest.fixture(scope="module")
def sp():
    return split.chronological_split(features.build_features(data.load_clean()))


def test_ols_matches_normal_equations(sp):
    """sklearn OLS == b = (X'X)^-1 X'y on centred, standardised data."""
    X = features.linear_design(sp.train, False).astype(float)
    y = sp.train[config.TARGET].values
    m = lm.fit_final(X, sp.train[config.TARGET], "ols")
    Xs = StandardScalerLike(X)
    b = np.linalg.solve(Xs.T @ Xs, Xs.T @ (y - y.mean()))
    np.testing.assert_allclose(m.named_steps["model"].coef_, b, rtol=1e-6, atol=1e-6)


def test_ridge_matches_closed_form(sp):
    """sklearn Ridge == b = (X'X + aI)^-1 X'y (centred, standardised)."""
    X = features.linear_design(sp.train, False).astype(float)
    y = sp.train[config.TARGET].values
    a = 50.0
    m = lm.fit_final(X, sp.train[config.TARGET], "ridge", a)
    Xs = StandardScalerLike(X)
    b = np.linalg.solve(Xs.T @ Xs + a * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))
    np.testing.assert_allclose(m.named_steps["model"].coef_, b, rtol=1e-6, atol=1e-6)


def StandardScalerLike(X):
    """Population-std standardisation, as sklearn's StandardScaler does."""
    Xv = X.values
    return (Xv - Xv.mean(0)) / Xv.std(0)


def test_scaler_fitted_on_training_rows_only():
    m = joblib.load(config.MODEL_DIR / "Ridge.joblib")
    sp_ = split.chronological_split(features.build_features(data.load_clean()))
    X_dev = features.linear_design(sp_.dev, False)
    np.testing.assert_allclose(m.named_steps["scaler"].mean_, X_dev.mean().values, rtol=1e-9)
    X_all = features.linear_design(pd.concat([sp_.dev, sp_.test]), False)
    assert not np.allclose(m.named_steps["scaler"].mean_, X_all.mean().values)


def test_cv_folds_move_forward_in_time(sp):
    for tr, va in TimeSeriesSplit(n_splits=lm.N_SPLITS).split(sp.dev):
        assert tr.max() < va.min()


def test_selected_alpha_is_interior_or_on_a_flat_plateau():
    a = pd.read_csv(config.TABLE_DIR / "phase4_alphas.csv")
    bad = a[a.best_at_low_edge & ~a.curve_flat_there]
    assert bad.empty, f"alpha on grid edge with a still-falling CV curve: {bad.model.tolist()}"
    assert (a.alpha_best < a.grid_max).all()


def test_lasso_one_se_is_sparser_than_best():
    c = pd.read_csv(config.TABLE_DIR / "phase4_coefficients_std.csv", index_col=0)
    for m in ["Lasso", "Lasso + hour×month"]:
        assert (c[f"{m} (1-SE α)"].abs() > 1e-9).sum() < (c[m].abs() > 1e-9).sum()


def test_linear_predictions_cover_same_rows_as_baselines():
    p = ev.load_all_predictions()
    t = p[p.split == "test"]
    ref = set(t[t.model == "Naive-1 (persistence)"].timestamp)
    for m, g in t.groupby("model"):
        assert set(g.timestamp) == ref, m


def test_table_reproducible_from_predictions():
    p = ev.load_all_predictions()
    tab = pd.read_csv(config.TABLE_DIR / "phase4_linear.csv")
    g = p[(p.split == "test") & (p.model == "Lasso + hour×month")]
    row = tab[(tab.split == "test") & (tab.model == "Lasso + hour×month") & (tab.subset == "all")].iloc[0]
    assert evaluate(g.y_true, g.y_pred)["MAE"] == pytest.approx(row.MAE, abs=1e-3)
