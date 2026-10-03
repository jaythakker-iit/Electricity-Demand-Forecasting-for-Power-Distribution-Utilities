"""Evaluator gate for Phase 5: tree models tuned and scored correctly."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, evaluation as ev, features, split  # noqa: E402
from src import tree_models as tm  # noqa: E402
from src.metrics import evaluate  # noqa: E402

F = features.FEATURE_SETS["tree"]


@pytest.fixture(scope="module")
def sp():
    return split.chronological_split(features.build_features(data.load_clean()))


def test_delta_wrapper_is_exact(sp):
    """'delta' mode must predict  lag_1 + f(X)  where f was trained on y - lag_1."""
    X, y = sp.train[F].iloc[:3000], sp.train[config.TARGET].iloc[:3000]
    m = tm.TargetMode(DecisionTreeRegressor(max_depth=4, random_state=0), "delta").fit(X, y)
    raw = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y - X["load_lag_1"])
    Xt = sp.val[F].iloc[:500]
    np.testing.assert_allclose(m.predict(Xt), raw.predict(Xt) + Xt["load_lag_1"].values)


def test_level_tree_cannot_exceed_training_max(sp):
    """Documents WHY the delta target exists: a level tree is capped at the training range."""
    X, y = sp.train[F], sp.train[config.TARGET]
    m = tm.TargetMode(DecisionTreeRegressor(max_depth=12, random_state=0), "level").fit(X, y)
    X_hot = sp.test[F].copy()
    X_hot[["load_lag_1", "load_lag_2", "load_lag_3", "load_roll_mean_24"]] *= 1.5   # an unseen heatwave
    assert m.predict(X_hot).max() <= y.max() + 1e-6


def test_saved_models_refit_on_dev_only(sp):
    for name in ["Decision Tree", "Random Forest", "XGBoost"]:
        path = config.MODEL_DIR / f"{name.replace(' ', '_')}.joblib"
        if not path.exists():          # Random Forest (212 MB) is not shipped; regenerate with run_phase5_trees.py
            continue
        m = joblib.load(path)
        assert list(m.feature_names_in_) == F
        # a model refit on dev reproduces the saved test predictions exactly
        p = ev.load_all_predictions()
        g = p[(p.split == "test") & (p.model == name)].set_index("timestamp")
        np.testing.assert_allclose(m.predict(sp.test[F]), g.loc[sp.test.index, "y_pred"].values, rtol=1e-5)


def test_search_used_forward_cv_and_mae():
    for name in ["Decision_Tree", "Random_Forest", "XGBoost"]:
        r = pd.read_csv(config.TABLE_DIR / f"phase5_cv_{name}.csv")
        assert {"cv_mae", "train_mae"} <= set(r.columns)
        assert (r.cv_mae > 0).all()


def test_same_rows_as_every_other_model():
    p = ev.load_all_predictions()
    t = p[p.split == "test"]
    ref = set(t[t.model == "Naive-1 (persistence)"].timestamp)
    for m, g in t.groupby("model"):
        assert set(g.timestamp) == ref, m


def test_table_reproducible():
    p = ev.load_all_predictions()
    tab = pd.read_csv(config.TABLE_DIR / "phase5_trees.csv")
    g = p[(p.split == "test") & (p.model == "XGBoost")]
    row = tab[(tab.split == "test") & (tab.model == "XGBoost") & (tab.subset == "all")].iloc[0]
    assert evaluate(g.y_true, g.y_pred)["MAE"] == pytest.approx(row.MAE, abs=1e-3)
