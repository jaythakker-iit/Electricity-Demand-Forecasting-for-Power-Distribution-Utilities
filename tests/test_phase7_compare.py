"""Evaluator gate for Phase 7: the comparison statistics are correct and calibrated."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import compare as cp  # noqa: E402
from src import config, evaluation as ev, features  # noqa: E402


def _ar(rng, n, phi):
    z = rng.normal(size=n)
    x = np.empty(n)
    x[0] = z[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + z[t]
    return x


def test_dm_is_calibrated_under_autocorrelation():
    """Equal-accuracy models with AR(1) errors: rejection rate must be near 5% (naive test gives ~12%)."""
    rng = np.random.default_rng(0)
    rej = 0
    for _ in range(200):
        c = _ar(rng, 3000, 0.5) * 60
        rej += cp.diebold_mariano(c + _ar(rng, 3000, 0.5) * 20, c + _ar(rng, 3000, 0.5) * 20)["p_value"] < 0.05
    assert 0.01 <= rej / 200 <= 0.10


def test_dm_detects_a_real_difference():
    rng = np.random.default_rng(1)
    c = rng.normal(size=4000) * 60
    r = cp.diebold_mariano(c + rng.normal(size=4000) * 20, c * 1.05 + rng.normal(size=4000) * 20)
    assert r["p_value"] < 0.01 and r["mean_diff"] < 0


def test_holm_is_monotone_and_conservative():
    p = pd.Series([0.01, 0.04, 0.03, 0.2])
    adj = cp.holm(p)
    assert (adj >= p).all() and adj.max() <= 1
    assert adj[0] == pytest.approx(0.04)


def test_block_bootstrap_centres_on_mae():
    rng = np.random.default_rng(2)
    ae = pd.DataFrame({"a": np.abs(rng.normal(size=5000)) * 50})
    b = cp.block_bootstrap(ae, block=168, n_boot=500)
    assert abs(b.mean() - ae["a"].mean()) < 1.0


def test_compare_all_is_single_source_of_truth():
    tab = pd.read_csv(config.TABLE_DIR / "compare_all.csv")
    p = ev.load_all_predictions()
    for m in cp.FINAL:
        g = p[(p.split == "test") & (p.model == m)]
        row = tab[(tab.split == "test") & (tab.model == m) & (tab.subset == "all")].iloc[0]
        assert np.mean(np.abs(g.y_true - g.y_pred)) == pytest.approx(row.MAE, abs=1e-3)


def test_ablation_full_set_reproduces_phase5_xgboost():
    """'C_+festival' == the Phase 5 tree feature set, so its test MAE must match the Phase 5 model
    (up to seed averaging: within 1 MW)."""
    assert set(features.ABLATION["C_+festival"]) == set(features.FEATURE_SETS["tree"])
    abl = pd.read_csv(config.TABLE_DIR / "phase7_ablation.csv")
    full = abl[(abl.feature_set == "C_+festival") & (abl.split == "test")].MAE_ensemble.iloc[0]
    xgb = pd.read_csv(config.TABLE_DIR / "compare_all.csv")
    ref = xgb[(xgb.split == "test") & (xgb.model == "XGBoost") & (xgb.subset == "all")].MAE.iloc[0]
    assert abs(full - ref) < 1.0


def test_walkforward_uses_only_past_data():
    wf = pd.read_csv(config.TABLE_DIR / "phase7_walkforward_predictions.csv", parse_dates=["timestamp"])
    for mth, g in wf.groupby("month"):
        assert g.timestamp.min() >= pd.Period(mth).start_time


def test_shap_values_add_up():
    sv = pd.read_csv(config.TABLE_DIR / "phase7_shap_values.csv", index_col=0)
    assert sv.shape == (2000, len(features.FEATURE_SETS["tree"]))
