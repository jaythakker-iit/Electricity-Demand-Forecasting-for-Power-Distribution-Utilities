"""Evaluator gate for Phase 6: the MLP is trained, reproduced and scored correctly."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, evaluation as ev, features, split  # noqa: E402
from src import mlp  # noqa: E402
from src.metrics import evaluate  # noqa: E402

F = features.FEATURE_SETS["mlp"]


@pytest.fixture(scope="module")
def sp():
    return split.chronological_split(features.build_features(data.load_clean()))


def test_architecture_matches_brief():
    m = mlp.EnergyMLP(len(F), [256, 128, 64], 0.3)
    kinds = [type(l).__name__ for l in m.net]
    assert kinds[:4] == ["Linear", "BatchNorm1d", "ReLU", "Dropout"]
    assert kinds[-1] == "Linear" and kinds.count("Linear") == 4
    assert m.net[3].p == 0.3


def test_same_seed_reproducible_different_seed_not(sp):
    cfg = mlp.MLPConfig("t", [32], max_epochs=3, patience=99)
    a = mlp.MLPForecaster(cfg, F, seed=1).fit(sp.train, sp.val).predict(sp.val)
    b = mlp.MLPForecaster(cfg, F, seed=1).fit(sp.train, sp.val).predict(sp.val)
    c = mlp.MLPForecaster(cfg, F, seed=2).fit(sp.train, sp.val).predict(sp.val)
    assert np.array_equal(a, b) and not np.allclose(a, c)


def test_scalers_fitted_on_fitting_rows_only(sp):
    cfg = mlp.MLPConfig("t", [16], max_epochs=1, patience=99)
    m = mlp.MLPForecaster(cfg, F).fit(sp.train, sp.val)
    np.testing.assert_allclose(m.xs_.mu_, sp.train[F].values.astype(np.float32).mean(0), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(m.ys_.mu_, sp.train[config.TARGET].mean(), rtol=1e-5)


def test_early_stopping_restores_best_epoch(sp):
    cfg = mlp.MLPConfig("t", [32], max_epochs=12, patience=3)
    m = mlp.MLPForecaster(cfg, F).fit(sp.train, sp.val)
    h = m.history_
    assert m.best_epoch_ == int(h.loc[h.val_loss.idxmin(), "epoch"])
    assert len(m.lr_schedule_) == m.best_epoch_


def test_delta_target_adds_back_lag1(sp):
    cfg = mlp.MLPConfig("t", [16], target="delta", max_epochs=1, patience=99)
    m = mlp.MLPForecaster(cfg, F).fit(sp.train, sp.val)
    with torch.no_grad():
        m.model_.eval()
        X = torch.tensor(m.xs_.transform(sp.val[F].values.astype(np.float32)))
        raw = m.ys_.inverse(m.model_(X).numpy())
    np.testing.assert_allclose(m.predict(sp.val), raw + sp.val["load_lag_1"].values, rtol=1e-5)


def test_selection_used_val_not_test():
    meta = json.load(open(config.TABLE_DIR / "phase6_meta.json"))
    vals = {k: v["val_MAE_ensemble"] for k, v in meta["configs"].items()}
    assert meta["selected_on_val"] == min(vals, key=vals.get)


def test_same_rows_and_table_reproducible():
    p = ev.load_all_predictions()
    t = p[p.split == "test"]
    ref = set(t[t.model == "Naive-1 (persistence)"].timestamp)
    for m, g in t.groupby("model"):
        assert set(g.timestamp) == ref, m
    meta = json.load(open(config.TABLE_DIR / "phase6_meta.json"))
    sel = meta["selected_on_val"]
    tab = pd.read_csv(config.TABLE_DIR / "phase6_mlp.csv")
    g = t[t.model == sel]
    row = tab[(tab.split == "test") & (tab.model == sel) & (tab.subset == "all")].iloc[0]
    assert evaluate(g.y_true, g.y_pred)["MAE"] == pytest.approx(row.MAE, abs=1e-3)
