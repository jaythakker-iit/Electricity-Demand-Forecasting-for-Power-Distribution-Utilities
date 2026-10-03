"""Evaluator gate for Phase 3: baselines are computed correctly and honestly."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import baselines, config, data, evaluation as ev, features, split  # noqa: E402
from src.metrics import evaluate  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    hourly = data.load_clean()
    feats = features.build_features(hourly)
    return hourly, split.chronological_split(feats)


def test_naive_equals_independently_shifted_series(ctx):
    """Recompute each naive forecast from the raw hourly series (not the feature table)."""
    hourly, sp = ctx
    for name, k in [("Naive-1 (persistence)", 1), ("Seasonal naive-24", 24), ("Seasonal naive-168", 168)]:
        pred = baselines.naive_forecasts(sp.test)[name]
        expected = hourly["load_mw"].shift(k).reindex(sp.test.index)
        np.testing.assert_allclose(pred.values, expected.values)


def test_climatology_uses_only_past(ctx):
    _, sp = ctx
    clim = baselines.CalendarClimatology().fit(sp.dev)
    assert clim.fit_end_ < sp.test.index.min()
    p = clim.predict(sp.test)
    assert p.notna().all()
    # spot check one cell by hand
    row = sp.test.iloc[100]
    cell = sp.dev[(sp.dev.hour == row.hour) & (sp.dev.dow == row.dow) & (sp.dev.month == row.month)]
    assert p.iloc[100] == pytest.approx(cell[config.TARGET].mean())


def test_metrics_known_values():
    y = pd.Series([100.0, 200.0, 300.0])
    m = evaluate(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1
    m = evaluate(y, y + 10)
    assert m["MAE"] == pytest.approx(10) and m["Bias"] == pytest.approx(-10)
    assert m["MAPE"] == pytest.approx(np.mean([10, 5, 10 / 3]))


def test_saved_table_matches_predictions():
    """The CSV that the report will cite must be reproducible from the saved predictions."""
    preds = pd.read_csv(ev.PRED_DIR / "predictions_phase3_baselines.csv", parse_dates=["timestamp"])
    table = pd.read_csv(config.TABLE_DIR / "phase3_baselines.csv")
    g = preds[(preds.split == "test") & (preds.model == "Naive-1 (persistence)")]
    row = table[(table.split == "test") & (table.model == "Naive-1 (persistence)") & (table.subset == "all")].iloc[0]
    assert evaluate(g.y_true, g.y_pred)["MAE"] == pytest.approx(row.MAE, abs=1e-3)


def test_every_split_scored_on_identical_rows():
    preds = pd.read_csv(ev.PRED_DIR / "predictions_phase3_baselines.csv", parse_dates=["timestamp"])
    for _, g in preds.groupby("split"):
        sets = [set(x.timestamp) for _, x in g.groupby("model")]
        assert all(s == sets[0] for s in sets)


def test_step_model_uses_only_past(ctx):
    _, sp = ctx
    m = baselines.PersistencePlusStep().fit(sp.dev)
    assert m.fit_end_ < sp.test.index.min()
    p = m.predict(sp.test)
    assert p.notna().all()
    row = sp.test.iloc[50]
    cell = sp.dev[(sp.dev.hour == row.hour) & (sp.dev.month == row.month)]
    expected = row.load_lag_1 + (cell[config.TARGET] - cell.load_lag_1).mean()
    assert p.iloc[50] == pytest.approx(expected)
