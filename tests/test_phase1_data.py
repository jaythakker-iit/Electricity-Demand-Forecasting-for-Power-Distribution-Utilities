"""Evaluator gate for Phase 1: integrity + leakage checks.

Run with:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, features, split  # noqa: E402


@pytest.fixture(scope="module")
def hourly():
    return data.load_clean()


@pytest.fixture(scope="module")
def feats(hourly):
    return features.build_features(hourly)


# ------------------------------------------------------------ integrity
def test_hourly_grid_is_regular_and_unique(hourly):
    assert hourly.index.is_unique
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()


def test_gap_policy(hourly):
    gaps = data.gap_report(hourly)
    # with long-gap seasonal fill: every missing hour is filled AND flagged
    assert hourly["load_mw"].isna().sum() == 0
    assert hourly["is_imputed"].sum() == gaps["hours"].sum()
    # strict policy: only short gaps filled
    strict = data.to_hourly(data.load_raw(), fill_long=False)
    long_gap_hours = gaps.loc[gaps.hours > config.MAX_FFILL_HOURS, "hours"].sum()
    assert strict["load_mw"].isna().sum() == long_gap_hours


def test_observed_values_never_modified(hourly):
    raw = data.load_raw()
    pd.testing.assert_series_equal(hourly.loc[raw.index, "load_mw"], raw["load_mw"],
                                   check_names=False, check_freq=False)


def test_no_imputed_targets_are_scored(hourly, feats):
    imputed_times = hourly.index[hourly["is_imputed"]]
    assert feats.index.intersection(imputed_times).empty


def test_feature_sets_exist_and_have_no_nans(feats):
    for name, cols in {**features.FEATURE_SETS, **features.ABLATION}.items():
        missing = set(cols) - set(feats.columns)
        assert not missing, f"{name} missing {missing}"
        assert not feats[cols].isna().any().any()


def test_main_feature_sets_exclude_weather_at_h():
    for name in ("tree", "linear", "mlp"):
        assert not any(c.startswith("fx_") for c in features.FEATURE_SETS[name])


# ------------------------------------------------------------ leakage
def test_lag_definitions_match_target_hour(hourly, feats):
    """load_lag_k at row h must equal the actual load at h-k (fixes v1 off-by-one)."""
    load = hourly["load_mw"]
    sample = feats.sample(500, random_state=0)
    for k in (1, 24, 168):
        expected = load.reindex(sample.index - pd.Timedelta(hours=k)).values
        np.testing.assert_allclose(sample[f"load_lag_{k}"].values, expected)


def test_future_perturbation_does_not_change_past_features(hourly):
    """Change load & weather from time T onward; features for rows <= T must be identical.

    Row T's features use data up to T-1, so perturbing T and later must not
    affect any feature of rows up to and including T (target excluded).
    """
    base = features.build_features(hourly)
    T = hourly.index[len(hourly) // 2]
    pert = hourly.copy()
    after = pert.index >= T
    pert.loc[after, "load_mw"] *= 3.0
    pert.loc[after, "temp"] += 20.0
    pert.loc[after, "humidity"] -= 10.0
    pert.loc[after, "weather"] = "Storm"
    new = features.build_features(pert)

    cols = [c for c in base.columns if c != config.TARGET and not c.startswith("fx_")]
    common = base.index[base.index <= T].intersection(new.index)
    pd.testing.assert_frame_equal(base.loc[common, cols], new.loc[common, cols])


@pytest.mark.parametrize("gap_kind", ["short", "long"])
def test_no_leakage_through_gap_imputation(gap_kind):
    """Perturb the first observed hour AFTER a gap. Features of that hour (which use
    the imputed values inside the gap) must not change -> imputation is causal."""
    raw = data.load_raw()
    hourly = data.to_hourly(raw)
    gaps = data.gap_report(hourly)
    sel = gaps[gaps.hours <= config.MAX_FFILL_HOURS] if gap_kind == "short" else \
        gaps[gaps.hours > config.MAX_FFILL_HOURS]
    T = sel.iloc[len(sel) // 2]
    after_gap = T.start + pd.Timedelta(hours=int(T.hours))   # first observed hour after the gap
    base = features.build_features(hourly)

    raw2 = raw.copy()
    raw2.loc[raw2.index >= after_gap, "load_mw"] *= 3.0
    new = features.build_features(data.to_hourly(raw2))

    cols = [c for c in base.columns if c != config.TARGET and not c.startswith("fx_")]
    common = base.index[base.index <= after_gap].intersection(new.index)
    pd.testing.assert_frame_equal(base.loc[common, cols], new.loc[common, cols])


def test_split_is_chronological_and_disjoint(feats):
    sp = split.chronological_split(feats)
    assert sp.train.index.max() < sp.val.index.min()
    assert sp.val.index.max() < sp.test.index.min()
    assert len(sp.train) + len(sp.val) + len(sp.test) == len(feats)
    assert abs(len(sp.test) / len(feats) - 0.20) < 0.001


def test_linear_feature_set_is_full_rank(feats):
    """No exact linear dependencies in the linear/MLP design matrix (OLS must be identifiable)."""
    X = feats[features.FEATURE_SETS["linear"]].astype(float)
    X = (X - X.mean()) / X.std()
    assert np.linalg.matrix_rank(X.values) == X.shape[1]
