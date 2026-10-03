"""Evaluator gate for Phase 2: every EDA figure exists and has a recorded finding + decision."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

EXPECTED = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10", "F11"]


def test_all_figures_exist():
    names = {p.name[:3] for p in config.FIG_DIR.glob("F*.png")}
    assert set(EXPECTED) <= names


def test_every_figure_has_finding_and_decision():
    f = pd.read_csv(config.TABLE_DIR / "phase2_eda_findings.csv")
    assert list(f["figure"]) == EXPECTED
    assert f[["question", "finding", "modelling_decision"]].notna().all().all()
    assert (f["finding"].str.len() > 40).all()
