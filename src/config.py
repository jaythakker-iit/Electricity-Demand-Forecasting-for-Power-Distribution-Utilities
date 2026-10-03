"""Central configuration: paths, constants and seeds.

Every script imports from here, so changing a setting (e.g. the split ratio)
changes it everywhere at once.
"""
from pathlib import Path
import random

import numpy as np

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "load_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DATA = PROCESSED_DIR / "features.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
MODEL_DIR = RESULTS_DIR / "models"

for _d in (PROCESSED_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data
TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M"   # raw file is day-first, e.g. 13-04-2023 07:00
TARGET = "target_mw"                  # load at target hour h (= t + 1)
MAX_FFILL_HOURS = 3                   # longest gap we are willing to interpolate

# ---------------------------------------------------------------- split
TRAIN_FRAC = 0.70                     # model fitting
VAL_FRAC = 0.10                       # early stopping / model selection
# remaining 0.20 = TEST (scored once, at the end); same 20% as v1 for comparability

# ---------------------------------------------------------------- reproducibility
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
