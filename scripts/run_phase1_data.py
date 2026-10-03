"""Phase 1 runner: raw CSV -> clean hourly series -> features -> split summary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, features, split  # noqa: E402


def main():
    raw = data.load_raw()
    hourly = data.to_hourly(raw)
    gaps = data.gap_report(hourly)
    feats = features.build_features(hourly)
    sp = split.chronological_split(feats)

    feats.to_csv(config.PROCESSED_DATA)
    gaps.to_csv(config.TABLE_DIR / "phase1_gap_report.csv", index=False)
    sp.summary().to_csv(config.TABLE_DIR / "phase1_split_summary.csv", index=False)

    audit = {
        "raw_rows": len(raw),
        "hourly_grid_rows": len(hourly),
        "missing_hours_raw": int(hourly["is_missing_raw"].sum()),
        "gap_runs": len(gaps),
        "gap_runs_filled(<=3h)": int((gaps["hours"] <= config.MAX_FFILL_HOURS).sum()),
        "hours_imputed": int(hourly["is_imputed"].sum()),
        "rows_dropped_in_features": feats.attrs["rows_dropped"],
        "model_ready_rows": len(feats),
        "n_features_tree": len(features.FEATURE_SETS["tree"]),
        "n_features_linear": len(features.FEATURE_SETS["linear"]),
    }
    import pandas as pd
    pd.Series(audit).to_csv(config.TABLE_DIR / "phase1_data_audit.csv", header=["value"])
    print(pd.Series(audit).to_string())
    print()
    print(sp.summary().to_string(index=False))


if __name__ == "__main__":
    main()
