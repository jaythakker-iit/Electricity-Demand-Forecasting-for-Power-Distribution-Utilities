"""Phase 2 runner: generate all EDA figures + findings table (logic lives in src/eda.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import eda  # noqa: E402

if __name__ == "__main__":
    for _, r in eda.run_all().iterrows():
        print(f"[{r['figure']}] {r['finding']}")
