# Electricity Demand Forecasting — EM 630 (IIT Gandhinagar)

One-hour-ahead forecasting of hourly electricity demand (MW). The project compares naive baselines, OLS / Ridge / Lasso, Decision Tree / Random Forest / XGBoost and a PyTorch MLP on a strict chronological hold-out.

> Work in progress. See [ROADMAP.md](ROADMAP.md) for the plan and [results/evaluator_log.md](results/evaluator_log.md) for the quality gates.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_phase1_data.py   # raw -> clean hourly -> features -> split
python scripts/run_phase2_eda.py    # 11 EDA figures + findings table
jupyter notebook notebooks/01_EDA.ipynb   # EDA walkthrough with explanations
python scripts/run_phase3_baselines.py      # 5 baselines, v1 bug check, F12–F14 (+ notebooks/02_baselines.ipynb)
python scripts/run_phase4_linear.py   # OLS/Ridge/Lasso ± hour×month, time-series CV (~4 min)
python scripts/run_phase4_figures.py  # F15–F19 (+ notebooks/03_ridge_lasso.ipynb)
python scripts/run_phase5_trees.py    # DT / RF / XGBoost, stage-1 randomised search (~25 min)
python scripts/run_phase5_refine.py   # stage-2 search past grid edges (~20 min)
python scripts/run_phase5_figures.py  # F20–F24 (+ notebooks/04_tree_models.ipynb)
python scripts/run_phase6_mlp.py      # EnergyMLP v1–v4 × 3 seeds (~10 min, CPU)
python scripts/run_phase6_diagnostics.py  # v3b scheduler check + refit-variance check (~10 min)
python scripts/run_phase6_figures.py  # F25–F27 (+ notebooks/05_mlp_model.ipynb)
python scripts/compare_all.py         # unified table, bootstrap CIs, Diebold–Mariano tests, breakdowns (seconds)
python scripts/run_phase7_experiments.py  # walk-forward, ablation, SHAP (~6 min)
python scripts/run_phase7_figures.py  # F28–F36 (+ notebooks/06_evaluation.ipynb)
pytest -q                           # integrity + leakage tests
```

## Layout

```
data/raw/            original CSV (never edited)
data/processed/      model-ready features (generated)
src/                 config, data, features, split, metrics, eda, plotstyle (+ models later)
notebooks/           01_EDA, 02_baselines, 03_ridge_lasso, 04_tree_models, 05_mlp_model, 06_evaluation (+ build_*.py)
FEATURES.md          register of used / dropped features and how to re-include them
scripts/             one runner per phase
tests/               evaluator gate tests
results/             tables/, figures/, models/, evaluator_log.md
legacy/              v1 script, kept for traceability
```
