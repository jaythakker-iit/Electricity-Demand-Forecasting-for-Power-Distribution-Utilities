# Electricity Demand Forecasting — EM 630 (IIT Gandhinagar)

One-hour-ahead forecasting of hourly electricity demand (MW). The project compares naive baselines, OLS / Ridge / Lasso, and Decision Tree / Random Forest / XGBoost on a strict chronological hold-out.

> Work in progress. See [ROADMAP.md](ROADMAP.md) for the plan and [results/evaluator_log.md](results/evaluator_log.md) for the quality gates.

## Quick start

```bash
py -m pip install -r requirements.txt
py scripts/run_phase1_data.py   # raw -> clean hourly -> features -> split
py scripts/run_phase2_eda.py    # 11 EDA figures + findings table
py -m jupyter notebook notebooks/01_EDA.ipynb  # EDA walkthrough with explanations
py scripts/run_phase3_baselines.py      # 5 baselines, v1 bug check, F12–F14 (+ notebooks/02_baselines.ipynb)
py scripts/run_phase4_linear.py   # OLS/Ridge/Lasso ± hour×month, time-series CV (~4 min)
py scripts/run_phase4_figures.py  # F15–F19 (+ notebooks/03_ridge_lasso.ipynb)
py scripts/run_phase5_trees.py    # DT / RF / XGBoost, stage-1 randomised search (~25 min)
py scripts/run_phase5_refine.py   # stage-2 search past grid edges (~20 min)
py scripts/run_phase5_figures.py  # F20–F24 (+ notebooks/04_tree_models.ipynb)
py scripts/compare_all.py         # unified table, bootstrap CIs, MAE-gap intervals, breakdowns (seconds)
py scripts/run_phase7_experiments.py  # walk-forward and ablation (~6 min)
py scripts/run_phase7_figures.py  # F28–F34 and F36 (+ notebooks/06_evaluation.ipynb)
py -m pytest -q                   # integrity + leakage tests
```
## Note : Separate Notebook file is also provided for each analysis phase (e.g.,`Electricity_Demand_Forecasting_Complete_Colab.ipynb`), containing the code along with the executed results and outputs.
> [Electricity_Demand_Forecasting_Complete_Colab.ipynb]([Electricity_Demand_Forecasting_Complete_Colab.ipynb] )

## Layout

```
data/raw/            original CSV (never edited)
data/processed/      model-ready features (generated)
src/                 config, data, features, split, metrics, eda, plotstyle (+ models later)
notebooks/           01_EDA, 02_baselines, 03_ridge_lasso, 04_tree_models, 06_evaluation (+ build_*.py)
FEATURES.md          register of used / dropped features and how to re-include them
scripts/             one runner per phase
tests/               evaluator gate tests
results/             tables/, figures/, models/, evaluator_log.md
legacy/              v1 script, kept for traceability
```
