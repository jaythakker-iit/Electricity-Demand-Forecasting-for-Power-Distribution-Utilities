"""Builds notebooks/05_mlp_model.ipynb (after run_phase6_mlp.py, run_phase6_diagnostics.py, run_phase6_figures.py)."""
from pathlib import Path

import nbformat as nbf

C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# 05 — Neural network: EnergyMLP (PyTorch)
**EM 630 · Electricity Demand Forecasting · Phase 6**

**Question.** Can a multilayer perceptron, trained by gradient descent, match the tree ensembles (**≈ 51.7 MW**)?

**Experiments**: v1–v3 exactly as in the project brief, plus one of ours:

| | Architecture | Training | Purpose |
|---|---|---|---|
| v1 | [64, 32], dropout 0.1 | Adam lr 1e-3, batch 256 | small network |
| v2 | [256, 128, 64], dropout 0.3 | Adam lr 1e-3, batch 256 | more capacity + more regularisation |
| v3 | as v2 | **lr 1e-4, batch 512** | learning-rate / batch tuning |
| v4 (ours) | best of v1–v3 on Val | **predict the hourly change** | carries over the Phase 5 finding |

**Expectations written down before running**
1. v2's extra capacity helps only if dropout keeps it from overfitting.
2. A smaller learning rate converges more slowly but more smoothly.
3. The Δ-target helps the MLP as it helped the trees.
""")
code(r"""
import sys, json, inspect, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display
from src import config, data, features, split, mlp
pd.set_option("display.precision", 3)
def show(name, width=1000):
    display(Image(filename=str(config.FIG_DIR / f"{name}.png"), width=width))
sp = split.chronological_split(features.build_features(data.load_clean()))
F = features.FEATURE_SETS["mlp"]
print(f"{len(F)} input features (same full-rank set as the linear models)")
""")

md(r"""
---
## 1. The network and how it learns
""")
code(r"""
print(inspect.getsource(mlp.EnergyMLP))
print(mlp.EnergyMLP(len(F), [256, 128, 64], 0.3))
""")
md(r"""
| Component | What it does | Course concept |
|---|---|---|
| `Linear` | $h = Wx + b$ | the parameters learned by gradient descent |
| `BatchNorm1d` | re-centres and re-scales each layer's activations per mini-batch | stabilises and speeds up optimisation |
| `ReLU` | $\max(0, z)$ | the non-linearity; without it the network collapses to one linear model |
| `Dropout(p)` | randomly zeroes a fraction $p$ of units during training | regularisation (a noisy ensemble of sub-networks) |
| `weight_decay=1e-4` in Adam | adds $\lambda\|W\|_2^2$ to the loss | **L2 regularisation**, Ridge for networks |
| **Adam** | gradient descent with momentum and per-parameter step sizes | the optimiser |
| **ReduceLROnPlateau** | halves the learning rate when validation loss stalls | step-size schedule |
| **Early stopping** | stop after `patience` epochs without improvement; restore the best weights | the number of epochs acts as a regularisation hyperparameter |
""")
code(r"""
print(inspect.getsource(mlp.MLPForecaster.fit))
""")
md(r"""
**Shuffling and leakage.** Mini-batches are shuffled *within the training rows* (that is what makes it stochastic gradient descent). It doesn't break the chronological split: Train, Val and Test remain separate blocks in time, and each row's features already encode its own past.

**Test protocol, the same as every other model.** After early stopping on Val, the network is **refit from scratch on Train+Val**, replaying exactly the recorded number of epochs and learning-rate schedule:
""")
code(r"""
print(inspect.getsource(mlp.MLPForecaster.refit_replay))
""")

md(r"""
---
## 2. Training curves
""")
code(r"""
show("F25_mlp_training_curves", 1100)
""")
md(r"""
💡 **Reading the curves**
- Each epoch is a full pass of mini-batch gradient descent. The validation curve is what early stopping watches; the dot marks the restored epoch.
- **Learning-rate halvings** (grey lines) cause the small steps down in the curves: smaller steps let the optimiser settle into a narrower valley.
- For the level models, training loss sits **above** validation loss. That isn't a bug: dropout is active while training and is switched off for evaluation. Measured: v1's training loss is 0.0138 with dropout on and 0.0033 with it off, below validation (0.0088).
""")

md(r"""
---
## 3. Results: seeds, ensembles and selection

Each configuration ran with 3 random seeds. A neural network's result depends on its random initialisation and mini-batch order, so one lucky seed proves nothing. The reported model is the **3-seed ensemble** (average prediction). The configuration carried forward is chosen on **Val**, not on test.
""")
code(r"""
seeds = pd.read_csv(config.TABLE_DIR / "phase6_seed_results.csv")
seeds.groupby("model")[["best_epoch", "final_lr", "val_MAE", "test_MAE"]].agg(["mean", "std"]).round(3)
""")
code(r"""
meta = json.load(open(config.TABLE_DIR / "phase6_meta.json"))
print("Selected on Val:", meta["selected_on_val"])
tab = pd.read_csv(config.TABLE_DIR / "phase6_mlp.csv")
cols = ["model", "MAE", "RMSE", "MAPE", "R2", "Bias", "skill_vs_step", "skill_vs_xgb"]
for s in ["val", "test"]:
    print(f"\n{s.upper()}")
    display(tab[(tab.split == s) & (tab.subset == "all")][cols].sort_values("MAE").reset_index(drop=True))
""")
code(r"""
show("F26_mlp_seed_variance")
""")

md(r"""
---
## 4. Two evaluator follow-ups

**(a) Was v3 bad because of lr = 1e-4, or because of the scheduler?** v3's learning rate was halved six times, to about $10^{-6}$, before it converged. With small steps, validation loss improves slowly and noisily, and ReduceLROnPlateau mistakes the noise for a plateau. **v3b** keeps v3's lr and batch size but uses a patient scheduler:
""")
code(r"""
seeds[seeds.model.str.contains("v3")][["model", "seed", "best_epoch", "final_lr", "val_MAE", "test_MAE"]]
""")
md(r"""
✅ Both effects are real. The patient scheduler recovers most of the loss (Val ≈ 97 → 79 MW), so **v3 was mostly starved by the schedule**. Even so, lr 1e-4 still trails v1's lr 1e-3 (Val ≈ 71): with this data and budget, the larger step size wins.

**(b) Does the Train+Val refit help or hurt the MLP?** We compared each seed's Train-only early-stopped model with its refit version, both scored on test:
""")
code(r"""
rv = pd.read_csv(config.TABLE_DIR / "phase6_refit_variance.csv")
display(rv.round(1))
rv.groupby("model")[["test_MAE_train_only", "test_MAE_refit"]].agg(["mean", "std"]).round(1)
""")
md(r"""
✅ For the MLP, the refit **slightly hurts**: it replays the epoch count but cannot "restore the best epoch", so it ends on noisier weights (v1's seed spread doubles). For v4 the cost is ≈ 0.7 MW. **We keep the protocol anyway**, because every model is treated identically and changing the protocol after seeing test scores would be test-set leakage. It goes in the report as a small, known disadvantage for the MLP.
""")

md(r"""
---
## 5. Peak hours
""")
code(r"""
tab[(tab.split == "test") & (tab.subset == "peak")][["model", "n", "MAE", "MAPE", "Bias"]].sort_values("MAE").reset_index(drop=True)
""")
md(r"""
⚠️ The **level-target** MLPs under-forecast peaks badly (v2 bias ≈ +73 MW, v3 ≈ +101 MW): they shrink toward typical load levels. The Δ-target MLP removes this (bias ≈ +5 MW), the same pattern we saw with the trees.
""")

md(r"""
---
## 6. Summary
""")
code(r"""
show("F27_results_after_mlp")
""")
md(r"""
| Expectation | Result |
|---|---|
| 1. v2's capacity helps only if dropout prevents overfitting | ✘ v2 (60.6) did **not** beat v1 (56.0); the extra capacity bought nothing here |
| 2. Smaller learning rate converges more slowly but more smoothly | ◐ slower ✔, but it ended **worse**, partly because of the scheduler (v3b) |
| 3. Δ-target helps the MLP as it helped the trees | ✔ 56.0 → **52.1 MW**, seed spread ±4.1 → **±0.4** |

**Takeaways for the report**
1. The selected MLP (v4, **52.1 MW**) is within **0.8%** of XGBoost (51.7) and Random Forest (51.9). Three very different model families converge on ≈ 52 MW once they predict the hourly change, which suggests we are near what these features can explain.
2. **The target formulation mattered more than the architecture**, for the networks as it did for the trees.
3. Seed variance is real for neural networks; report ensembles or mean ± std, never a single run.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("05_mlp_model.ipynb")
nbf.write(nb, out)
print("wrote", out)
