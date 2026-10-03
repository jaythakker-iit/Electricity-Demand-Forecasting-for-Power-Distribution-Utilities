"""Phase 6: EnergyMLP (PyTorch).

Course concepts
---------------
Forward pass   : each block = Linear -> BatchNorm -> ReLU -> Dropout; output = Linear(·, 1).
Loss           : mean squared error on the STANDARDISED target.
Optimiser      : Adam = gradient descent with per-parameter adaptive step sizes (momentum + RMS
                 scaling). weight_decay adds an L2 penalty on the weights (Ridge, for networks).
Mini-batches   : stochastic gradient estimates; batch size trades gradient noise against speed.
LR schedule    : ReduceLROnPlateau halves the step size when validation loss stops improving.
Early stopping : stop when validation loss hasn't improved for `patience` epochs and restore the
                 best weights; the number of epochs acts as a regularisation hyperparameter.

Scaling is fitted on the fitting rows only. Target standardisation keeps the MSE gradients on a
sensible scale (raw MW^2 would make Adam's first steps meaningless).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config

LAG1 = "load_lag_1"


@dataclass
class MLPConfig:
    name: str
    hidden: list[int]
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 1e-4
    target: str = "level"            # 'level' or 'delta' (y_h - y_{h-1})
    max_epochs: int = 300
    patience: int = 20               # early stopping
    lr_patience: int = 6             # ReduceLROnPlateau
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    notes: str = ""
    extra: dict = field(default_factory=dict)


class EnergyMLP(nn.Module):
    def __init__(self, n_in: int, hidden: list[int], dropout: float):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Scaler:
    """Standardisation fitted on the fitting rows only (no leakage)."""

    def fit(self, a: np.ndarray):
        self.mu_, self.sd_ = a.mean(0), a.std(0)
        self.sd_ = np.where(self.sd_ < 1e-12, 1.0, self.sd_)
        return self

    def transform(self, a):
        return (a - self.mu_) / self.sd_

    def inverse(self, a):
        return a * self.sd_ + self.mu_


def _target(df: pd.DataFrame, cfg: MLPConfig) -> np.ndarray:
    y = df[config.TARGET].values
    return y - df[LAG1].values if cfg.target == "delta" else y


def _set_seed(seed: int):
    """Seed everything. torch.use_deterministic_algorithms is switched OFF here: on CPU it costs
    ~2.5x speed and the evaluator verified (tests/test_phase6_mlp.py) that same seed -> identical
    predictions without it."""
    config.set_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(2)


class MLPForecaster:
    """Fit/predict wrapper. `fit` with a validation set = early stopping run (records the schedule).
    `refit_replay` = retrain on more data for exactly the recorded epochs and learning rates."""

    def __init__(self, cfg: MLPConfig, features: list[str], seed: int = config.SEED):
        self.cfg, self.features, self.seed = cfg, features, seed

    # ------------------------------------------------------------ internals
    def _prep(self, fit_df):
        self.xs_ = Scaler().fit(fit_df[self.features].values.astype(np.float32))
        self.ys_ = Scaler().fit(_target(fit_df, self.cfg).astype(np.float32))

    def _tensors(self, df):
        X = torch.tensor(self.xs_.transform(df[self.features].values.astype(np.float32)), dtype=torch.float32)
        y = torch.tensor(self.ys_.transform(_target(df, self.cfg).astype(np.float32)), dtype=torch.float32)
        return X, y

    def _new_model(self):
        _set_seed(self.seed)
        m = EnergyMLP(len(self.features), self.cfg.hidden, self.cfg.dropout)
        opt = torch.optim.Adam(m.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return m, opt

    def _epoch(self, m, opt, X, y, gen):
        m.train()
        idx = torch.randperm(len(X), generator=gen)
        tot = 0.0
        loss_fn = nn.MSELoss()
        for i in range(0, len(X), self.cfg.batch_size):
            b = idx[i:i + self.cfg.batch_size]
            if len(b) < 2:            # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = loss_fn(m(X[b]), y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        return tot / len(X)

    @staticmethod
    @torch.no_grad()
    def _eval_loss(m, X, y):
        m.eval()
        return nn.functional.mse_loss(m(X), y).item()

    # ------------------------------------------------------------ public
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train with early stopping on val_df. Shuffling happens WITHIN the training rows only
        (mini-batch SGD); the chronological split between train and val is untouched."""
        t0 = time.time()
        self._prep(train_df)
        Xtr, ytr = self._tensors(train_df)
        Xva, yva = self._tensors(val_df)
        m, opt = self._new_model()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=self.cfg.lr_factor,
                                                           patience=self.cfg.lr_patience, min_lr=self.cfg.min_lr)
        gen = torch.Generator().manual_seed(self.seed)
        best, best_state, best_ep, wait = np.inf, None, 0, 0
        hist = []
        for ep in range(1, self.cfg.max_epochs + 1):
            lr_used = opt.param_groups[0]["lr"]
            tr = self._epoch(m, opt, Xtr, ytr, gen)
            va = self._eval_loss(m, Xva, yva)
            sched.step(va)
            hist.append({"epoch": ep, "train_loss": tr, "val_loss": va, "lr": lr_used})
            if va < best - 1e-6:
                best, best_state, best_ep, wait = va, copy.deepcopy(m.state_dict()), ep, 0
            else:
                wait += 1
                if wait >= self.cfg.patience:
                    break
        m.load_state_dict(best_state)
        self.model_ = m
        self.history_ = pd.DataFrame(hist)
        self.best_epoch_ = best_ep
        self.lr_schedule_ = self.history_["lr"].iloc[:best_ep].tolist()
        self.fit_seconds_ = time.time() - t0
        return self

    def refit_replay(self, dev_df: pd.DataFrame, schedule: list[float]):
        """Retrain from scratch on dev_df for len(schedule) epochs, using exactly those learning
        rates. Nothing from the test period is used, and no early stopping is needed."""
        self._prep(dev_df)
        X, y = self._tensors(dev_df)
        m, opt = self._new_model()
        gen = torch.Generator().manual_seed(self.seed)
        for lr in schedule:
            for g in opt.param_groups:
                g["lr"] = lr
            self._epoch(m, opt, X, y, gen)
        self.model_ = m
        self.best_epoch_ = len(schedule)
        return self

    @torch.no_grad()
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.model_.eval()
        X = torch.tensor(self.xs_.transform(df[self.features].values.astype(np.float32)), dtype=torch.float32)
        p = self.ys_.inverse(self.model_(X).numpy())
        return p + df[LAG1].values if self.cfg.target == "delta" else p

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model_.parameters())
