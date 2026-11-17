"""Tunable causal LSTM for the fair-comparison follow-up (docs/fair_followup_declared_protocol.md, Part B).
Net: optional per-step bottleneck Linear(F, b) + ReLU (+ dropout), 1 or 2 unidirectional LSTM layers, dropout on the LSTM output, linear classifier per step.
With the options off (bottleneck 0, 1 layer, dropout 0, noise 0, no clip) it is the same network as run_lstm_training_check.LSTMNet and train_cfg follows run_lstm_training_check.train step by step
(same seed use, same class weights, same batching), so the earlier configurations 'base', 'lr3' and 'reg' are reproduced exactly (check V12a).
"""
import time

import numpy as np
import torch
import torch.nn as nn

import run_lstm_training_check as L
from lstm_model import _make_batches

OLD = {
    "base": dict(hidden=64, layers=1, bottleneck=0, dropout=0.0, wd=0.0, lr=1e-3, noise=0.0, batch=16, clip=None),
    "lr3": dict(hidden=64, layers=1, bottleneck=0, dropout=0.0, wd=0.0, lr=3e-3, noise=0.0, batch=16, clip=None),
    "reg": dict(hidden=64, layers=1, bottleneck=0, dropout=0.3, wd=1e-3, lr=1e-3, noise=0.0, batch=16, clip=None),
}
SPACE = dict(hidden=[32, 64, 128], layers=[1, 2], bottleneck=[0, 32], dropout=[0.0, 0.3, 0.5], wd=[0.0, 1e-3, 1e-2], lr=[1e-3, 3e-3, 1e-2], noise=[0.0, 0.3], batch=[8, 16])
N_RANDOM = 32
SEARCH_SEED = 20261109
MAX_EPOCHS = 400
EVAL_EVERY = 5
PATIENCE = 10            # validations (= 50 epochs) without a better validation loss
N_SEEDS = 5


def cfg_key(cfg):
    return tuple((k, cfg[k]) for k in sorted(SPACE))


def random_configs(n=N_RANDOM, seed=SEARCH_SEED):
    """n distinct random configurations (never equal to an old one), named r01 ... ."""
    rng = np.random.default_rng(seed)
    seen = {cfg_key(c) for c in OLD.values()}
    out = {}
    while len(out) < n:
        c = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        if cfg_key(c) in seen:
            continue
        seen.add(cfg_key(c))
        c["clip"] = 1.0
        out[f"r{len(out) + 1:02d}"] = c
    return out


def all_configs():
    return {**OLD, **random_configs()}


# Round 2 (docs/fair_followup_declared_protocol.md, amendment A2): the axes on which the round-1 top 6 sat at an edge are extended by one step.
SPACE_R2 = dict(hidden=[16, 32, 64, 128], layers=[1, 2], bottleneck=[0, 32, 64], dropout=[0.0, 0.3, 0.5], wd=[0.0, 1e-3, 1e-2], lr=[1e-3, 3e-3, 1e-2, 3e-2], noise=[0.0, 0.3, 0.6], batch=[4, 8, 16])
SEARCH_SEED_R2 = 20261110


def random_configs_r2(n=N_RANDOM, seed=SEARCH_SEED_R2):
    """n distinct random configurations of SPACE_R2, never equal to a round-1 configuration (old or random), named s01 ... ."""
    rng = np.random.default_rng(seed)
    seen = {cfg_key(c) for c in all_configs().values()}
    out = {}
    while len(out) < n:
        c = {k: v[int(rng.integers(len(v)))] for k, v in SPACE_R2.items()}
        if cfg_key(c) in seen:
            continue
        seen.add(cfg_key(c))
        c["clip"] = 1.0
        out[f"s{len(out) + 1:02d}"] = c
    return out


class Net(nn.Module):
    def __init__(self, n_features, cfg, n_classes):
        super().__init__()
        b, p = cfg["bottleneck"], cfg["dropout"]
        self.proj = nn.Linear(n_features, b) if b else None
        self.in_drop = nn.Dropout(p) if b else None
        self.lstm = nn.LSTM(b or n_features, cfg["hidden"], num_layers=cfg["layers"], batch_first=True, dropout=p if cfg["layers"] > 1 else 0.0)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(cfg["hidden"], n_classes)

    def forward(self, x):
        if self.proj is not None:
            x = self.in_drop(torch.relu(self.proj(x)))
        out, _ = self.lstm(x)
        return self.fc(self.drop(out))


def train_cfg(train_seqs, c2i, n_classes, n_features, cfg, epochs=None, val_seqs=None, seed=0):
    """Train one network. epochs=None with val_seqs: early stopping (validate every EVAL_EVERY epochs, stop after PATIENCE validations without a lower validation loss, at most MAX_EPOCHS) and return
    (model, dict(best_epoch, best_val_loss, val_acc, epochs_run, seconds)); epochs=N: train exactly N epochs without validation and return (model, dict(epochs_run, seconds))."""
    t0 = time.time()
    torch.manual_seed(seed)
    model = Net(n_features, cfg, n_classes)
    labels = np.concatenate([[c2i[c]] * len(x) for _, x, c in train_seqs])
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / np.where(counts > 0, counts, 1)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    rng = np.random.default_rng(seed)
    limit = MAX_EPOCHS if epochs is None else epochs
    best = dict(best_epoch=0, best_val_loss=float("inf"), val_acc=float("nan"))
    bad = 0
    ran = 0
    for epoch in range(1, limit + 1):
        model.train()
        for batch in _make_batches(train_seqs, cfg["batch"], shuffle=True, rng=rng):
            x, y = L._batch_tensors(batch, c2i)
            if cfg["noise"] > 0:
                x = x + cfg["noise"] * torch.randn_like(x)
            optimizer.zero_grad()
            loss = criterion(model(x).reshape(-1, n_classes), y.reshape(-1))
            loss.backward()
            if cfg["clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            optimizer.step()
        ran = epoch
        if epochs is None and val_seqs is not None and epoch % EVAL_EVERY == 0:
            acc, ce, _, _ = L.evaluate(model, val_seqs, c2i, n_classes)
            if ce < best["best_val_loss"] - 1e-12:
                best = dict(best_epoch=epoch, best_val_loss=float(ce), val_acc=float(acc))
                bad = 0
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    info = dict(epochs_run=ran, seconds=time.time() - t0)
    if epochs is None:
        info.update(best)
    return model, info


def predict_proba(model, seqs, c2i, n_classes):
    """Softmax outputs (n_windows, n_classes) and true class indices for a list of sequences, in sequence order (same order as run_lstm_training_check.evaluate)."""
    model.eval()
    outs, trues = [], []
    with torch.no_grad():
        for batch in _make_batches(seqs, 32, shuffle=False, rng=None):
            x, y = L._batch_tensors(batch, c2i)
            mask = y != -100
            outs.append(torch.softmax(model(x), dim=-1)[mask].numpy())
            trues.append(y[mask].numpy())
    return np.concatenate(outs), np.concatenate(trues)


def macs_per_step(n_features, cfg, n_classes):
    """Analytic multiply-accumulates per window step (batch of one): bottleneck + LSTM layers (4 gates) + classifier."""
    h, b, nl = cfg["hidden"], cfg["bottleneck"], cfg["layers"]
    in_dim = b or n_features
    m = n_features * b if b else 0
    m += 4 * h * (in_dim + h)
    if nl == 2:
        m += 4 * h * (h + h)
    return m + h * n_classes
