"""Was the LSTM under-trained, over-fitted, or just short of data?

The headline LSTM (1 layer, 64 units, Adam 1e-3, 40 epochs, batch 16, no early stopping) scored
94.6 / 83.7 on P1 / P2 against 97.3 / 93.0 for Random Forest. With 52-77 training sequences per fold and
batch 16 that is only about 160-200 gradient updates, and training loss was never logged. Two parts:

Part 1, diagnostic curve. Same model, same folds, same features (the Stage-2 set), same z-scoring,
trained for up to 400 epochs. Every 10 epochs the training loss/accuracy and the held-out fold's
accuracy are recorded. Epoch 40 must reproduce the headline per-fold numbers exactly. The held-out curve
is for diagnosis only: nothing below picks an epoch from it.

Part 2, honest tuned runs. Per outer fold, for each configuration, train on 80% of the training
sequences (stratified by movement) and watch the loss on the other 20%; the best epoch (lowest validation
loss, up to 400) is then used to retrain on ALL the outer-training sequences and score the untouched
outer fold. The configuration is also chosen on the validation loss. Configurations:
  base   lr 1e-3, no regularisation
  lr3    lr 3e-3
  reg    lr 1e-3, weight decay 1e-3, dropout 0.3 on the LSTM output
"""
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn.utils.rnn import pad_sequence

from config import CACHE_DIR
from lstm_model import build_segment_sequences, _make_batches
from models import compute_cv_folds
from run_models_norest import compute_metrics

RESULTS_PATH = CACHE_DIR / "lstm_training_check.json"
PARTICIPANTS = ["P1_combined_noRest", "P2_combined_noRest"]
SEED = 0
MAX_EPOCHS = 400
CONFIGS = {
    "base": dict(lr=1e-3, weight_decay=0.0, dropout=0.0),
    "lr3": dict(lr=3e-3, weight_decay=0.0, dropout=0.0),
    "reg": dict(lr=1e-3, weight_decay=1e-3, dropout=0.3),
}


class LSTMNet(nn.Module):
    """Same as lstm_model.SequenceLSTM, plus optional dropout on the LSTM output (0 = identical model)."""

    def __init__(self, input_size, hidden_size, n_classes, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out))


def _batch_tensors(batch, class_to_idx):
    xs = [torch.from_numpy(x) for _, x, _ in batch]
    ys = [torch.full((len(x),), class_to_idx[c], dtype=torch.long) for _, x, c in batch]
    return pad_sequence(xs, batch_first=True), pad_sequence(ys, batch_first=True, padding_value=-100)


def evaluate(model, seqs, class_to_idx, n_classes):
    """(window accuracy, unweighted mean cross-entropy, predicted idx, true idx) over a list of sequences."""
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    correct = total = 0
    loss_sum = 0.0
    preds_all, true_all = [], []
    with torch.no_grad():
        for batch in _make_batches(seqs, 32, shuffle=False, rng=None):
            x, y = _batch_tensors(batch, class_to_idx)
            logits = model(x)
            mask = y != -100
            pred = logits.argmax(dim=-1)
            correct += int((pred[mask] == y[mask]).sum())
            total += int(mask.sum())
            loss_sum += float(ce(logits.reshape(-1, n_classes), y.reshape(-1)))
            preds_all.append(pred[mask].numpy())
            true_all.append(y[mask].numpy())
    return correct / total, loss_sum / total, np.concatenate(preds_all), np.concatenate(true_all)


def train(train_seqs, eval_sets, class_to_idx, n_classes, input_size, epochs, lr, weight_decay, dropout,
          hidden=64, batch_size=16, eval_every=10, eval_from=0):
    """Same procedure and RNG use as lstm_model.train_lstm_fold; evaluation passes touch no RNG.
    Returns (model, history) where history rows hold the epoch, mean training loss and, every
    `eval_every` epochs, accuracy and loss on each named set in eval_sets."""
    torch.manual_seed(SEED)
    model = LSTMNet(input_size, hidden, n_classes, dropout)
    labels = np.concatenate([[class_to_idx[c]] * len(x) for _, x, c in train_seqs])
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / np.where(counts > 0, counts, 1)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(SEED)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in _make_batches(train_seqs, batch_size, shuffle=True, rng=rng):
            x, y = _batch_tensors(batch, class_to_idx)
            optimizer.zero_grad()
            loss = criterion(model(x).reshape(-1, n_classes), y.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if epoch % eval_every == 0 and epoch >= eval_from:
            for name, seqs in eval_sets.items():
                acc, ce, _, _ = evaluate(model, seqs, class_to_idx, n_classes)
                row[f"{name}_acc"], row[f"{name}_loss"] = acc, ce
        history.append(row)
    return model, history


def load_participant(name):
    df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
    fs = json.load(open(CACHE_DIR / "feature_selection_results_norest.json"))
    feats = fs[name]["reduced_features"]
    fold_ids = compute_cv_folds(df)
    classes = sorted(df["cue"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    seqs = build_segment_sequences(df, feats)
    by_id = {sid: (sid, x, c) for sid, x, c in seqs}
    seg_fold = df.groupby("segment_id", sort=False).apply(lambda g: fold_ids[g.index[0]], include_groups=False)
    return df, feats, classes, class_to_idx, by_id, seg_fold


def scaled(seqs, mean, std):
    return [(sid, ((x - mean) / std).astype(np.float32), c) for sid, x, c in seqs]


def fold_data(by_id, seg_fold, fold):
    tr = [by_id[s] for s in seg_fold.index if seg_fold[s] != fold]
    te = [by_id[s] for s in seg_fold.index if seg_fold[s] == fold]
    cat = np.concatenate([x for _, x, _ in tr], axis=0)
    mean, std = cat.mean(axis=0), cat.std(axis=0)
    std[std == 0] = 1.0
    return scaled(tr, mean, std), scaled(te, mean, std), mean, std


def part1(name):
    _, feats, classes, c2i, by_id, seg_fold = load_participant(name)
    n_classes = len(classes)
    rows = []
    for fold in sorted(seg_fold.unique()):
        tr, te, _, _ = fold_data(by_id, seg_fold, fold)
        t0 = time.time()
        _, hist = train(tr, {"train": tr, "test": te}, c2i, n_classes, len(feats), MAX_EPOCHS,
                        **CONFIGS["base"])
        rows.append(hist)
        h40 = next(r for r in hist if r["epoch"] == 40)
        print(f"  {name} fold {fold}: {len(tr)} train seqs, {len(te)} test seqs; epoch 40: train loss "
              f"{h40['train_loss']:.3f}, train acc {h40['train_acc']:.3f}, test acc {h40['test_acc']:.4f}; "
              f"epoch {MAX_EPOCHS}: train acc {hist[-1]['train_acc']:.3f}, test acc {hist[-1]['test_acc']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return rows


def prepare(df, feats, fold_ids):
    """Same tuple as load_participant, for a feature table that is already in memory (used by
    run_networks_at_bitrate.py, which builds tables at other bitrates)."""
    classes = sorted(df["cue"].unique())
    c2i = {c: i for i, c in enumerate(classes)}
    by_id = {sid: (sid, x, c) for sid, x, c in build_segment_sequences(df, feats)}
    seg_fold = df.groupby("segment_id", sort=False).apply(lambda g: fold_ids[g.index[0]], include_groups=False)
    return df, feats, classes, c2i, by_id, seg_fold


def part2(name, loaded=None):
    df, feats, classes, c2i, by_id, seg_fold = loaded or load_participant(name)
    n_classes = len(classes)
    class_labels = classes
    y_true_all, y_pred_all, per_fold = [], [], []
    for fold in sorted(seg_fold.unique()):
        tr, te, _, _ = fold_data(by_id, seg_fold, fold)
        cues = np.array([c for _, _, c in tr])
        # 20% of the training sequences, but never fewer than one per class (P2 has only 52, and 20% = 11 < 12
        # classes). For P1 this is the same size as a plain 20% split.
        val_size = max(n_classes, int(np.ceil(0.2 * len(tr))))
        inner_tr_idx, inner_va_idx = next(StratifiedShuffleSplit(1, test_size=val_size, random_state=SEED).split(
            np.zeros(len(tr)), cues))
        itr, iva = [tr[i] for i in inner_tr_idx], [tr[i] for i in inner_va_idx]
        best = None
        cfg_report = {}
        for cname, cfg in CONFIGS.items():
            _, hist = train(itr, {"val": iva}, c2i, n_classes, len(feats), MAX_EPOCHS, eval_every=5, **cfg)
            evald = [r for r in hist if "val_loss" in r]
            r_best = min(evald, key=lambda r: r["val_loss"])
            cfg_report[cname] = dict(best_epoch=r_best["epoch"], val_loss=r_best["val_loss"], val_acc=r_best["val_acc"])
            if best is None or r_best["val_loss"] < best[2]:
                best = (cname, r_best["epoch"], r_best["val_loss"])
        cname, epoch, _ = best
        model, _ = train(tr, {}, c2i, n_classes, len(feats), epoch, **CONFIGS[cname])
        acc, _, pred, true = evaluate(model, te, c2i, n_classes)
        y_pred_all.append(pred)
        y_true_all.append(true)
        per_fold.append(dict(chosen_config=cname, chosen_epochs=int(epoch), outer_test_acc=acc, inner=cfg_report))
        print(f"  {name} outer fold {fold}: chose {cname} for {epoch} epochs (val loss {best[2]:.3f}); "
              f"inner best epochs {({k: v['best_epoch'] for k, v in cfg_report.items()})}; outer test acc {acc:.4f}",
              flush=True)
    idx_to_class = {i: c for c, i in c2i.items()}
    y_true = np.array([idx_to_class[i] for i in np.concatenate(y_true_all)])
    y_pred = np.array([idx_to_class[i] for i in np.concatenate(y_pred_all)])
    m = compute_metrics(y_true, y_pred, class_labels)
    return dict(tuned_accuracy=m["accuracy"], tuned_weighted_f1=m["weighted_f1"], tuned_kappa=m["cohen_kappa"],
                per_fold=per_fold)


def main():
    results = json.load(open(RESULTS_PATH)) if RESULTS_PATH.exists() else {}
    untuned = json.load(open(CACHE_DIR / "model_results_norest.json"))
    for name in PARTICIPANTS:
        entry = results.setdefault(name, {})
        entry["headline_accuracy"] = untuned[name]["LSTM"]["accuracy"]
        entry["headline_fold_accs"] = untuned[name]["LSTM"]["nested_fold_accs"]
        if "curve" not in entry:
            print(f"=== {name}: part 1, training curve ===", flush=True)
            entry["curve"] = part1(name)
            json.dump(results, open(RESULTS_PATH, "w"))
            e40 = [next(r for r in h if r["epoch"] == 40)["test_acc"] for h in entry["curve"]]
            print("  epoch-40 fold accs vs headline:", np.round(e40, 4).tolist(),
                  np.round(entry["headline_fold_accs"], 4).tolist(), flush=True)
        if "tuned" not in entry:
            print(f"=== {name}: part 2, early-stopped nested runs ===", flush=True)
            entry["tuned"] = part2(name)
            json.dump(results, open(RESULTS_PATH, "w"))
            print(f"== {name}: tuned {entry['tuned']['tuned_accuracy']*100:.2f}% vs headline "
                  f"{entry['headline_accuracy']*100:.2f}%", flush=True)
    print("All done ->", RESULTS_PATH)


if __name__ == "__main__":
    main()
