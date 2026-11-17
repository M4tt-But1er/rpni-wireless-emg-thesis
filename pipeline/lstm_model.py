"""LSTM classifier with sequence-level (not per-feature) handling.

Per spec: the LSTM does not get a per-feature-subset wrapper selection like
the other models -- it uses the full stage-2 reduced feature set directly,
one feature vector per window, and consumes the *sequence of windows within
a trimmed segment* as its input sequence (many-to-many: predicts the
(constant) segment cue at every timestep, trained with a per-timestep
cross-entropy loss). This lets it use temporal context across windows,
which the per-window classifiers can't.

Sequences are batched (padded to the batch's max length, loss masked at
pad positions via ignore_index=-100) for reasonable training speed on CPU.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

RANDOM_STATE = 0


class SequenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)


def build_segment_sequences(df, feature_cols):
    """Returns list of (segment_id, X_seq (n_win, n_feat), cue)."""
    sequences = []
    for seg_id, g in df.groupby("segment_id", sort=False):
        g = g.sort_values("window_time_s")
        X_seq = g[feature_cols].to_numpy(dtype=np.float32)
        cue = g["cue"].iloc[0]
        sequences.append((seg_id, X_seq, cue))
    return sequences


def _make_batches(sequences, batch_size, shuffle, rng):
    idx = np.arange(len(sequences))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield [sequences[j] for j in idx[i:i + batch_size]]


def train_lstm_fold(train_seqs, test_seqs, n_classes, class_to_idx, input_size,
                     hidden_size=64, epochs=40, batch_size=16, lr=1e-3, device="cpu"):
    torch.manual_seed(RANDOM_STATE)
    model = SequenceLSTM(input_size, hidden_size, n_classes).to(device)

    # inverse-frequency class weights (rest dominates ~53-55% of windows)
    all_labels = np.concatenate([[class_to_idx[c]] * len(x) for _, x, c in train_seqs])
    counts = np.bincount(all_labels, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / np.where(counts > 0, counts, 1)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(RANDOM_STATE)

    model.train()
    for epoch in range(epochs):
        for batch in _make_batches(train_seqs, batch_size, shuffle=True, rng=rng):
            xs = [torch.from_numpy(x) for _, x, _ in batch]
            ys = [torch.full((len(x),), class_to_idx[c], dtype=torch.long) for _, x, c in batch]
            xpad = pad_sequence(xs, batch_first=True).to(device)
            ypad = pad_sequence(ys, batch_first=True, padding_value=-100).to(device)

            optimizer.zero_grad()
            logits = model(xpad)  # (B, T, C)
            loss = criterion(logits.reshape(-1, n_classes), ypad.reshape(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in _make_batches(test_seqs, batch_size=32, shuffle=False, rng=rng):
            xs = [torch.from_numpy(x) for _, x, _ in batch]
            ys = [torch.full((len(x),), class_to_idx[c], dtype=torch.long) for _, x, c in batch]
            xpad = pad_sequence(xs, batch_first=True).to(device)
            ypad = pad_sequence(ys, batch_first=True, padding_value=-100).to(device)
            logits = model(xpad)
            preds = logits.argmax(dim=-1)
            mask = ypad != -100
            correct += (preds[mask] == ypad[mask]).sum().item()
            total += mask.sum().item()
    return correct / total


def train_lstm_fold_with_predictions(train_seqs, test_seqs, n_classes, class_to_idx, input_size,
                                      hidden_size=64, epochs=40, batch_size=16, lr=1e-3, device="cpu"):
    """Same training procedure as train_lstm_fold, but additionally returns
    every test-window's true and predicted class (mapped back to the
    original cue values, not class indices) -- needed to pool out-of-fold
    predictions across all 5 folds into one confusion matrix, rather than
    only ever reporting a single averaged accuracy number."""
    torch.manual_seed(RANDOM_STATE)
    model = SequenceLSTM(input_size, hidden_size, n_classes).to(device)
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    all_labels = np.concatenate([[class_to_idx[c]] * len(x) for _, x, c in train_seqs])
    counts = np.bincount(all_labels, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / np.where(counts > 0, counts, 1)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(RANDOM_STATE)

    model.train()
    for epoch in range(epochs):
        for batch in _make_batches(train_seqs, batch_size, shuffle=True, rng=rng):
            xs = [torch.from_numpy(x) for _, x, _ in batch]
            ys = [torch.full((len(x),), class_to_idx[c], dtype=torch.long) for _, x, c in batch]
            xpad = pad_sequence(xs, batch_first=True).to(device)
            ypad = pad_sequence(ys, batch_first=True, padding_value=-100).to(device)

            optimizer.zero_grad()
            logits = model(xpad)
            loss = criterion(logits.reshape(-1, n_classes), ypad.reshape(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for batch in _make_batches(test_seqs, batch_size=32, shuffle=False, rng=rng):
            xs = [torch.from_numpy(x) for _, x, _ in batch]
            ys = [torch.full((len(x),), class_to_idx[c], dtype=torch.long) for _, x, c in batch]
            xpad = pad_sequence(xs, batch_first=True).to(device)
            ypad = pad_sequence(ys, batch_first=True, padding_value=-100).to(device)
            logits = model(xpad)
            preds = logits.argmax(dim=-1)
            mask = ypad != -100
            y_true_all.append(ypad[mask].cpu().numpy())
            y_pred_all.append(preds[mask].cpu().numpy())

    y_true_idx = np.concatenate(y_true_all)
    y_pred_idx = np.concatenate(y_pred_all)
    y_true = np.array([idx_to_class[i] for i in y_true_idx])
    y_pred = np.array([idx_to_class[i] for i in y_pred_idx])
    return y_true, y_pred


def evaluate_lstm_with_folds_with_predictions(df, feature_cols, fold_ids, **train_kwargs):
    """Same as evaluate_lstm_with_folds, but pools every fold's out-of-fold
    (true, predicted) window labels together and also returns per-fold
    accuracy, so callers can compute weighted-F1/kappa/confusion-matrix from
    genuine held-out predictions rather than only an averaged accuracy."""
    classes = sorted(df["cue"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    seg_fold = df.groupby("segment_id", sort=False).apply(
        lambda g: fold_ids[g.index[0]], include_groups=False)

    sequences = build_segment_sequences(df, feature_cols)
    seg_id_to_seq = {sid: (sid, x, c) for sid, x, c in sequences}

    accs = []
    y_true_pooled, y_pred_pooled = [], []
    for fold in sorted(np.unique(fold_ids)):
        train_segs = [seg_id_to_seq[sid] for sid in seg_fold.index if seg_fold[sid] != fold]
        test_segs = [seg_id_to_seq[sid] for sid in seg_fold.index if seg_fold[sid] == fold]

        train_x_concat = np.concatenate([x for _, x, _ in train_segs], axis=0)
        mean = train_x_concat.mean(axis=0)
        std = train_x_concat.std(axis=0)
        std[std == 0] = 1.0

        def _scale(segs):
            return [(sid, ((x - mean) / std).astype(np.float32), c) for sid, x, c in segs]

        y_true, y_pred = train_lstm_fold_with_predictions(
            _scale(train_segs), _scale(test_segs), n_classes, class_to_idx,
            input_size=len(feature_cols), **train_kwargs)
        acc = (y_true == y_pred).mean()
        accs.append(acc)
        y_true_pooled.append(y_true)
        y_pred_pooled.append(y_pred)
        print(f"  fold {fold}: acc={acc:.4f}  (train segs={len(train_segs)}, test segs={len(test_segs)})")

    return np.array(accs), np.concatenate(y_true_pooled), np.concatenate(y_pred_pooled)


def train_lstm_final(all_seqs, n_classes, class_to_idx, input_size,
                      hidden_size=64, epochs=40, batch_size=16, lr=1e-3, device="cpu"):
    """Train on ALL available sequences (no held-out fold) -- this is the
    actual deployable model saved to disk, distinct from the 5 per-fold
    models trained only for honest accuracy estimation. Mirrors
    train_lstm_fold's training loop exactly, minus the evaluation half."""
    torch.manual_seed(RANDOM_STATE)
    model = SequenceLSTM(input_size, hidden_size, n_classes).to(device)

    all_labels = np.concatenate([[class_to_idx[c]] * len(x) for _, x, c in all_seqs])
    counts = np.bincount(all_labels, minlength=n_classes).astype(np.float64)
    weights = counts.sum() / np.where(counts > 0, counts, 1)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(RANDOM_STATE)

    model.train()
    for epoch in range(epochs):
        for batch in _make_batches(all_seqs, batch_size, shuffle=True, rng=rng):
            xs = [torch.from_numpy(x) for _, x, _ in batch]
            ys = [torch.full((len(x),), class_to_idx[c], dtype=torch.long) for _, x, c in batch]
            xpad = pad_sequence(xs, batch_first=True).to(device)
            ypad = pad_sequence(ys, batch_first=True, padding_value=-100).to(device)

            optimizer.zero_grad()
            logits = model(xpad)
            loss = criterion(logits.reshape(-1, n_classes), ypad.reshape(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def evaluate_lstm_with_folds(df, feature_cols, fold_ids, **train_kwargs):
    classes = sorted(df["cue"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    seg_fold = df.groupby("segment_id", sort=False).apply(
        lambda g: fold_ids[g.index[0]], include_groups=False)

    sequences = build_segment_sequences(df, feature_cols)
    seg_id_to_seq = {sid: (sid, x, c) for sid, x, c in sequences}

    accs = []
    for fold in sorted(np.unique(fold_ids)):
        train_segs = [seg_id_to_seq[sid] for sid in seg_fold.index if seg_fold[sid] != fold]
        test_segs = [seg_id_to_seq[sid] for sid in seg_fold.index if seg_fold[sid] == fold]

        train_x_concat = np.concatenate([x for _, x, _ in train_segs], axis=0)
        mean = train_x_concat.mean(axis=0)
        std = train_x_concat.std(axis=0)
        std[std == 0] = 1.0

        def _scale(segs):
            return [(sid, ((x - mean) / std).astype(np.float32), c) for sid, x, c in segs]

        acc = train_lstm_fold(_scale(train_segs), _scale(test_segs), n_classes, class_to_idx,
                               input_size=len(feature_cols), **train_kwargs)
        accs.append(acc)
        print(f"  fold {fold}: acc={acc:.4f}  (train segs={len(train_segs)}, test segs={len(test_segs)})")

    return np.array(accs)
