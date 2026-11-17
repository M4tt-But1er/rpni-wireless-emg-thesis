"""Causal majority vote over a model's per-window decisions (docs/fair_lstm_declared_protocol.md).

Rows must be in table order: the windows of one segment are contiguous and in time order. The vote restarts at every segment (the onset), uses only the current and earlier windows of the
segment, and ties between classes go to the class of the most recent decision in the voting span. k = None means 'all decisions since the onset'.
"""
import numpy as np

K_GRID = [1, 3, 5, 9, 15, None]          # candidate spans of the nested choice; None = all since onset (the largest)
K_NAMES = {1: "k1", 3: "k3", 5: "k5", 9: "k9", 15: "k15", None: "all"}


def segment_starts(seg_ids):
    """Row indices where a new segment starts, plus the end (len)."""
    seg = np.asarray(seg_ids)
    change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    return np.concatenate([[0], change, [len(seg)]])


def window_index(seg_ids):
    """1-based index of every window within its segment (1 = first window after the onset)."""
    b = segment_starts(seg_ids)
    out = np.empty(b[-1], dtype=int)
    for i, j in zip(b[:-1], b[1:]):
        out[i:j] = np.arange(1, j - i + 1)
    return out


def causal_vote(pred_idx, seg_ids, k, n_classes):
    """pred_idx: integer class indices (n,), seg_ids: (n,), k: int >= 1 or None. Returns the voted class indices (n,)."""
    p = np.asarray(pred_idx).tolist()
    b = segment_starts(seg_ids)
    res = [0] * len(p)
    for i, j in zip(b[:-1], b[1:]):
        counts = [0] * n_classes
        for t in range(i, j):
            counts[p[t]] += 1
            if k is not None and t - i >= k:
                counts[p[t - k]] -= 1
            lo = i if k is None else max(i, t - k + 1)
            top = max(counts)
            winner = p[t]
            if counts[winner] != top:                      # the newest decision is not a top class: take the most recent one that is
                for s in range(t, lo - 1, -1):
                    if counts[p[s]] == top:
                        winner = p[s]
                        break
            res[t] = winner
    return np.asarray(res, dtype=int)


def choose_k(inner_pred_idx, y_idx, seg_ids, n_classes):
    """The k of K_GRID with the best accuracy of the voted predictions; ties go to the earlier (smaller) k. Returns (k, [accuracy per candidate])."""
    accs = [float((causal_vote(inner_pred_idx, seg_ids, k, n_classes) == y_idx).mean()) for k in K_GRID]
    best = int(np.argmax(accs))                            # argmax returns the first maximum = smallest k
    return K_GRID[best], accs
