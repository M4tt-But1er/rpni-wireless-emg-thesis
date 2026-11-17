"""History for per-window classifiers (docs/fair_followup_declared_protocol.md, Part A): stacked-window features and the 22 vote candidates.

Rows must be in table order (the windows of one segment contiguous and in time order, V10e). Everything is causal and restarts at the segment onset.
  stack_features(X, seg_ids, variant)        variant in STACK_VARIANTS: none, ST3, ST5 (lagged windows, 4 windows = 200 ms apart), CUM (current window + running mean since the onset)
  soft_vote(P, seg_ids, family, param)       family mean / log / ema over class probabilities P (n, C)
  apply_candidate(cand, hard_idx, P, ...)    one of CANDIDATES (hard / mean / log / ema)
  candidate_accuracies(...)                  accuracy of every candidate (used for the nested choice)
"""
import numpy as np

from fair_vote import K_GRID, causal_vote, segment_starts

STACK_VARIANTS = ["none", "ST3", "ST5", "CUM"]
LAGS = {"none": (0,), "ST3": (0, 4, 8), "ST5": (0, 4, 8, 12, 16)}      # in windows (hop 50 ms)
EMA_ALPHAS = [0.5, 0.8, 0.9, 0.95]
LOG_CLIP = 1e-3
TIE = 1e-9                                                              # scores closer than this are a tie

CANDIDATES = ([("hard", k) for k in K_GRID] + [("mean", k) for k in K_GRID] + [("log", k) for k in K_GRID] + [("ema", a) for a in EMA_ALPHAS])


def cand_name(cand):
    fam, p = cand
    if fam == "ema":
        return f"ema_{p}"
    return f"{fam}_{'all' if p is None else 'k%d' % p}"


CAND_NAMES = [cand_name(c) for c in CANDIDATES]
SOFT_IDX = [i for i, c in enumerate(CANDIDATES) if c[0] != "hard"]      # the 16 soft candidates
HARD_IDX = [i for i, c in enumerate(CANDIDATES) if c[0] == "hard"]      # the 6 hard candidates


def stack_features(X, seg_ids, variant):
    """Causal history features of one table. A lag that reaches before the first window of the segment is clamped to the first window."""
    if variant == "none":
        return X
    b = segment_starts(seg_ids)
    n, F = X.shape
    if variant in LAGS:
        lags = LAGS[variant]
        out = np.empty((n, F * len(lags)), dtype=X.dtype)
        for i, j in zip(b[:-1], b[1:]):
            idx = np.arange(j - i)
            out[i:j] = np.concatenate([X[i + np.maximum(idx - l, 0)] for l in lags], axis=1)
        return out
    if variant == "CUM":
        out = np.empty((n, 2 * F), dtype=np.float64)
        for i, j in zip(b[:-1], b[1:]):
            seg = X[i:j].astype(np.float64)
            out[i:j, :F] = seg
            out[i:j, F:] = np.cumsum(seg, axis=0) / np.arange(1, j - i + 1)[:, None]
        return out
    raise ValueError(variant)


def _pick(score, p_now):
    """argmax per row; scores within TIE of the top are tied and go to the class with the larger current-window probability, then to the lower class index."""
    top = score.max(axis=1, keepdims=True)
    tied = score >= top - TIE
    return np.where(tied, p_now, -1.0).argmax(axis=1)


def soft_vote(P, seg_ids, family, param):
    """family 'mean' (summed probabilities), 'log' (summed log probabilities clipped at LOG_CLIP) over the last `param` windows (None = all since the onset),
    or 'ema' (exponential average with factor `param`, s(1) = p(1)). Returns class indices (n,)."""
    P = np.asarray(P, dtype=np.float64)
    Z = np.log(np.maximum(P, LOG_CLIP)) if family == "log" else P
    b = segment_starts(seg_ids)
    out = np.empty(len(P), dtype=int)
    for i, j in zip(b[:-1], b[1:]):
        z, p = Z[i:j], P[i:j]
        if family == "ema":
            score = np.empty_like(z)
            score[0] = z[0]
            for t in range(1, len(z)):
                score[t] = param * score[t - 1] + (1.0 - param) * z[t]
        elif family in ("mean", "log"):
            S = np.cumsum(z, axis=0)
            score = S.copy()
            if param is not None and len(z) > param:
                score[param:] = S[param:] - S[:-param]
        else:
            raise ValueError(family)
        out[i:j] = _pick(score, p)
    return out


def apply_candidate(cand, hard_idx, P, seg_ids, n_classes):
    fam, p = cand
    if fam == "hard":
        return causal_vote(hard_idx, seg_ids, p, n_classes)
    return soft_vote(P, seg_ids, fam, p)


def candidate_accuracies(hard_idx, P, y_idx, seg_ids, n_classes):
    """Accuracy of every candidate in CANDIDATES order."""
    return [float((apply_candidate(c, hard_idx, P, seg_ids, n_classes) == y_idx).mean()) for c in CANDIDATES]
