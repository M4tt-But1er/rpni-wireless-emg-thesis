"""V11b (candidates) and V11c (stacked features): unit tests of fair_history.py. Run: python -W ignore test_fair_history.py"""
import numpy as np

import fair_history as H
from fair_vote import causal_vote

rng = np.random.default_rng(0)
C = 4


def make(lens):
    seg = np.concatenate([[s] * m for s, m in enumerate(lens)])
    P = rng.dirichlet(np.ones(C), size=len(seg))
    return seg, P


def brute(P, seg, family, param):
    out = []
    for s in np.unique(seg):
        p = P[seg == s]
        for t in range(len(p)):
            lo = 0 if param is None else max(0, t - param + 1)
            z = np.log(np.maximum(p[lo:t + 1], H.LOG_CLIP)) if family == "log" else p[lo:t + 1]
            out.append(int(np.argmax(z.sum(axis=0))))
    return np.array(out)


def test_k1_is_argmax():
    seg, P = make([7, 9, 5])
    for fam in ("mean", "log"):
        assert (H.soft_vote(P, seg, fam, 1) == (P.argmax(axis=1) if fam == "mean" else np.log(np.maximum(P, H.LOG_CLIP)).argmax(axis=1))).all()
    assert (H.soft_vote(P, seg, "ema", 0.0) == P.argmax(axis=1)).all()


def test_against_brute_force():
    seg, P = make([12, 1, 20, 3, 15])
    for fam in ("mean", "log"):
        for k in (1, 2, 3, 5, 9, 15, None):
            assert (H.soft_vote(P, seg, fam, k) == brute(P, seg, fam, k)).all(), (fam, k)


def test_ema_manual():
    seg, P = make([10, 6])
    a = 0.8
    ref = []
    for s in np.unique(seg):
        p = P[seg == s]
        st = p[0].copy()
        ref.append(int(np.argmax(st)))
        for t in range(1, len(p)):
            st = a * st + (1 - a) * p[t]
            ref.append(int(np.argmax(st)))
    assert (H.soft_vote(P, seg, "ema", a) == np.array(ref)).all()


def test_tie_rule():
    seg = np.array([0, 0, 1, 1])
    P = np.array([[0.6, 0.4, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    r = H.soft_vote(P, seg, "mean", None)
    assert r[1] == 1, "tie must go to the class with the larger probability in the current window"
    assert r[3] == 0, "full tie must go to the lower class index"


def test_all_equals_span_of_segment_length():
    seg, P = make([8, 11])
    for fam in ("mean", "log"):
        assert (H.soft_vote(P, seg, fam, None) == H.soft_vote(P, seg, fam, 11)).all()


def test_no_leakage_between_segments_and_causal():
    seg, P = make([9, 13, 6])
    full = {(f, k): H.soft_vote(P, seg, f, k) for f in ("mean", "log") for k in (3, None)}
    full["ema"] = H.soft_vote(P, seg, "ema", 0.9)
    for s, (i, j) in enumerate([(0, 9), (9, 22), (22, 28)]):
        for key, v in full.items():
            f, k = key if isinstance(key, tuple) else ("ema", 0.9)
            alone = H.soft_vote(P[i:j], seg[i:j], f, k)
            assert (alone == v[i:j]).all(), (key, s)
            m = (j - i) // 2 + 1                      # truncating the segment must not change earlier decisions
            trunc = H.soft_vote(P[i:i + m], seg[i:i + m], f, k)
            assert (trunc == v[i:i + m]).all(), (key, s, "causality")


def test_apply_candidate_hard():
    seg, P = make([9, 13])
    h = P.argmax(axis=1)
    assert (H.apply_candidate(("hard", 1), h, P, seg, C) == h).all()
    assert (H.apply_candidate(("hard", None), h, P, seg, C) == causal_vote(h, seg, None, C)).all()


def test_candidate_list():
    assert len(H.CANDIDATES) == 22 and len(set(H.CAND_NAMES)) == 22
    assert H.CAND_NAMES[:6] == ["hard_k1", "hard_k3", "hard_k5", "hard_k9", "hard_k15", "hard_all"]
    assert len(H.SOFT_IDX) == 16 and len(H.HARD_IDX) == 6


def test_stack_features():
    seg = np.concatenate([[0] * 20, [1] * 3, [2] * 10])
    X = rng.normal(size=(len(seg), 5))
    for var in ("ST3", "ST5"):
        S = H.stack_features(X, seg, var)
        lags = H.LAGS[var]
        assert S.shape == (len(seg), 5 * len(lags))
        b = H.segment_starts(seg)
        for i, j in zip(b[:-1], b[1:]):
            for t in range(j - i):
                for q, l in enumerate(lags):
                    assert (S[i + t, 5 * q:5 * (q + 1)] == X[i + max(t - l, 0)]).all(), (var, i, t, l)
    # causal: changing windows after t does not change row t
    S0 = H.stack_features(X, seg, "ST5")
    X2 = X.copy()
    X2[12:20] += 100.0
    S1 = H.stack_features(X2, seg, "ST5")
    assert (S0[:12] == S1[:12]).all() and (S0[20:] == S1[20:]).all(), "stacked features changed before the edit or across segments"
    # CUM: running mean since the onset
    Cm = H.stack_features(X, seg, "CUM")
    assert Cm.shape == (len(seg), 10) and (Cm[:, :5] == X).all()
    for i, j in [(0, 20), (20, 23), (23, 33)]:
        for t in range(j - i):
            assert np.allclose(Cm[i + t, 5:], X[i:i + t + 1].mean(axis=0))
    assert H.stack_features(X, seg, "none") is X


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} checks passed")
