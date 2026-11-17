"""V10c and V10d of docs/fair_lstm_declared_protocol.md: unit tests of the causal vote, and the causality of the LSTM used by the networks.
Usage: python -W ignore test_fair_vote.py > cache/grid/fair/check_v10cd.log"""
import numpy as np
import torch

from fair_vote import K_GRID, causal_vote, choose_k, segment_starts, window_index

rng = np.random.default_rng(0)
n_ok = 0


def check(name, cond):
    global n_ok
    assert cond, f"FAILED: {name}"
    n_ok += 1
    print(f"ok  {name}")


seg = np.array(list("aaaaaabbbbcc"))
# 1. k = 1 is the identity
p = rng.integers(0, 4, len(seg))
check("k=1 is the identity", (causal_vote(p, seg, 1, 4) == p).all())
# 2. hand-computed: one segment, classes 0/1/2, k = 3, then all
s1 = np.array(["x"] * 8)
p1 = np.array([0, 1, 1, 2, 2, 2, 0, 0])
exp3 = [0, 1, 1, 1, 2, 2, 2, 0]   # t0: {0}->0; t1: {0,1} tie -> most recent 1; t2: {0,1,1}->1; t3: {1,1,2}->1; t4: {1,2,2}->2; t5: {2,2,2}->2; t6: {2,2,0}->2; t7: {2,0,0}->0
check("hand-computed k=3", causal_vote(p1, s1, 3, 3).tolist() == exp3)
# t0 {0}->0; t1 {0,1} tie->1; t2 {0,1,1}->1; t3 {0,1,1,2}->1; t4 {0,1,1,2,2} tie 1/2 -> most recent 2; t5 {..,2,2,2}->2; t6 counts 0:2,1:2,2:3 ->2; t7 counts 0:3,1:2,2:3 tie 0/2 -> most recent is 0
expall = [0, 1, 1, 1, 2, 2, 2, 0]
check("hand-computed all", causal_vote(p1, s1, None, 3).tolist() == expall)
# 3. 'all' equals k = T
for _ in range(20):
    L = int(rng.integers(1, 40))
    q = rng.integers(0, 5, L)
    sg = np.array(["s"] * L)
    check_ok = (causal_vote(q, sg, None, 5) == causal_vote(q, sg, L, 5)).all()
    assert check_ok
check("'all' equals k = segment length (20 random sequences)", True)
# 4. segments do not leak into each other: the result for a segment is the same alone or inside a longer table, and in any segment order
tabs = []
for name in "abcdef":
    L = int(rng.integers(3, 30))
    tabs.append((np.array([name] * L), rng.integers(0, 6, L)))
for k in K_GRID:
    alone = [causal_vote(q, sg, k, 6) for sg, q in tabs]
    sg_all = np.concatenate([sg for sg, _ in tabs])
    q_all = np.concatenate([q for _, q in tabs])
    joint = causal_vote(q_all, sg_all, k, 6)
    order = rng.permutation(len(tabs))
    sg_perm = np.concatenate([tabs[i][0] for i in order])
    q_perm = np.concatenate([tabs[i][1] for i in order])
    joint_perm = causal_vote(q_perm, sg_perm, k, 6)
    assert (joint == np.concatenate(alone)).all()
    assert (joint_perm == np.concatenate([alone[i] for i in order])).all()
check("no leakage between segments, any segment order, every k", True)
# 5. causality: changing later decisions never changes an earlier voted decision
for k in K_GRID:
    for _ in range(30):
        L = int(rng.integers(5, 40))
        q = rng.integers(0, 5, L)
        sg = np.array(["s"] * L)
        t0 = int(rng.integers(0, L - 1))
        q2 = q.copy()
        q2[t0 + 1:] = rng.integers(0, 5, L - t0 - 1)
        assert (causal_vote(q, sg, k, 5)[:t0 + 1] == causal_vote(q2, sg, k, 5)[:t0 + 1]).all()
check("causal: changing later decisions leaves earlier voted decisions unchanged, every k", True)
# 6. a voted output is always one of the classes that occurred in the span
q = rng.integers(0, 6, 60)
sg = np.array(["s"] * 60)
v = causal_vote(q, sg, 5, 6)
check("voted class occurs in the span", all(v[t] in q[max(0, t - 4):t + 1] for t in range(60)))
# 7. helpers
check("segment_starts", segment_starts(seg).tolist() == [0, 6, 10, 12])
check("window_index", window_index(seg).tolist() == [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2])
# 8. choose_k on a sequence where a long vote clearly helps (mostly right, random errors) picks a k > 1, and on a perfect sequence picks 1 (tie -> smallest)
y = np.repeat([0, 1, 2], 30)
sg = np.repeat(["a", "b", "c"], 30)
noisy = y.copy()
flip = rng.random(len(y)) < 0.25
noisy[flip] = rng.integers(0, 3, flip.sum())
k_noisy, accs = choose_k(noisy, y, sg, 3)
check(f"choose_k picks a long span on noisy decisions (chose {k_noisy})", k_noisy != 1)
k_perfect, _ = choose_k(y, y, sg, 3)
check("choose_k on perfect decisions picks k=1 (tie -> smaller k)", k_perfect == 1)

# V10d: the LSTM of run_lstm_training_check (LSTMNet) is causal: logits at window t unchanged when the sequence is cut after t
import run_lstm_training_check as L

torch.manual_seed(0)
net = L.LSTMNet(input_size=17, hidden_size=64, n_classes=12, dropout=0.0).eval()
x = torch.randn(3, 30, 17)
with torch.no_grad():
    full = net(x)
    for t in (0, 4, 11, 29):
        cut = net(x[:, :t + 1])
        assert torch.allclose(full[:, :t + 1], cut, atol=1e-6), t
check("V10d: LSTM output at window t does not depend on later windows (cut after t = 0, 4, 11, 29)", True)
print(f"\nAll {n_ok} checks passed")
