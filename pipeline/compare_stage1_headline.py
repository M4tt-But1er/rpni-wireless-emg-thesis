"""Extra summaries for the Stage 1 comparison (docs/stage1_declared_protocol.md). Additive: reads the same result files as compare_stage1_ops.py, writes nothing but stdout.
Redirect to cache/stage1_headline.txt.

1. 4-patient mean accuracy per operating point x set x model.
2. Headline: the original outline (set A at ref) against the low-rate, low-bit operating point with the wider sets (UNPAIRED; tables differ).
3. low - ref with a rough Welch interval on the 5 fold accuracies of each side (rough guide only, as the protocol says).
4. Cohen's kappa (declared in the protocol, not printed by compare_stage1_ops.py).
5. Best model per patient at each operating point x set, and how often each model is best.
6. Bit rate of each operating point (8 channels, raw samples)."""
import collections
import json

import numpy as np
from scipy.stats import t as student_t

from compare_stage1_ops import MODELS, OPS, PARTS, SETS, add_tuned, cell


def welch(a, b):
    """Difference of means a - b of two samples of fold accuracies, with a Welch 95% interval (points)."""
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else len(a) + len(b) - 2
    h = student_t.ppf(0.975, df) * se
    d = a.mean() - b.mean()
    return 100 * d, 100 * (d - h), 100 * (d + h)


def main():
    add_tuned()
    print("### 1. Mean accuracy over the 4 patients (%), per operating point x set x model")
    print("| Operating point | Model | Set A | Set B | Set C |")
    print("|---|---|---|---|---|")
    mean_acc = {}
    for op, opname in OPS:
        for m in MODELS:
            row = []
            for s in SETS:
                v = [100 * cell(p, op, s, m)["accuracy"] for p, _ in PARTS]
                mean_acc[(op, s, m)] = np.mean(v)
                row.append(f"{np.mean(v):.2f}")
            print(f"| {opname} | {m} | " + " | ".join(row) + " |")

    print("\n### 2. Headline: original outline (set A, ref 1 kHz / 16 bit) against the wider sets at low (500 Hz / 8 bit). UNPAIRED, 4-patient means (points)")
    for m in MODELS:
        base = mean_acc[("ref", "A", m)]
        print(f"  {m}: ref A {base:.2f} | low A {mean_acc[('low', 'A', m)] - base:+.2f} | low B {mean_acc[('low', 'B', m)] - base:+.2f} | low C {mean_acc[('low', 'C', m)] - base:+.2f}"
              f" | ref C {mean_acc[('ref', 'C', m)] - base:+.2f}")
    print("  Per patient, LDA: low C minus ref A:  " + "; ".join(
        f"{s_} {100 * (cell(p, 'low', 'C', 'LDA')['accuracy'] - cell(p, 'ref', 'A', 'LDA')['accuracy']):+.1f}" for p, s_ in PARTS))
    print("  Per patient, SVM: low C minus ref A:  " + "; ".join(
        f"{s_} {100 * (cell(p, 'low', 'C', 'SVM')['accuracy'] - cell(p, 'ref', 'A', 'SVM')['accuracy']):+.1f}" for p, s_ in PARTS))

    print("\n### 3. low minus ref per cell: difference of pooled accuracy, and mean fold accuracy with a rough Welch 95% interval over 5 + 5 folds (* = excludes 0). Rough guide only (tables differ)")
    for s in SETS:
        n_cells = n_res_neg = n_res_pos = 0
        print(f"\nSet {s}")
        for p, s_ in PARTS:
            row = []
            for m in MODELS:
                lo_, hi_ = cell(p, "low", s, m), cell(p, "ref", s, m)
                d, a, b = welch(lo_["fold_accs"], hi_["fold_accs"])
                n_cells += 1
                n_res_neg += b < 0
                n_res_pos += a > 0
                row.append(f"{m} {d:+.1f} [{a:+.1f},{b:+.1f}]{'*' if (a > 0 or b < 0) else ''}")
            print(f"  {s_}: " + "; ".join(row))
        print(f"  {n_cells} cells: interval entirely below 0 in {n_res_neg}, entirely above 0 in {n_res_pos}")

    print("\n### 4. Cohen's kappa (pooled predictions)")
    print("| Operating point | Patient | Model | Set A | Set B | Set C |")
    print("|---|---|---|---|---|---|")
    for op, opname in OPS:
        for p, s_ in PARTS:
            for m in MODELS:
                print(f"| {opname} | {s_} | {m} | " + " | ".join(f"{cell(p, op, s, m)['cohen_kappa']:.3f}" for s in SETS) + " |")

    print("\n### 5. Best model per patient (pooled accuracy; ties within 0.5 point are listed together)")
    wins = collections.Counter()
    for op, opname in OPS:
        for s in SETS:
            row = []
            for p, s_ in PARTS:
                acc = {m: 100 * cell(p, op, s, m)["accuracy"] for m in MODELS}
                top = max(acc.values())
                best = [m for m, v in acc.items() if v >= top - 0.5]
                for m in best:
                    wins[m] += 1
                row.append(f"{s_}: {'/'.join(best)} ({top:.1f})")
            print(f"  {opname}, set {s}: " + "; ".join(row))
    print("  Times within 0.5 point of the best, out of 24 patient x operating point x set cases: " + ", ".join(f"{m} {wins[m]}" for m in MODELS))

    print("\n### 6. Raw bit rate assuming 8 channels")
    for (op, opname), (fs, bits) in zip(OPS, [(1000, 16), (500, 8)]):
        print(f"  {opname}: {fs} Hz x {bits} bit x 8 channels = {fs * bits * 8 / 1000:.0f} kb/s")


if __name__ == "__main__":
    main()
