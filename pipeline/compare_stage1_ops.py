"""Summarise run_stage1_ops.py exactly as declared in docs/stage1_declared_protocol.md. Redirect to cache/stage1_ops_summary.txt.
Within an operating point, sets share rows and folds, so differences are paired by fold (95% t-interval, 4 df; * = excludes 0).
Between operating points the tables differ, so differences are unpaired (rough guide only)."""
import collections
import sys

import json

import numpy as np
from scipy.stats import t as student_t

from build_wide_tables import TAG
from config import CACHE_DIR

R = json.load(open(CACHE_DIR / f"stage1_ops_results{'_' + TAG if TAG else ''}.json"))
PARTS = [("P1_combined", "P1"), ("P2_combined", "P2"), ("Wallace_S1", "S1"), ("Wallace_S2", "S2")]
OPS = [("ref", "ref 1 kHz / 16 bit"), ("low", "low 500 Hz / 8 bit")]
NO_TUNED = "--no-tuned" in sys.argv  # the four models declared before Amendment 2 (16 cells per comparison instead of 20)
MODELS = ["LDA", "SVM", "RandomForest", "RBFNN"] + ([] if NO_TUNED else ["RBFNN_tuned"])
SETS = ["A", "B", "C"]
TCRIT = student_t.ppf(0.975, 4)


def add_tuned():
    """Pool the per-fold tuned-RBFNN tasks (run_stage1_rbfnn_tuned.py) into cells, for every (patient, op, set) whose 5 folds are all present.
    Returns the list of hyper-parameter records for the summary."""
    import pandas as pd
    from build_wide_tables import table_path
    from run_models_norest import compute_metrics
    path = CACHE_DIR / f"stage1_rbfnn_tuned_results{'_' + TAG if TAG else ''}.json"
    if not path.exists():
        return []
    T = json.load(open(path))
    hp = []
    for p, _ in PARTS:
        for op, _ in OPS:
            t = None
            for s in SETS:
                keys = [f"{p}|{op}|{s}|{f}" for f in range(5)]
                if not all(k in T for k in keys):
                    continue
                if t is None:
                    t = pd.read_parquet(table_path(p, op), columns=["cue"])
                y = t["cue"].to_numpy()
                pooled = np.empty_like(y)
                for k in keys:
                    pooled[T[k]["test_index"]] = np.array(T[k]["pred"], dtype=y.dtype)
                cell_ = {"fold_accs": [T[k]["acc"] for k in keys], "n_features": T[keys[0]]["n_features"], "seconds": sum(T[k]["seconds"] for k in keys)}
                cell_.update(compute_metrics(y, pooled, sorted(t["cue"].unique().tolist())))
                R.setdefault(f"{p}|{op}", {})[f"{s}|RBFNN_tuned"] = cell_
                hp += [(op, s, T[k]["n_centers"], T[k]["sigma_scale"], T[k]["C"]) for k in keys]
    return hp


def cell(p, op, s, m):
    return R.get(f"{p}|{op}", {}).get(f"{s}|{m}")


def paired(a, b):
    d = np.array(a["fold_accs"]) - np.array(b["fold_accs"])
    h = TCRIT * d.std(ddof=1) / np.sqrt(len(d))
    return 100 * d.mean(), 100 * (d.mean() - h), 100 * (d.mean() + h)


def main():
    hp = [] if NO_TUNED else add_tuned()
    print("Accuracy % (weighted F1 % in brackets); columns = number of features")
    for op, opname in OPS:
        print(f"\n### {opname}")
        print("| Patient | Model | " + " | ".join(f"Set {s}" for s in SETS) + " |")
        print("|---|---|---|---|---|")
        for p, s_ in PARTS:
            for m in MODELS:
                row = []
                for s in SETS:
                    c = cell(p, op, s, m)
                    row.append("n/a" if c is None else f"{100 * c['accuracy']:.1f} ({100 * c['weighted_f1']:.1f}) F={c['n_features']}")
                print(f"| {s_} | {m} | " + " | ".join(row) + " |")

    for op, opname in OPS:
        print(f"\n### Paired set differences at {opname} (points; 95% interval over 5 folds; * = excludes 0)")
        for a, b in [("B", "A"), ("C", "B"), ("C", "A")]:
            print(f"\n{a} minus {b}")
            allc, per_model, rp, rn = [], collections.defaultdict(list), 0, 0
            for p, s_ in PARTS:
                row = []
                for m in MODELS:
                    ca, cb = cell(p, op, a, m), cell(p, op, b, m)
                    if ca is None or cb is None:
                        row.append(f"{m} n/a")
                        continue
                    mean, lo, hi = paired(ca, cb)
                    allc.append(mean); per_model[m].append(mean); rp += lo > 0; rn += hi < 0
                    row.append(f"{m} {mean:+.1f} [{lo:+.1f},{hi:+.1f}]{'*' if (lo > 0 or hi < 0) else ''}")
                print(f"  {s_}: " + "; ".join(row))
            if allc:
                print(f"  mean {np.mean(allc):+.2f} over {len(allc)} cells; better in {sum(x > 0 for x in allc)}/{len(allc)}; resolved better {rp}, worse {rn}; "
                      + ", ".join(f"{m} {np.mean(v):+.1f}" for m, v in per_model.items()))

    print("\n### low (500 Hz, 8 bit) minus ref (1 kHz, 16 bit), pooled accuracy, UNPAIRED (points)")
    for s in SETS:
        print(f"\nSet {s}")
        allc, per_model = [], collections.defaultdict(list)
        for p, s_ in PARTS:
            row = []
            for m in MODELS:
                lo_, hi_ = cell(p, "low", s, m), cell(p, "ref", s, m)
                if lo_ is None or hi_ is None:
                    row.append(f"{m} n/a")
                    continue
                d = 100 * (lo_["accuracy"] - hi_["accuracy"])
                allc.append(d); per_model[m].append(d)
                row.append(f"{m} {d:+.1f}")
            print(f"  {s_}: " + "; ".join(row))
        if allc:
            print(f"  mean {np.mean(allc):+.2f} over {len(allc)} cells; " + ", ".join(f"{m} {np.mean(v):+.1f}" for m, v in per_model.items()))

    print("\n### Tuned minus untuned RBFNN (paired by fold)")
    for op, opname in OPS:
        for s in SETS:
            row, cells = [], []
            for p, s_ in PARTS:
                ca, cb = cell(p, op, s, "RBFNN_tuned"), cell(p, op, s, "RBFNN")
                if ca is None or cb is None:
                    continue
                mean, lo, hi = paired(ca, cb)
                cells.append(mean)
                row.append(f"{s_} {mean:+.1f} [{lo:+.1f},{hi:+.1f}]{'*' if (lo > 0 or hi < 0) else ''}")
            if cells:
                print(f"  {opname}, set {s}: " + "; ".join(row) + f" | mean {np.mean(cells):+.2f}")
    if hp:
        print("\n### Hyper-parameters chosen by the inner CV for the tuned RBFNN (counts over patients x folds)")
        for op, _ in OPS:
            for s in SETS:
                sel = [h for h in hp if h[0] == op and h[1] == s]
                if not sel:
                    continue
                cnt = lambda i: dict(sorted(collections.Counter(h[i] for h in sel).items()))
                print(f"  {op} set {s} (n={len(sel)}): centres {cnt(2)}; width x {cnt(3)}; C {cnt(4)}")
        print("  Grid edges: centres 16/256, width 0.5/4, C 0.1/100; an edge choice means the optimum may lie outside the grid.")

    print("\n### LDA minus each other model, set C and set A at each operating point (paired by fold)")
    for op, opname in OPS:
        for s in ["A", "C"]:
            print(f"\n{opname}, set {s}")
            for other in ["SVM", "RandomForest", "RBFNN", "RBFNN_tuned"]:
                row, cells = [], []
                for p, s_ in PARTS:
                    ca, cb = cell(p, op, s, "LDA"), cell(p, op, s, other)
                    if ca is None or cb is None:
                        continue
                    mean, lo, hi = paired(ca, cb)
                    cells.append(mean)
                    row.append(f"{s_} {mean:+.1f} [{lo:+.1f},{hi:+.1f}]{'*' if (lo > 0 or hi < 0) else ''}")
                if cells:
                    print(f"  LDA - {other}: " + "; ".join(row) + f" | mean {np.mean(cells):+.2f}")

    print("\n### Run time (minutes per table)")
    for key, v in R.items():
        print(f"  {key}: {sum(c['seconds'] for c in v.values()) / 60:.1f} min")


if __name__ == "__main__":
    main()
