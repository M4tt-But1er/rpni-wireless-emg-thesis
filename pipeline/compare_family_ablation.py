"""Summarise family_ablation.py exactly as declared in docs/family_ablation_declared_protocol.md. Redirect to cache/family_ablation_summary.txt.
Differences are paired by fold (same folds for every subset of a table): mean with a 95% t-interval over 5 folds (4 df); '*' = excludes 0. 'resolved' counts are over the 8 tables (4 patients x ref/low)."""
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from build_wide_tables import OPS, PARTS, TAG, table_path
from config import CACHE_DIR
from families import BRANCHES, LEAVES, MULTI_LEAF_BRANCHES, leaf_of
from wide_features import set_columns

R = json.load(open(CACHE_DIR / "family_ablation_results.json"))
MODELS = ["LDA", "SVM"]
TCRIT = student_t.ppf(0.975, 4)
TABLES = [(p, o) for p in PARTS for o in OPS]
SHORT = {"P1_combined": "P1", "P2_combined": "P2", "Wallace_S1": "S1", "Wallace_S2": "S2"}


def cell(p, o, s, m):
    return R.get(f"{p}|{o}|{s}|{m}")


def paired(a, b):
    d = np.array(a["fold_accs"]) - np.array(b["fold_accs"])
    h = TCRIT * d.std(ddof=1) / np.sqrt(len(d))
    return 100 * d.mean(), 100 * (d.mean() - h), 100 * (d.mean() + h)


def agg(subset, ref, model):
    """Mean paired difference of `subset` against `ref` over the tables that have both; counts of resolved better/worse."""
    ds, rp, rn = [], 0, 0
    for p, o in TABLES:
        a, b = cell(p, o, subset, model), cell(p, o, ref, model)
        if a is None or b is None:
            continue
        d, lo, hi = paired(a, b)
        ds.append(d)
        rp += lo > 0
        rn += hi < 0
    return (float(np.mean(ds)) if ds else float("nan")), len(ds), rp, rn


def label(drop_l, drop_s, add_l, add_s):
    disp = abs(drop_l) <= 0.3 and abs(drop_s) <= 0.3
    use = add_l >= 0.5 and add_s >= 0.5
    return "useful and not dispensable" if use and not disp else "useful" if use else "dispensable" if disp else "mixed"


def main():
    cols = list(pd.read_parquet(table_path("P1_combined", "ref")).columns[5:])
    ncol = {l: sum(leaf_of(c) == l for c in cols) for l in LEAVES}
    print(f"Cells present: {len(R)} of {28 * 2 * 8}\n")

    print("### Reference cells (pooled accuracy %): A and C, LDA / SVM")
    for p, o in TABLES:
        print(f"  {SHORT[p]} {o}: " + "; ".join(f"{s} {m} {100 * cell(p, o, s, m)['accuracy']:.1f}" for s in ("A", "C") for m in MODELS if cell(p, o, s, m)))

    print("\n### V6: A and C cells equal the Stage 1 cells (same tables, same folds, same models)")
    s1 = json.load(open(CACHE_DIR / "stage1_ops_results_fixedrange.json"))
    ok = True
    for p, o in TABLES:
        for s in ("A", "C"):
            for m in MODELS:
                c = cell(p, o, s, m)
                if c is None:
                    continue
                ref = s1[f"{p}|{o}"][f"{s}|{m}"]
                same = abs(c["accuracy"] - ref["accuracy"]) < 1e-12 and c["n_features"] == ref["n_features"]
                ok &= same
                if not same:
                    print(f"  MISMATCH {p} {o} {s} {m}: {c['accuracy']} vs {ref['accuracy']}")
    print("  all equal" if ok else "  SOME DIFFER")

    def block(title, items, kind):
        print(f"\n### {title}")
        print("| Family | Columns (P1 ref) | LDA mean diff | resolved +/- | SVM mean diff | resolved +/- |")
        print("|---|---|---|---|---|---|")
        rows = {}
        for name in items:
            sub = f"{kind}:{name}"
            ref = "C" if kind == "drop" else "A"
            out = []
            for m in MODELS:
                d, n, rp, rn = agg(sub, ref, m)
                rows[(name, m)] = d
                out += [f"{d:+.2f}" if n else "n/a", f"{rp}/{rn} of {n}"]
            nc = ncol.get(name, "")
            print(f"| {name} | {nc} | " + " | ".join(out) + " |")
        return rows

    leaves = list(LEAVES)
    drop = block("Drop one leaf family from C (mean difference against C, points; negative = the family was useful)", leaves, "drop")
    add = block("Add one leaf family's columns to A (mean difference against A, points; positive = the family is useful)", leaves, "add")
    block("Drop a whole branch from C (branches with more than one leaf)", MULTI_LEAF_BRANCHES, "drop")
    block("Add a whole branch to A (branches with more than one leaf)", MULTI_LEAF_BRANCHES, "add")

    print("\n### Descriptive labels (rules fixed in the protocol: dispensable = |mean drop| <= 0.3 for LDA and SVM; useful = mean add >= +0.5 for LDA and SVM)")
    for l in leaves:
        print(f"  {l:18s} drop LDA {drop[(l, 'LDA')]:+.2f} SVM {drop[(l, 'SVM')]:+.2f} | add LDA {add[(l, 'LDA')]:+.2f} SVM {add[(l, 'SVM')]:+.2f} -> "
              + label(drop[(l, 'LDA')], drop[(l, 'SVM')], add[(l, 'LDA')], add[(l, 'SVM')]))

    for kind, ref, title in (("drop", "C", "Drop one leaf from C"), ("add", "A", "Add one leaf to A")):
        for m in MODELS:
            print(f"\n### {title}: per table, {m} (points; * = interval excludes 0)")
            print("| Family | " + " | ".join(f"{SHORT[p]} {o}" for p, o in TABLES) + " |")
            print("|---|" + "---|" * len(TABLES))
            for l in leaves:
                row = []
                for p, o in TABLES:
                    a, b = cell(p, o, f"{kind}:{l}", m), cell(p, o, ref, m)
                    if a is None or b is None:
                        row.append("n/a")
                        continue
                    d, lo, hi = paired(a, b)
                    row.append(f"{d:+.1f}{'*' if lo > 0 or hi < 0 else ''}")
                print(f"| {l} | " + " | ".join(row) + " |")

    print("\n### Run time (core-minutes per model)")
    for m in MODELS:
        print(f"  {m}: {sum(v['seconds'] for k, v in R.items() if k.endswith('|' + m)) / 60:.0f} min over {sum(k.endswith('|' + m) for k in R)} cells")


if __name__ == "__main__":
    main()
