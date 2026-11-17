"""Step 3 summary (docs/grid_declared_protocol.md): at every winner configuration of the grid, LDA / SVM / RF (grid cells) next to the tuned RBFNN, the LSTM (headline setting) and the tuned LSTM
(cache/grid/networks_results.json): accuracy, weighted F1, kappa, FLOPs and MACs per decision, and paired differences by fold (all models share the outer folds; the LSTM scores the same windows, taken
as whole segments). Also writes the confusion matrices of every model at every winner to cache/grid/confusion_matrices.md.
Usage: python -W ignore analyse_networks.py > cache/grid/networks_summary.txt"""
import collections
import json

import numpy as np
from scipy.stats import t as student_t

from build_wide_tables import PARTS
from config import CACHE_DIR, CUE_LEGEND
from feature_cost import DECISIONS_PER_SECOND, model_cost

G = json.load(open(CACHE_DIR / "grid" / "grid_results.json"))
N = json.load(open(CACHE_DIR / "grid" / "networks_results.json"))
WIN = json.load(open(CACHE_DIR / "grid" / "winners.json"))
SHORT = {"P1_combined": "P1", "P2_combined": "P2", "Wallace_S1": "S1", "Wallace_S2": "S2"}
MODELS = ["LDA", "SVM", "RandomForest", "RBFNN_tuned", "LSTM_default", "LSTM_tuned"]
TCRIT = student_t.ppf(0.975, 4)


def cellrec(part, rate, nch, bits, fset, model):
    key = f"{part}|{rate}Hz|{nch}ch|{bits}bit|{fset}|{model}"
    return (N if model in ("RBFNN_tuned", "LSTM_default", "LSTM_tuned") else G).get(key)


def costs(rec, model):
    C = len(rec["class_labels"])
    F = rec["n_features"]
    info = rec.get("fold_info") or [{}]
    if model == "SVM":
        mc = model_cost("SVM", F, C, n_sv=float(np.mean([i["n_sv"] for i in info])))
    elif model == "RandomForest":
        mc = model_cost("RandomForest", F, C, trees=int(info[0]["trees"]), max_depth=int(max(i["max_depth"] for i in info)))
    elif model == "RBFNN_tuned":
        mc = model_cost("RBFNN", F, C, centres=float(np.mean([i["n_centers"] for i in info])))
    elif model.startswith("LSTM"):
        mc = model_cost("LSTM", F, C, hidden=64)
    else:
        mc = model_cost("LDA", F, C)
    ex = rec["extraction"]["flops"]
    return mc["macs"], mc["macs"] + mc["other_ops"], ex, ex + 2 * mc["macs"] + mc["other_ops"]


def fmt(cfg):
    return f"{cfg[0]} Hz x {cfg[1]} ch x {cfg[2]} b"


def main():
    rows = []
    print("### 1. All models at every winner configuration (accuracy %, weighted F1 %, kappa; cost per decision: model MACs, total FLOPs = extraction + 2 x MACs + comparisons)")
    for pf, cfgs in WIN.items():
        part, fset = pf.split("|")
        for rate, nch, bits in cfgs:
            print(f"\n{SHORT[part]} set {fset}, {fmt((rate, nch, bits))} = {rate * nch * bits / 1000:.0f} kb/s")
            for m in MODELS:
                r = cellrec(part, rate, nch, bits, fset, m)
                if r is None:
                    print(f"  {m:13s} not run")
                    continue
                macs, ops, ex, tot = costs(r, m)
                extra = ""
                if m == "RBFNN_tuned":
                    extra = f" centres {np.mean([i['n_centers'] for i in r['fold_info']]):.0f}"
                if m == "LSTM_tuned":
                    extra = " configs/epochs " + ",".join(f"{i['config']}/{i['epochs']}" for i in r["fold_info"])
                print(f"  {m:13s} acc {100 * r['accuracy']:.1f} F1 {100 * r['weighted_f1']:.1f} kappa {r['cohen_kappa']:.3f} | F={r['n_features']:4d} | model {macs / 1000:8.1f} kMAC, total {tot / 1000:8.0f} kFLOP{extra}")
                rows.append(dict(part=part, fset=fset, cfg=(rate, nch, bits), model=m, acc=r["accuracy"], fold=np.array(r["fold_accs"])))

    print("\n### 2. Mean accuracy over the winner cells, per feature set and model (%)")
    acc = collections.defaultdict(list)
    for r in rows:
        acc[(r["fset"], r["model"])].append(100 * r["acc"])
    for fset in ("A", "W"):
        print(f"  set {fset}: " + ", ".join(f"{m} {np.mean(acc[(fset, m)]):.2f} (n={len(acc[(fset, m)])})" for m in MODELS if acc[(fset, m)]))

    print("\n### 3. Paired by fold: each network minus the best classical model at the same cell, and minus LDA (points; mean over cells; resolved = interval excludes 0)")
    by_cell = collections.defaultdict(dict)
    for r in rows:
        by_cell[(r["part"], r["fset"], r["cfg"])][r["model"]] = r
    for net in ("RBFNN_tuned", "LSTM_default", "LSTM_tuned"):
        for ref in ("LDA", "SVM", "RandomForest", "best"):
            ds, rp, rn = [], 0, 0
            for cell, d in by_cell.items():
                if net not in d:
                    continue
                if ref == "best":
                    cands = [d[m] for m in ("LDA", "SVM", "RandomForest") if m in d]
                    if not cands:
                        continue
                    other = max(cands, key=lambda x: x["acc"])
                elif ref in d:
                    other = d[ref]
                else:
                    continue
                diff = d[net]["fold"] - other["fold"]
                h = TCRIT * diff.std(ddof=1) / np.sqrt(len(diff))
                ds.append(100 * diff.mean())
                rp += diff.mean() - h > 0
                rn += diff.mean() + h < 0
            if ds:
                print(f"  {net:13s} minus {ref:12s}: mean {np.mean(ds):+.2f} over {len(ds)} cells; resolved better {rp}, worse {rn}")

    print("\n### 4. Tuned RBFNN hyper-parameters chosen (counts over all winner cells x folds)")
    ch = collections.Counter()
    for k, r in N.items():
        if k.endswith("RBFNN_tuned"):
            for i in r["fold_info"]:
                ch[("centres", i["n_centers"])] += 1
                ch[("width", i["sigma_scale"])] += 1
                ch[("C", i["C"])] += 1
    for name in ("centres", "width", "C"):
        print(f"  {name}: " + ", ".join(f"{v}: {c}" for (n, v), c in sorted(ch.items()) if n == name))

    names = {int(k): v for k, v in CUE_LEGEND.items()}
    with open(CACHE_DIR / "grid" / "confusion_matrices.md", "w", encoding="utf-8") as fh:
        fh.write("# Confusion matrices at the winner configurations (pooled out-of-fold; rows = true class, columns = predicted class)\n\n")
        for pf, cfgs in WIN.items():
            part, fset = pf.split("|")
            for rate, nch, bits in cfgs:
                for m in MODELS:
                    r = cellrec(part, rate, nch, bits, fset, m)
                    if r is None:
                        continue
                    labels = r["class_labels"]
                    fh.write(f"## {SHORT[part]} set {fset}, {fmt((rate, nch, bits))}, {m}: accuracy {100 * r['accuracy']:.1f}%, weighted F1 {100 * r['weighted_f1']:.1f}%, kappa {r['cohen_kappa']:.3f}\n\n")
                    fh.write("| true \\ predicted | " + " | ".join(f"{names.get(int(l), l)}" for l in labels) + " |\n|---|" + "---|" * len(labels) + "\n")
                    for l, row in zip(labels, r["confusion_matrix"]):
                        fh.write(f"| {names.get(int(l), l)} | " + " | ".join(str(int(v)) for v in row) + " |\n")
                    fh.write("\n")
    print("\nConfusion matrices: cache/grid/confusion_matrices.md")


if __name__ == "__main__":
    main()
