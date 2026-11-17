"""Fair LSTM comparison, analysis (docs/fair_lstm_declared_protocol.md): applies the causal votes to the stored raw classical predictions (cache/grid/fair/classical/), reads the LSTM rerun
(cache/grid/fair/lstm/) and writes, for the 28 winner cells: accuracy / weighted F1 / kappa / confusion matrix / fold accuracies of every method and variant, the declared comparisons H1-H3 (paired by fold,
4 df, 95% t-interval, reading rule fixed in the protocol), the vote's effect, per-patient means, time-resolved accuracy, analytic cost, and confusion matrices.
Outputs: cache/grid/fair/fair_results.json, fair_cells.csv, fair_confusion_matrices.md; the summary text goes to stdout (redirect to cache/grid/fair/fair_summary.txt).
Usage: python -W ignore analyse_fair.py > cache/grid/fair/fair_summary.txt"""
import collections
import json
import math

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from analyse_networks import costs
from config import CACHE_DIR, CUE_LEGEND
from fair_vote import K_GRID, K_NAMES, causal_vote, window_index
from grid_tables import table_path
from run_models_norest import compute_metrics

FAIR = CACHE_DIR / "grid" / "fair"
G = json.load(open(CACHE_DIR / "grid" / "grid_results.json"))
N = json.load(open(CACHE_DIR / "grid" / "networks_results.json"))
WIN = json.load(open(CACHE_DIR / "grid" / "winners.json"))
SHORT = {"P1_combined": "P1", "P2_combined": "P2", "Wallace_S1": "S1", "Wallace_S2": "S2"}
CLASSICAL = ["LDA", "SVM", "RandomForest"]
VARIANTS = [K_NAMES[k] for k in K_GRID] + ["nested"]
LSTMS = ["LSTM_default", "LSTM_tuned"]
BINS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 10 ** 9)]
BIN_NAMES = ["1-2", "3-5", "6-10", "11-20", "21+"]
TCRIT = student_t.ppf(0.975, 4)


def npz(kind, key):
    return np.load(FAIR / kind / (key.replace("|", "__") + ".npz"), allow_pickle=False)


def fmt(cfg):
    return f"{cfg[0]} Hz x {cfg[1]} ch x {cfg[2]} b"


def record(y, pred, labels, folds, widx):
    m = compute_metrics(y, pred, labels)
    fa = np.array([float((pred[folds == f] == y[folds == f]).mean()) for f in sorted(np.unique(folds))])
    ok = (pred == y)
    return dict(acc=m["accuracy"], f1=m["weighted_f1"], kappa=m["cohen_kappa"], cm=m["confusion_matrix"], fold=fa,
                bin_correct=[int(ok[(widx >= lo) & (widx <= hi)].sum()) for lo, hi in BINS], bin_total=[int(((widx >= lo) & (widx <= hi)).sum()) for lo, hi in BINS])


def build():
    cells, R, chosen = [], {}, {}
    for pf, cfgs in WIN.items():
        part, fset = pf.split("|")
        for rate, nch, bits in cfgs:
            cells.append((part, fset, (rate, nch, bits)))
    meta_cache = {}
    for part, fset, (rate, nch, bits) in cells:
        if (part, rate, bits) not in meta_cache:
            meta_cache[(part, rate, bits)] = pd.read_parquet(table_path(part, rate, bits), columns=["cue", "segment_id", "window_time_s"])
        t = meta_cache[(part, rate, bits)]
        y, seg = t["cue"].to_numpy(), t["segment_id"].to_numpy()
        widx = window_index(seg)
        cell = (part, fset, (rate, nch, bits))
        R[cell] = {}
        for model in CLASSICAL:
            z = npz("classical", f"{part}|{rate}Hz|{nch}ch|{bits}bit|{fset}|{model}")
            labels = [int(c) for c in z["labels"]]
            c2i = {c: i for i, c in enumerate(labels)}
            folds = z["folds"]
            pooled_i = np.array([c2i[c] for c in z["pooled"]])
            lab = np.array(labels)
            votes = {K_NAMES[k]: causal_vote(pooled_i, seg, k, len(labels)) for k in K_GRID}
            vn = np.empty_like(pooled_i)
            for fi, f in enumerate(sorted(np.unique(folds))):
                te = folds == f
                kk = int(z["chosen_k"][fi])
                vn[te] = causal_vote(pooled_i[te], seg[te], None if kk == 0 else kk, len(labels))
            votes["nested"] = vn
            chosen[cell + (model,)] = [int(k) for k in z["chosen_k"]]
            for v, idx in votes.items():
                R[cell][f"{model}/{v}"] = record(y, lab[idx], labels, folds, widx)
        for method in LSTMS:
            z = npz("lstm", f"{part}|{rate}Hz|{nch}ch|{bits}bit|{fset}|{method}")
            assert (z["folds"] == folds).all()
            R[cell][method] = record(y, z["pred"], labels, folds, widx)
    return cells, R, chosen


def compare(cells, R, ma, mb):
    """Paired by fold: method `ma` minus `mb` (a method name, or a function (cell, R) -> fold accuracies). Returns (mean difference in points over cells, n cells, cells resolved better, cells resolved worse)."""
    ds, rp, rn = [], 0, 0
    for c in cells:
        other = mb(c, R) if callable(mb) else R[c][mb]["fold"]
        d = R[c][ma]["fold"] - other
        h = TCRIT * d.std(ddof=1) / np.sqrt(len(d))
        ds.append(100 * d.mean())
        rp += d.mean() - h > 0
        rn += d.mean() + h < 0
    return float(np.mean(ds)), len(ds), int(rp), int(rn)


def reading(mean, n, rp, rn, pos, neg):
    """Declared reading rule: |mean| < 1 point = about equal; otherwise better/worse only if resolved in at least a third of the cells."""
    if abs(mean) < 1.0:
        return "about equal"
    need = math.ceil(n / 3)
    if mean > 0:
        return pos if rp >= need else f"unresolved ({pos} ahead, resolved in {rp} of {n} cells)"
    return neg if rn >= need else f"unresolved ({neg} ahead, resolved in {rn} of {n} cells)"


def signs(cells, R, ma, mb):
    """(cells where `ma` is ahead, cells where it is behind) by the sign of the mean fold difference. Added after the first look at the results (protocol amendment 1); descriptive only."""
    ahead = behind = 0
    for c in cells:
        other = mb(c, R) if callable(mb) else R[c][mb]["fold"]
        d = (R[c][ma]["fold"] - other).mean()
        ahead += d > 0
        behind += d < 0
    return int(ahead), int(behind)


def cmp_line(cells, R, ma, mb, pos, neg):
    mean, n, rp, rn = compare(cells, R, ma, mb)
    ahead, behind = signs(cells, R, ma, mb)
    return f"{mean:+.2f} points; LSTM ahead in {ahead}, behind in {behind} of {n} cells (resolved better {rp}, worse {rn}) -> {reading(mean, n, rp, rn, pos, neg)}"


def best_fold(R, c, variant):
    """Fold accuracies of the best (by pooled accuracy) of the three classical models at one cell, for one vote variant."""
    best = max(CLASSICAL, key=lambda m: R[c][f"{m}/{variant}"]["acc"])
    return R[c][f"{best}/{variant}"]["fold"]


def main():
    cells, R, chosen = build()
    scopes = [("all 28 cells", cells), ("set A (14 cells)", [c for c in cells if c[1] == "A"]), ("set W (14 cells)", [c for c in cells if c[1] == "W"])]

    print("### 1. Every cell: accuracy % (weighted F1 %, kappa) of LSTM and of the classical models without vote (k1), with the vote over all windows since the onset, and with the nested vote")
    for part, fset, cfg in cells:
        c = (part, fset, cfg)
        print(f"\n{SHORT[part]} set {fset}, {fmt(cfg)} = {cfg[0] * cfg[1] * cfg[2] / 1000:.0f} kb/s")
        for m in LSTMS + [f"{mo}/{v}" for mo in CLASSICAL for v in ("k1", "all", "nested")]:
            r = R[c][m]
            extra = ""
            if m.endswith("/nested"):
                extra = f"  k per fold {['all' if k == 0 else k for k in chosen[c + (m.split('/')[0],)]]}"
            print(f"  {m:20s} acc {100 * r['acc']:5.1f}  F1 {100 * r['f1']:5.1f}  kappa {r['kappa']:.3f}{extra}")

    print("\n### 2. Mean accuracy (%) over the cells, by method and vote variant")
    for name, cs in scopes:
        print(f"\n  {name}")
        print("  " + "model".ljust(14) + "".join(v.rjust(8) for v in VARIANTS))
        for mo in CLASSICAL:
            print("  " + mo.ljust(14) + "".join(f"{100 * np.mean([R[c][f'{mo}/{v}']['acc'] for c in cs]):8.2f}" for v in VARIANTS))
        print("  " + "best of three".ljust(14) + "".join(f"{100 * np.mean([max(R[c][f'{mo}/{v}']['acc'] for mo in CLASSICAL) for c in cs]):8.2f}" for v in VARIANTS))
        print("  " + "  ".join(f"{m} {100 * np.mean([R[c][m]['acc'] for c in cs]):.2f}" for m in LSTMS))

    print("\n### 3. Effect of the vote: variant minus no vote (k1), same model, paired by fold (points; mean over cells; resolved better / worse cells)")
    for name, cs in scopes:
        print(f"\n  {name}")
        for mo in CLASSICAL:
            parts = []
            for v in VARIANTS[1:]:
                mean, _, rp, rn = compare(cs, R, f"{mo}/{v}", f"{mo}/k1")
                parts.append(f"{v}: {mean:+.2f} ({rp}/{rn})")
            print("  " + mo.ljust(14) + "  ".join(parts))

    print("\n### 4. H1 (main): LSTM minus the best of the three classical models, nested vote, paired by fold; and the same against the best of three without vote (the earlier comparison)")
    for lstm in ("LSTM_tuned", "LSTM_default"):
        for name, cs in scopes:
            for variant, label in (("nested", "best classical with nested vote"), ("k1", "best classical without vote (as in the first run)")):
                print(f"  {lstm:13s} minus {label:48s} {name:18s}: " + cmp_line(cs, R, lstm, lambda c, R_, v=variant: best_fold(R_, c, v), "LSTM better", "classical better"))

    print("\n### 5. H2/H3 (information-matched, no free parameter): LSTM minus each classical model with the vote over all windows since the onset")
    for lstm in ("LSTM_tuned", "LSTM_default"):
        for name, cs in scopes:
            for mo in CLASSICAL:
                print(f"  {lstm:13s} minus {mo + '/all':18s} {name:18s}: " + cmp_line(cs, R, lstm, f"{mo}/all", "LSTM better", "classical better"))
            print(f"  {lstm:13s} minus {'best of three/all':18s} {name:18s}: " + cmp_line(cs, R, lstm, lambda c, R_: best_fold(R_, c, "all"), "LSTM better", "classical better"))
    mean, n, rp, rn = compare(cells, R, "LSTM_tuned", "LSTM_default")
    print(f"\n  (for reference) LSTM_tuned minus LSTM_default, all 28 cells: {mean:+.2f}, better in {rp}, worse in {rn}")

    print("\n### 5b. Per patient (protocol amendment 1): LSTM minus the best of three classical models, by patient (mean over that patient's cells; cells of one patient are not independent)")
    for lstm in ("LSTM_tuned", "LSTM_default"):
        for variant, label in (("nested", "nested vote"), ("all", "vote over all windows"), ("k1", "no vote")):
            parts = []
            for part in SHORT:
                cs = [c for c in cells if c[0] == part]
                mean, n, rp, rn = compare(cs, R, lstm, lambda c, R_, v=variant: best_fold(R_, c, v))
                ahead, behind = signs(cs, R, lstm, lambda c, R_, v=variant: best_fold(R_, c, v))
                parts.append(f"{SHORT[part]} {mean:+.2f} (ahead {ahead}/behind {behind} of {n})")
            print(f"  {lstm:13s} minus best classical, {label:22s}: " + "; ".join(parts))

    print("\n### 6. Per patient: mean accuracy (%) over that patient's cells (sets A and W together)")
    cols = ["LDA/k1", "LDA/nested", "SVM/k1", "SVM/nested", "RandomForest/k1", "RandomForest/nested", "LDA/all", "SVM/all", "RandomForest/all", "LSTM_default", "LSTM_tuned"]
    print("  " + "patient".ljust(8) + "".join(c.replace("RandomForest", "RF").rjust(14) for c in cols))
    for part in SHORT:
        cs = [c for c in cells if c[0] == part]
        print("  " + f"{SHORT[part]} (n={len(cs)})".ljust(8) + "".join(f"{100 * np.mean([R[c][m]['acc'] for c in cs]):14.2f}" for m in cols))
    print("\n  weighted F1 (%) and kappa, same columns")
    for part in SHORT:
        cs = [c for c in cells if c[0] == part]
        print("  " + f"{SHORT[part]} F1".ljust(8) + "".join(f"{100 * np.mean([R[c][m]['f1'] for c in cs]):14.2f}" for m in cols))
        print("  " + f"{SHORT[part]} kappa".ljust(8) + "".join(f"{np.mean([R[c][m]['kappa'] for c in cs]):14.3f}" for m in cols))

    print("\n### 7. Which k the nested choice picked (counts over cells x outer folds; 'all' = whole history since the onset)")
    for mo in CLASSICAL:
        cnt = collections.Counter(("all" if k == 0 else k) for c in cells for k in chosen[c + (mo,)])
        print(f"  {mo:13s} " + ", ".join(f"{k}: {cnt[k]}" for k in [1, 3, 5, 9, 15, "all"]))

    print("\n### 8. Time-resolved accuracy (%): pooled per cell by window index since the onset (window i is decided 200 + 50 (i-1) ms after the onset), then the mean over the 28 cells")
    methods = ["LDA/k1", "LDA/all", "SVM/k1", "SVM/all", "RandomForest/k1", "RandomForest/all", "LSTM_default", "LSTM_tuned"]
    tot = np.sum([R[c]["LDA/k1"]["bin_total"] for c in cells], axis=0)
    print("  " + "bin".ljust(18) + "".join(b.rjust(9) for b in BIN_NAMES))
    print("  " + "share of windows".ljust(18) + "".join(f"{100 * x / tot.sum():8.1f}%" for x in tot))
    for m in methods:
        row = []
        for b in range(len(BINS)):
            vals = [R[c][m]["bin_correct"][b] / R[c][m]["bin_total"][b] for c in cells if R[c][m]["bin_total"][b] > 0]
            row.append(100 * np.mean(vals))
        print("  " + m.replace("RandomForest", "RF").ljust(18) + "".join(f"{x:9.1f}" for x in row))

    print("\n### 9. Cost per decision (analytic counts, mean over the cells of each set; kFLOP = extraction + 2 x model MACs + comparisons; the vote adds C + 1 integer operations, shown separately)")
    for name, cs in scopes[1:]:
        ex, tot_m, vote = [], collections.defaultdict(list), []
        for c in cs:
            part, fset, cfg = c
            key = lambda mo: f"{part}|{cfg[0]}Hz|{cfg[1]}ch|{cfg[2]}bit|{fset}|{mo}"
            for mo in CLASSICAL:
                _, _, e, tt = costs(G[key(mo)], mo)
                tot_m[mo].append(tt)
            for mo in LSTMS:
                tot_m[mo].append(costs(N[key(mo)], mo)[3])
            ex.append(G[key("LDA")]["extraction"]["flops"])
            vote.append(len(G[key("LDA")]["class_labels"]) + 1)
        print(f"  {name}: extraction {np.mean(ex) / 1000:.1f} kFLOP; " + "; ".join(f"{m.replace('RandomForest', 'RF')} {np.mean(v) / 1000:.1f}" for m, v in tot_m.items()) + f"; vote {np.mean(vote):.0f} integer ops (about {100 * np.mean(vote) / np.mean(tot_m['LDA']):.2f}% of the LDA total)")
    print("  Memory for the history: the vote keeps k labels (k = 15 at most, or a running count per class for 'all') plus C counters; the LSTM keeps 2 x 64 state values. Counts under stated conventions, not measurements.")

    # files
    out = {}
    for c in cells:
        part, fset, (rate, nch, bits) = c
        for m, r in R[c].items():
            out[f"{part}|{rate}Hz|{nch}ch|{bits}bit|{fset}|{m}"] = dict(accuracy=r["acc"], weighted_f1=r["f1"], cohen_kappa=r["kappa"], confusion_matrix=r["cm"], fold_accs=r["fold"].tolist(),
                                                                       bin_correct=r["bin_correct"], bin_total=r["bin_total"])
    nested_k = {f"{c[0]}|{c[2][0]}Hz|{c[2][1]}ch|{c[2][2]}bit|{c[1]}|{mo}": chosen[c + (mo,)] for c in cells for mo in CLASSICAL}      # 0 = all since the onset
    json.dump(dict(results=out, nested_k=nested_k), open(FAIR / "fair_results.json", "w"))
    rows = []
    for c in cells:
        part, fset, (rate, nch, bits) = c
        for m, r in R[c].items():
            rows.append(dict(patient=SHORT[part], set=fset, rate_hz=rate, channels=nch, bits=bits, kbps=rate * nch * bits / 1000, method=m, accuracy=r["acc"], weighted_f1=r["f1"], kappa=r["kappa"]))
    pd.DataFrame(rows).to_csv(FAIR / "fair_cells.csv", index=False)
    names = {int(k): v for k, v in CUE_LEGEND.items()}
    with open(FAIR / "fair_confusion_matrices.md", "w", encoding="utf-8") as fh:
        fh.write("# Confusion matrices of the fair comparison at the 28 winner cells (pooled out-of-fold windows; rows = true class, columns = predicted class)\n\n")
        for c in cells:
            part, fset, cfg = c
            labels = [int(l) for l in G[f"{part}|{cfg[0]}Hz|{cfg[1]}ch|{cfg[2]}bit|{fset}|LDA"]["class_labels"]]
            for m in ["LSTM_tuned", "LDA/nested", "SVM/nested", "RandomForest/nested"]:
                r = R[c][m]
                fh.write(f"## {SHORT[part]} set {fset}, {fmt(cfg)}, {m}: accuracy {100 * r['acc']:.1f}%, weighted F1 {100 * r['f1']:.1f}%, kappa {r['kappa']:.3f}\n\n")
                fh.write("| true \\ predicted | " + " | ".join(names.get(l, str(l)) for l in labels) + " |\n|---|" + "---|" * len(labels) + "\n")
                for l, row in zip(labels, r["cm"]):
                    fh.write(f"| {names.get(l, l)} | " + " | ".join(str(int(v)) for v in row) + " |\n")
                fh.write("\n")
    print("\nFiles: cache/grid/fair/fair_results.json, fair_cells.csv, fair_confusion_matrices.md")


if __name__ == "__main__":
    main()
