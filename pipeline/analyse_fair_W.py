"""E5 (temporal context at the winner cells) re-summarised on the 14 cells of the wide set W only (user, 11 Nov 2026: W for everything except the
feature-set comparisons). Uses analyse_fair2.build() on the stored predictions; nothing is re-run. Output: cache/grid/fair2/fair_W_summary.md.
Usage: python -W ignore analyse_fair_W.py"""
import json

import numpy as np

import analyse_fair2 as F
from analyse_fair import BIN_NAMES, BINS, CLASSICAL, G, SHORT
from config import CACHE_DIR
from feature_cost import model_cost

NET = json.load(open(CACHE_DIR / "grid" / "networks_results.json"))
OUT = CACHE_DIR / "grid" / "fair2" / "fair_W_summary.md"


def acc(R, cs, m):
    return 100 * np.mean([R[c][m]["acc"] for c in cs])


def main():
    cells, R, choices, fixed, info, lstm_meta = F.build()
    W = [c for c in cells if c[1] == "W"]
    L = [f"# E5 on the 14 W winner cells (analyse_fair_W.py)\n", "Cells: " + "; ".join(f"{SHORT[c[0]]} {c[2][0]}Hz x {c[2][1]}ch x {c[2][2]}b" for c in W) + "\n"]
    by = {pt: [c for c in W if c[0] == pt] for pt in SHORT}
    rb = {c: NET.get(f"{c[0]}|{c[2][0]}Hz|{c[2][1]}ch|{c[2][2]}bit|W|RBFNN_tuned", {}).get("accuracy") for c in W}

    # table like tab:res_e5
    rows = [("LDA, no vote", "LDA/P0"), ("SVM, no vote", "SVM/P0"), ("RF, no vote", "RandomForest/P0"),
            ("LDA + vote (P3)", "LDA/P3"), ("SVM + vote (P3)", "SVM/P3"), ("RF + vote (P3)", "RandomForest/P3"),
            ("LDA + hard vote, nested k (P1)", "LDA/P1"), ("SVM + hard vote, nested k (P1)", "SVM/P1"), ("RF + hard vote, nested k (P1)", "RandomForest/P1"),
            ("LSTM, 3-setting tuning", "LSTM_tuned"), ("LSTM, default", "LSTM_default"), ("LSTM, search, 1 seed", "LSTM_best"), ("LSTM, search, 5 seeds", "LSTM_best_ens5")]
    L.append("## Accuracy (%) per participant (W cells) and all 14\n")
    L.append("| Method | " + " | ".join(f"{SHORT[p]} ({len(by[p])})" for p in SHORT) + " | All 14 |\n|---|" + "---|" * (len(SHORT) + 1))
    L.append("| RBFNN tuned, no vote | " + " | ".join(f"{100 * np.mean([rb[c] for c in by[p]]):.1f}" for p in SHORT) + f" | {100 * np.mean(list(rb.values())):.1f} |")
    for name, m in rows:
        L.append(f"| {name} | " + " | ".join(f"{acc(R, by[p], m):.1f}" for p in SHORT) + f" | {acc(R, W, m):.1f} |")
    best_p0 = 100 * np.mean([max(R[c][f"{mo}/P0"]["acc"] for mo in CLASSICAL) for c in W])
    best_p3 = 100 * np.mean([max(R[c][f"{mo}/P3"]["acc"] for mo in CLASSICAL) for c in W])
    L.append(f"\nBest of the three per cell: no vote {best_p0:.1f}, with vote (P3) {best_p3:.1f}.")
    L.append(f"RBFNN tuned minus LDA (no vote): " + ", ".join(f"{SHORT[p]} {100 * np.mean([rb[c] - R[c]['LDA/P0']['acc'] for c in by[p]]):+.1f}" for p in SHORT)
             + f"; all {100 * np.mean([rb[c] - R[c]['LDA/P0']['acc'] for c in W]):+.1f}")

    L.append("\n## Paired comparisons on the 14 W cells (mean points; ahead/behind; resolved better/worse; reading by the declared rule)\n")
    L.append("| comparison | mean | ahead/behind | resolved | reading |\n|---|---|---|---|---|")
    for lab, ma, mb in [("LSTM_tuned minus best classical, no vote", "LSTM_tuned", lambda c, R_: F.best_fold(R_, c, "P0")),
                        ("LSTM_tuned minus best classical with vote (P3)", "LSTM_tuned", lambda c, R_: F.best_fold(R_, c, "P3")),
                        ("LSTM_best (search, 1 seed) minus best classical with vote (P3)", "LSTM_best", lambda c, R_: F.best_fold(R_, c, "P3")),
                        ("LSTM_best_ens5 minus best classical with vote (P3)", "LSTM_best_ens5", lambda c, R_: F.best_fold(R_, c, "P3")),
                        ("LSTM_best_ens5 minus LDA/P3", "LSTM_best_ens5", "LDA/P3"), ("LSTM_best_ens5 minus SVM/P3", "LSTM_best_ens5", "SVM/P3"),
                        ("LSTM_best_ens5 minus RF/P3", "LSTM_best_ens5", "RandomForest/P3"),
                        ("LSTM_best minus LSTM_tuned", "LSTM_best", "LSTM_tuned"),
                        ("LDA/P3 minus LDA/P1 (soft adds to hard?)", "LDA/P3", "LDA/P1"), ("SVM/P3 minus SVM/P1", "SVM/P3", "SVM/P1"), ("RF/P3 minus RF/P1", "RandomForest/P3", "RandomForest/P1")]:
        mean, ab, res, rd = F.cmp_row(W, R, ma, mb, "first better", "second better")
        L.append(f"| {lab} | {mean} | {ab} | {res} | {rd} |")
    for mo in CLASSICAL:
        g = [100 * (R[c][f"{mo}/P1"]["acc"] - R[c][f"{mo}/P0"]["acc"]) for c in W]
        never = all(R[c][f"{mo}/P1"]["acc"] >= R[c][f"{mo}/P0"]["acc"] for c in W)
        L.append(f"\nVote gain {mo} (P1 minus P0): mean {np.mean(g):+.2f}; vote never lower in any cell: {never}")
    for v in ("ST3", "ST5", "CUM"):
        L.append(f"Stack {v}, no vote, minus P0: " + ", ".join(f"{SHORT.get(mo, mo)} {acc(R, W, f'{mo}/stack:{v}') - acc(R, W, f'{mo}/P0'):+.2f}" for mo in CLASSICAL)
                 + f"; with vote minus P3: " + ", ".join(f"{acc(R, W, f'{mo}/stackvote:{v}') - acc(R, W, f'{mo}/P3'):+.2f}" for mo in CLASSICAL))
    allk = [k for (c, mo, pn), ch in choices.items() if c in W and pn == "P1" for k in ch]
    L.append(f"\nNested hard vote (P1): share of fold choices that took the full history: {100 * np.mean([x[1] == 'hard_all' for x in allk]):.0f}% ({len(allk)} choices)")

    pil = [c for c in W if tuple(c[2]) == (1000, 8, 16)]
    non = [c for c in W if tuple(c[2]) != (1000, 8, 16)]
    L.append(f"\n## Pilot (reference) cells ({len(pil)}) against the other W cells ({len(non)})\n")
    for name, cs in (("pilot", pil), ("non-pilot", non)):
        L.append(f"- {name}: best classical with vote (P3) {100 * np.mean([max(R[c][f'{mo}/P3']['acc'] for mo in CLASSICAL) for c in cs]):.1f}; "
                 f"LSTM_tuned {acc(R, cs, 'LSTM_tuned'):.1f}; LSTM_best {acc(R, cs, 'LSTM_best'):.1f}; LSTM_r1 {acc(R, cs, 'LSTM_r1'):.1f}; LSTM_best_ens5 {acc(R, cs, 'LSTM_best_ens5'):.1f}")
    inner_over = []
    for c in non:
        m = lstm_meta.get((c, "LSTM_best"))
        if m and "inner" in m:
            inner_over.append(np.mean(m["inner"]) - R[c]["LSTM_best"]["acc"])
    if inner_over:
        L.append(f"- LSTM_best inner-validation minus test accuracy on the non-pilot W cells: {100 * np.mean(inner_over):+.1f} points")

    L.append("\n## Time-resolved accuracy (%), mean over the 14 W cells\n")
    tot = np.sum([R[c]["LDA/P0"]["bin_total"] for c in W], axis=0)
    L.append("| method | " + " | ".join(BIN_NAMES) + " |\n|---|" + "---|" * len(BINS))
    L.append("| share of windows | " + " | ".join(f"{100 * x / tot.sum():.1f}%" for x in tot) + " |")
    for m in ["LDA/P0", "LDA/P3", "SVM/P0", "SVM/P3", "RandomForest/P0", "RandomForest/P3", "LSTM_tuned", "LSTM_best", "LSTM_best_ens5", "LSTM_best+hard_all", "LSTM_tuned"]:
        L.append(f"| {m} | " + " | ".join(f"{100 * np.mean([R[c][m]['bin_correct'][b] / R[c][m]['bin_total'][b] for c in W if R[c][m]['bin_total'][b] > 0]):.1f}" for b in range(len(BINS))) + " |")
    L.append("LSTM with a vote on top, minus LSTM: " + ", ".join(f"{m} {acc(R, W, m + '+hard_all') - acc(R, W, m):+.2f}" for m in ("LSTM_best", "LSTM_best_ens5")))

    L.append("\n## Cost per decision on the W cells (kFLOP; same counting as analyse_fair2 F14)\n")
    for mo in CLASSICAL:
        vals = []
        for c in W:
            part, fset, cfg = c
            ex = G[F.key_of(part, fset, cfg, "LDA")]["extraction"]["flops"]
            Cn = len(G[F.key_of(part, fset, cfg, "LDA")]["class_labels"])
            Fn = int(np.load(F.FAIR2 / "classical" / (F.key_of(part, fset, cfg, mo).replace("|", "__") + ".npz"))["n_features"])
            if mo == "LDA":
                mc = model_cost("LDA", Fn, Cn)
            elif mo == "SVM":
                mc = model_cost("SVM", Fn, Cn, n_sv=float(np.mean([i["n_sv"] for i in info[(c, mo)]["none"]])))
            else:
                fi = G[F.key_of(part, fset, cfg, "RandomForest")]["fold_info"]
                mc = model_cost("RandomForest", Fn, Cn, trees=int(fi[0]["trees"]), max_depth=int(max(i["max_depth"] for i in fi)))
            vals.append(ex + 2 * mc["macs"] + mc["other_ops"])
        L.append(f"- {mo}: {np.mean(vals) / 1000:.1f} kFLOP per decision")
    from analyse_networks import costs as net_costs
    import lstm_tuning as T
    for m in ("RBFNN_tuned", "LSTM_tuned"):
        tot = [net_costs(NET[f"{c[0]}|{c[2][0]}Hz|{c[2][1]}ch|{c[2][2]}bit|W|{m}"], m)[3] for c in W]
        L.append(f"- {m}: {np.mean(tot) / 1000:.1f} kFLOP per decision ({len(tot)} cells)")
    single, ens = [], []
    for c in W:
        part, fset, cfg = c
        ex = G[F.key_of(part, fset, cfg, "LDA")]["extraction"]["flops"]
        Cn = len(G[F.key_of(part, fset, cfg, "LDA")]["class_labels"])
        m = lstm_meta[(c, "LSTM_best")]
        macs = float(np.mean([T.macs_per_step(m["features"], cf, Cn) + m["features"] for cf in m["cfgs"]]))
        single.append(ex + 2 * macs)
        ens.append(ex + 2 * macs * T.N_SEEDS)
    L.append(f"- LSTM_best (search, 1 seed): {np.mean(single) / 1000:.1f} kFLOP; 5 seeds: {np.mean(ens) / 1000:.1f} kFLOP")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
