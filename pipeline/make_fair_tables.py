"""Markdown tables of the fair LSTM comparison for docs/fair_lstm_results.md, generated from the stored predictions (no hand-typed numbers): cache/grid/fair/fair_tables.md.
Usage: python -W ignore make_fair_tables.py"""
import collections

import numpy as np

import analyse_fair as A
from analyse_networks import costs
from config import CACHE_DIR

OUT = CACHE_DIR / "grid" / "fair" / "fair_tables.md"
VN = {"k1": "no vote", "k3": "k=3", "k5": "k=5", "k9": "k=9", "k15": "k=15", "all": "all since onset", "nested": "nested k"}


def main():
    cells, R, chosen = A.build()
    scopes = [("all 28 cells", cells), ("set A, 14 cells", [c for c in cells if c[1] == "A"]), ("set W, 14 cells", [c for c in cells if c[1] == "W"])]
    L = []

    L.append("### F1. Mean accuracy (%) over the winner cells, by model and vote variant (the vote is causal and restarts at each segment onset; k = number of last decisions)\n")
    for name, cs in scopes:
        L.append(f"**{name}**\n")
        L.append("| Model | " + " | ".join(VN[v] for v in A.VARIANTS) + " |\n|---|" + "---|" * len(A.VARIANTS))
        for mo in A.CLASSICAL:
            L.append(f"| {mo} | " + " | ".join(f"{100 * np.mean([R[c][f'{mo}/{v}']['acc'] for c in cs]):.2f}" for v in A.VARIANTS) + " |")
        L.append("| best of three, per cell | " + " | ".join(f"{100 * np.mean([max(R[c][f'{mo}/{v}']['acc'] for mo in A.CLASSICAL) for c in cs]):.2f}" for v in A.VARIANTS) + " |")
        L.append("\nLSTM (no vote, it has its own memory): " + ", ".join(f"{m} {100 * np.mean([R[c][m]['acc'] for c in cs]):.2f}" for m in A.LSTMS) + "\n")

    L.append("### F2. Paired differences in points (mean over cells; in brackets: cells where the first model is ahead / behind by the sign of the mean fold difference; cells resolved better / worse by the 95% interval)\n")
    L.append("| Comparison | all 28 cells | set A (14) | set W (14) | Reading (all 28 cells, declared rule) |\n|---|---|---|---|---|")
    rows = []
    for lstm in ("LSTM_tuned", "LSTM_default"):
        rows.append((f"{lstm} minus best classical, no vote (the first run's comparison)", lstm, lambda c, R_: A.best_fold(R_, c, "k1")))
        rows.append((f"{lstm} minus best classical, nested vote (H1)", lstm, lambda c, R_: A.best_fold(R_, c, "nested")))
        for mo in A.CLASSICAL:
            rows.append((f"{lstm} minus {mo}, vote over all windows since the onset (H2/H3)", lstm, f"{mo}/all"))
        rows.append((f"{lstm} minus best of three, vote over all windows (H2/H3)", lstm, lambda c, R_: A.best_fold(R_, c, "all")))
    for label, ma, mb in rows:
        cellsx = []
        for name, cs in scopes:
            mean, n, rp, rn = A.compare(cs, R, ma, mb)
            ah, be = A.signs(cs, R, ma, mb)
            cellsx.append(f"{mean:+.2f} ({ah}/{be}; {rp}/{rn})")
        mean, n, rp, rn = A.compare(cells, R, ma, mb)
        L.append(f"| {label} | " + " | ".join(cellsx) + f" | {A.reading(mean, n, rp, rn, 'LSTM better', 'classical better')} |")
    L.append("")
    L.append("Effect of the vote itself (variant minus no vote, same model):\n")
    L.append("| Model | variant | all 28 cells | set A (14) | set W (14) |\n|---|---|---|---|---|")
    for mo in A.CLASSICAL:
        for v in ("k5", "k15", "all", "nested"):
            cellsx = []
            for name, cs in scopes:
                mean, n, rp, rn = A.compare(cs, R, f"{mo}/{v}", f"{mo}/k1")
                cellsx.append(f"{mean:+.2f} ({rp}/{rn})")
            L.append(f"| {mo} | {VN[v]} | " + " | ".join(cellsx) + " |")
    L.append("")

    L.append("### F3. Per patient (mean over that patient's cells, sets A and W together): accuracy % / weighted F1 % / kappa\n")
    cols = [("LDA/k1", "LDA, no vote"), ("SVM/k1", "SVM, no vote"), ("RandomForest/k1", "RF, no vote"), ("LDA/nested", "LDA, nested vote"), ("SVM/nested", "SVM, nested vote"), ("RandomForest/nested", "RF, nested vote"),
            ("LSTM_default", "LSTM default"), ("LSTM_tuned", "LSTM tuned")]
    L.append("| Patient (cells) | " + " | ".join(c[1] for c in cols) + " |\n|---|" + "---|" * len(cols))
    for part in A.SHORT:
        cs = [c for c in cells if c[0] == part]
        L.append(f"| {A.SHORT[part]} ({len(cs)}) | " + " | ".join(f"{100 * np.mean([R[c][m]['acc'] for c in cs]):.1f} / {100 * np.mean([R[c][m]['f1'] for c in cs]):.1f} / {np.mean([R[c][m]['kappa'] for c in cs]):.3f}" for m, _ in cols) + " |")
    L.append("")
    L.append("LSTM tuned minus the best classical model of each cell, by patient (points; cells where the LSTM is ahead / behind):\n")
    L.append("| Patient | no vote | nested vote | vote over all windows |\n|---|---|---|---|")
    for part in A.SHORT:
        cs = [c for c in cells if c[0] == part]
        out = []
        for v in ("k1", "nested", "all"):
            f = lambda c, R_, v=v: A.best_fold(R_, c, v)
            mean, n, rp, rn = A.compare(cs, R, "LSTM_tuned", f)
            ah, be = A.signs(cs, R, "LSTM_tuned", f)
            out.append(f"{mean:+.2f} ({ah}/{be})")
        L.append(f"| {A.SHORT[part]} | " + " | ".join(out) + " |")
    L.append("")

    L.append("### F4. Time-resolved accuracy (%): pooled per cell by window index since the onset, then the mean over the 28 cells (window i is decided 200 + 50 (i-1) ms after the onset)\n")
    tot = np.sum([R[c]["LDA/k1"]["bin_total"] for c in cells], axis=0)
    ms = [("LDA/k1", "LDA, no vote"), ("LDA/all", "LDA, vote over all"), ("SVM/k1", "SVM, no vote"), ("SVM/all", "SVM, vote over all"), ("RandomForest/k1", "RF, no vote"), ("RandomForest/all", "RF, vote over all"),
          ("LSTM_default", "LSTM default"), ("LSTM_tuned", "LSTM tuned")]
    L.append("| Window index since onset | " + " | ".join(A.BIN_NAMES) + " |\n|---|" + "---|" * len(A.BIN_NAMES))
    L.append("| Time of decision after onset (ms) | 200-250 | 300-400 | 450-650 | 700-1150 | 1200+ |")
    L.append("| Share of all windows | " + " | ".join(f"{100 * x / tot.sum():.1f}%" for x in tot) + " |")
    for m, lab in ms:
        row = []
        for b in range(len(A.BINS)):
            vals = [R[c][m]["bin_correct"][b] / R[c][m]["bin_total"][b] for c in cells if R[c][m]["bin_total"][b] > 0]
            row.append(f"{100 * np.mean(vals):.1f}")
        L.append(f"| {lab} | " + " | ".join(row) + " |")
    L.append("")

    L.append("### F5. Every winner cell: accuracy % (weighted F1 %, kappa) without vote, with the nested vote, and the LSTM\n")
    L.append("| Patient | Set | Configuration | kb/s | LDA no vote | SVM no vote | RF no vote | Best classical, nested vote | LSTM default | LSTM tuned | LSTM tuned minus best nested vote |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        part, fset, cfg = c
        best = max(A.CLASSICAL, key=lambda m: R[c][f"{m}/nested"]["acc"])
        rb = R[c][f"{best}/nested"]
        def f(r):
            return f"{100 * r['acc']:.1f} ({100 * r['f1']:.1f}, {r['kappa']:.3f})"
        d = R[c]["LSTM_tuned"]["fold"] - rb["fold"]
        L.append(f"| {A.SHORT[part]} | {fset} | {A.fmt(cfg)} | {cfg[0] * cfg[1] * cfg[2] / 1000:.0f} | " + " | ".join(f(R[c][f'{m}/k1']) for m in A.CLASSICAL) + f" | {best.replace('RandomForest', 'RF')}: {f(rb)} | {f(R[c]['LSTM_default'])} | {f(R[c]['LSTM_tuned'])} | {100 * d.mean():+.1f} |")
    L.append("")

    L.append("### F6. Nested choice of the vote span (counts over cells x outer folds)\n")
    L.append("| Model | k=1 | k=3 | k=5 | k=9 | k=15 | all since onset |\n|---|---|---|---|---|---|---|")
    for mo in A.CLASSICAL:
        cnt = collections.Counter(("all" if k == 0 else k) for c in cells for k in chosen[c + (mo,)])
        L.append(f"| {mo} | " + " | ".join(str(cnt[k]) for k in [1, 3, 5, 9, 15, "all"]) + " |")
    L.append("")

    L.append("### F7. Cost per decision (analytic counts, mean over the cells; total = extraction + 2 x model MACs + comparisons; the vote adds C + 1 integer operations, C = number of classes)\n")
    L.append("| Set | Extraction kFLOP | LDA total | SVM total | RF total | LSTM total | Vote (integer ops) |\n|---|---|---|---|---|---|---|")
    for name, cs in scopes[1:]:
        ex, tm, vote = [], collections.defaultdict(list), []
        for c in cs:
            part, fset, cfg = c
            key = lambda mo: f"{part}|{cfg[0]}Hz|{cfg[1]}ch|{cfg[2]}bit|{fset}|{mo}"
            for mo in A.CLASSICAL:
                tm[mo].append(costs(A.G[key(mo)], mo)[3])
            tm["LSTM"].append(costs(A.N[key("LSTM_tuned")], "LSTM_tuned")[3])
            ex.append(A.G[key("LDA")]["extraction"]["flops"])
            vote.append(len(A.G[key("LDA")]["class_labels"]) + 1)
        L.append(f"| {name} | {np.mean(ex) / 1000:.1f} | " + " | ".join(f"{np.mean(tm[m]) / 1000:.1f}" for m in ("LDA", "SVM", "RandomForest", "LSTM")) + f" | {np.mean(vote):.0f} |")
    L.append("")
    OUT.write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
