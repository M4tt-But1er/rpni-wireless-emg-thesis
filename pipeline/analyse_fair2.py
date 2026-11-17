"""Analysis of the fair-comparison follow-up (docs/fair_followup_declared_protocol.md): applies every declared pipeline to the stored classical outputs (cache/grid/fair2/classical/), reads the new LSTM
runs (cache/grid/fair2/lstm/, if present) and the earlier LSTM runs (cache/grid/fair/lstm/), and writes the declared comparisons H4-H9 (paired by fold, 4 df, 95% t-interval, reading rule of the protocol),
accuracy tables, chosen-candidate counts, per-patient and time-resolved views and analytic costs.
Outputs: cache/grid/fair2/fair2_tables.md (all tables), fair2_results.json, fair2_cells.csv. Usage: python -W ignore analyse_fair2.py
Everything is computed from stored predictions; nothing is refitted. If the new LSTM files are missing the LSTM sections are skipped."""
import collections
import json
import warnings

import numpy as np
import pandas as pd

import lstm_tuning as T
from analyse_fair import BIN_NAMES, BINS, CLASSICAL, G, SHORT, WIN, compare, npz as npz_old, record, reading, signs
from config import CACHE_DIR
from fair_history import CAND_NAMES, CANDIDATES, HARD_IDX, SOFT_IDX, STACK_VARIANTS, apply_candidate, soft_vote
from fair_vote import causal_vote, window_index
from feature_cost import model_cost
from grid_tables import table_path
from run_lstm_final import task_path as lstm_task_path

warnings.filterwarnings("ignore")
FAIR2 = CACHE_DIR / "grid" / "fair2"
MULT = {"none": 1, "ST3": 3, "ST5": 5, "CUM": 2}
SM = {"LDA": "LDA", "SVM": "SVM", "RandomForest": "RF"}
ROUND_OF = {"LSTM_r1": 1, "LSTM_best": 2}            # round 1 = the declared search, round 2 = after the edge extension of amendment A2 (primary)
LSTM_NEW = [f"{b}{s}" for b in ROUND_OF for s in ("", "_ens5", "+hard_all", "+mean_all", "_ens5+hard_all", "_ens5+mean_all")]
LINES = []


def p(s=""):
    LINES.append(s)
    print(s)


def cells_list():
    return [(pf.split("|")[0], pf.split("|")[1], tuple(c)) for pf, cfgs in WIN.items() for c in cfgs]


def key_of(part, fset, cfg, tail):
    return f"{part}|{cfg[0]}Hz|{cfg[1]}ch|{cfg[2]}bit|{fset}|{tail}"


def round2_reuses_round1():
    """Amendment A2: if no new configuration reached the round-2 top 6, the round-1 files are the round-2 result."""
    p2 = FAIR2 / "lstm_candidates_r2.json"
    if not p2.exists():
        return False
    return {c["name"] for c in json.load(open(FAIR2 / "lstm_candidates.json"))["final"]} == {c["name"] for c in json.load(open(p2))["final"]}


def load_lstm_new(part, fset, cfg, n, C, rnd):
    """Assemble the 5 outer-fold task files of one cell and round into arrays in table-row order; None if any is missing."""
    files = [lstm_task_path(part, cfg[0], cfg[1], cfg[2], fset, f, 1 if (rnd == 1 or round2_reuses_round1()) else 2) for f in range(5)]
    if not all(f.exists() for f in files):
        return None
    out = dict(single=np.full(n, -1), ens=np.full(n, -1), P1=np.zeros((n, C)), Pe=np.zeros((n, C)), chosen=[], epochs=[], cfgs=[], seconds=0.0, features=0)
    for f in files:
        z = np.load(f, allow_pickle=False)
        r = z["rows"]
        out["single"][r], out["ens"][r] = z["pred_single"], z["pred_ens"]
        out["P1"][r], out["Pe"][r] = z["proba_single"], z["proba_ens"]
        out["chosen"].append(str(z["chosen"])); out["epochs"].append(int(z["epochs"])); out["cfgs"].append(json.loads(str(z["chosen_cfg"]))); out["seconds"] += float(z["seconds"])
        out["features"] = int(z["n_features"])
    assert (out["single"] >= 0).all() and (out["ens"] >= 0).all(), "LSTM rows not covered"
    return out


def build():
    cells = cells_list()
    R, choices, fixed, info, lstm_meta = {}, {}, {}, {}, {}
    meta = {}
    for part, fset, cfg in cells:
        rate, nch, bits = cfg
        if (part, rate, bits) not in meta:
            meta[(part, rate, bits)] = pd.read_parquet(table_path(part, rate, bits), columns=["cue", "segment_id", "window_time_s"])
        t = meta[(part, rate, bits)]
        y, seg = t["cue"].to_numpy(), t["segment_id"].to_numpy()
        widx = window_index(seg)
        c = (part, fset, cfg)
        R[c] = {}
        labels = None
        for model in CLASSICAL:
            z = np.load(FAIR2 / "classical" / (key_of(part, fset, cfg, model).replace("|", "__") + ".npz"), allow_pickle=False)
            labels = [int(l) for l in z["labels"]]
            lab, folds, C = np.array(labels), z["folds"], len(labels)
            c2i = {l: i for i, l in enumerate(labels)}
            yi = np.array([c2i[v] for v in y])
            fl = sorted(np.unique(folds))
            PRED = {v: [apply_candidate(cd, z[f"hard__{v}"], z[f"proba__{v}"], seg, C) for cd in CANDIDATES] for v in STACK_VARIANTS}
            inner = {v: z[f"inner__{v}"] for v in STACK_VARIANTS}
            fixed[(c, model)] = {v: [float((PRED[v][i] == yi).mean()) for i in range(len(CANDIDATES))] for v in STACK_VARIANTS}
            info[(c, model)] = {v: json.loads(str(z[f"info__{v}"])) for v in STACK_VARIANTS}

            def nested(opts):
                pred, ch = np.empty(len(y), dtype=int), []
                for fi, f in enumerate(fl):
                    j = int(np.argmax([inner[v][fi][i] for v, i in opts]))          # first maximum = earlier candidate (declared tie rule)
                    v, i = opts[j]
                    te = folds == f
                    pred[te] = PRED[v][i][te]
                    ch.append((v, CAND_NAMES[i]))
                return pred, ch
            pipes = {"P0": (PRED["none"][0], None)}
            pipes["P1"] = nested([("none", i) for i in HARD_IDX])
            pipes["P2"] = nested([("none", i) for i in SOFT_IDX])
            pipes["P3"] = nested([("none", i) for i in range(len(CANDIDATES))])
            for v in STACK_VARIANTS[1:]:
                pipes[f"stack:{v}"] = (PRED[v][0], None)
                pipes[f"stackvote:{v}"] = nested([(v, i) for i in range(len(CANDIDATES))])
            pipes["P6"] = nested([(v, i) for v in STACK_VARIANTS for i in range(len(CANDIDATES))])
            for name, (pred, ch) in pipes.items():
                R[c][f"{model}/{name}"] = record(y, lab[pred], labels, folds, widx)
                if ch is not None:
                    choices[(c, model, name)] = ch
        for method in ("LSTM_default", "LSTM_tuned"):
            z = npz_old("lstm", key_of(part, fset, cfg, method))
            R[c][method] = record(y, z["pred"], labels, folds, widx)
        lab = np.array(labels)
        for base, rnd in ROUND_OF.items():
            new = load_lstm_new(part, fset, cfg, len(y), len(labels), rnd)
            if new is None:
                continue
            R[c][base] = record(y, lab[new["single"]], labels, folds, widx)
            R[c][f"{base}_ens5"] = record(y, lab[new["ens"]], labels, folds, widx)
            for tag, pr, Pm in ((base, new["single"], new["P1"]), (f"{base}_ens5", new["ens"], new["Pe"])):
                R[c][f"{tag}+hard_all"] = record(y, lab[causal_vote(pr, seg, None, len(labels))], labels, folds, widx)
                R[c][f"{tag}+mean_all"] = record(y, lab[soft_vote(Pm, seg, "mean", None)], labels, folds, widx)
            lstm_meta[(c, base)] = new
    return cells, R, choices, fixed, info, lstm_meta


def mean_acc(R, cs, m):
    return 100 * np.mean([R[c][m]["acc"] for c in cs])


def best_fold(R, c, pipe):
    best = max(CLASSICAL, key=lambda mo: R[c][f"{mo}/{pipe}"]["acc"])
    return R[c][f"{best}/{pipe}"]["fold"]


def cmp_row(cs, R, ma, mb, pos, neg):
    mean, n, rp, rn = compare(cs, R, ma, mb)
    ahead, behind = signs(cs, R, ma, mb)
    return f"{mean:+.2f}", f"{ahead}/{behind}", f"{rp}/{rn}", reading(mean, n, rp, rn, pos, neg)


def table(header, rows):
    p("| " + " | ".join(header) + " |")
    p("|" + "---|" * len(header))
    for r in rows:
        p("| " + " | ".join(str(x) for x in r) + " |")
    p()


def main():
    cells, R, choices, fixed, info, lstm_meta = build()
    bases = [b for b in ROUND_OF if all((c, b) in lstm_meta for c in cells)]                 # LSTM rounds whose 140 task files are all present
    have_lstm = bool(bases)
    lstm_methods = [m for m in LSTM_NEW if m.split("_ens5")[0].split("+")[0] in bases]
    lstm_curves = [x for b in bases for x in (b, b + "_ens5")]
    scopes = [("all 28 cells", cells), ("set A (14)", [c for c in cells if c[1] == "A"]), ("set W (14)", [c for c in cells if c[1] == "W"]),
              ("20 non-pilot cells", [c for c in cells if not (tuple(c[2]) == (1000, 8, 16))])]
    pipe_names = ["P0", "P1", "P2", "P3"] + [f"stack:{v}" for v in STACK_VARIANTS[1:]] + [f"stackvote:{v}" for v in STACK_VARIANTS[1:]] + ["P6"]
    pdesc = {"P0": "raw per-window (grid)", "P1": "hard vote, nested k (earlier headline)", "P2": "soft/EMA vote, nested", "P3": "hard + soft vote, nested (22 candidates)", "stack:ST3": "stack ST3, no vote",
             "stack:ST5": "stack ST5, no vote", "stack:CUM": "stack CUM, no vote", "stackvote:ST3": "ST3 + vote nested (22)", "stackvote:ST5": "ST5 + vote nested (22)", "stackvote:CUM": "CUM + vote nested (22)",
             "P6": "full nested (4 variants x 22)"}
    p("# Fair comparison follow-up: generated tables (analyse_fair2.py; every number is computed from stored predictions)")
    p()
    p("## F8. Part A: mean window accuracy (%) over the cells, by pipeline and model (P0..P6 as in the protocol)")
    p()
    for name, cs in scopes:
        p(f"**{name}**")
        p()
        rows = [[pn, pdesc[pn]] + [f"{mean_acc(R, cs, f'{mo}/{pn}'):.2f}" for mo in CLASSICAL] + [f"{100 * np.mean([max(R[c][f'{mo}/{pn}']['acc'] for mo in CLASSICAL) for c in cs]):.2f}"] for pn in pipe_names]
        lrows = [["", "LSTM_default (earlier)", "", "", "", f"{mean_acc(R, cs, 'LSTM_default'):.2f}"], ["", "LSTM_tuned (earlier)", "", "", "", f"{mean_acc(R, cs, 'LSTM_tuned'):.2f}"]]
        if have_lstm:
            lrows += [["", m, "", "", "", f"{mean_acc(R, cs, m):.2f}"] for m in lstm_methods]
        table(["pipeline", "description", "LDA", "SVM", "RF", "best of three (post hoc) / LSTM"], rows + lrows)

    p("## F9. Declared comparisons H4-H6 (classical pipelines; paired by fold; mean difference in points over cells; ahead/behind = cells where the first is ahead/behind by the sign of the mean fold difference; res = cells resolved better/worse by the 95% interval)")
    p()
    comps = [("H4", "P3 minus P1 (does a soft/EMA candidate add to the hard vote?)", lambda mo: (f"{mo}/P3", f"{mo}/P1")),
             ("H5a", "stack ST3, no vote, minus P0", lambda mo: (f"{mo}/stack:ST3", f"{mo}/P0")), ("H5a", "stack ST5, no vote, minus P0", lambda mo: (f"{mo}/stack:ST5", f"{mo}/P0")),
             ("H5a", "stack CUM, no vote, minus P0", lambda mo: (f"{mo}/stack:CUM", f"{mo}/P0")),
             ("H5b", "ST3 + vote minus P3", lambda mo: (f"{mo}/stackvote:ST3", f"{mo}/P3")), ("H5b", "ST5 + vote minus P3", lambda mo: (f"{mo}/stackvote:ST5", f"{mo}/P3")),
             ("H5b", "CUM + vote minus P3", lambda mo: (f"{mo}/stackvote:CUM", f"{mo}/P3")), ("H6", "P6 minus P1", lambda mo: (f"{mo}/P6", f"{mo}/P1"))]
    for name, cs in [scopes[0], scopes[1], scopes[2]]:
        p(f"**{name}**")
        p()
        rows = []
        for h, desc, f in comps:
            for mo in CLASSICAL:
                a, b = f(mo)
                rows.append([h, desc, SM[mo]] + list(cmp_row(cs, R, a, b, "first better", "second better")))
        table(["id", "comparison", "model", "mean diff (pts)", "ahead/behind", "res better/worse", "reading"], rows)

    p("## F10. Part A: mean accuracy (%) of every FIXED candidate (no choice, applied to the stored per-window outputs), all 28 cells, by model and stack variant")
    p()
    hdr = ["candidate"] + [f"{SM[mo]} {v}" for mo in CLASSICAL for v in STACK_VARIANTS]
    rows = [[CAND_NAMES[i]] + [f"{100 * np.mean([fixed[(c, mo)][v][i] for c in cells]):.2f}" for mo in CLASSICAL for v in STACK_VARIANTS] for i in range(len(CANDIDATES))]
    table(hdr, rows)
    p("(hard_k1 of `none` is the raw grid result; for the SVM the soft candidates use Platt probabilities, so mean_k1 differs from hard_k1.)")
    p()

    p("## F11. Which candidate the nested choice picked (counts over 28 cells x 5 outer folds = 140 choices per model)")
    p()
    for pn in ("P3", "P6"):
        p(f"**{pn}** ({pdesc[pn]})")
        p()
        rows = []
        for mo in CLASSICAL:
            ch = [x for c in cells for x in choices[(c, mo, pn)]]
            fam = collections.Counter(n.split("_")[0] for _, n in ch)
            spans = collections.Counter(n for _, n in ch)
            top = ", ".join(f"{n} {k}" for n, k in spans.most_common(5))
            row = [SM[mo], ", ".join(f"{f} {fam.get(f, 0)}" for f in ("hard", "mean", "log", "ema"))]
            if pn == "P6":
                vc = collections.Counter(v for v, _ in ch)
                row.append(", ".join(f"{v} {vc.get(v, 0)}" for v in STACK_VARIANTS))
            row.append(top)
            rows.append(row)
        table(["model", "family counts"] + (["variant counts"] if pn == "P6" else []) + ["top 5 candidates"], rows)

    p("## F12. Per patient: mean accuracy (%) over that patient's cells (sets A and W together)")
    p()
    cols = [f"{mo}/{pn}" for mo in CLASSICAL for pn in ("P0", "P1", "P3", "P6")] + ["LSTM_tuned"] + lstm_curves
    rows = [[f"{SHORT[pt]} (n={len([c for c in cells if c[0] == pt])})"] + [f"{mean_acc(R, [c for c in cells if c[0] == pt], m):.2f}" for m in cols] for pt in SHORT]
    table(["patient"] + [m.replace("RandomForest", "RF") for m in cols], rows)

    p("## F13. Time-resolved accuracy (%): pooled per cell by window index since the onset, then the mean over the 28 cells (window i is decided 200 + 50 (i-1) ms after the onset)")
    p()
    tm = [f"{mo}/{pn}" for mo in CLASSICAL for pn in ("P0", "P3", "stack:CUM", "P6")] + ["LSTM_tuned"] + lstm_curves
    tot = np.sum([R[c]["LDA/P0"]["bin_total"] for c in cells], axis=0)
    rows = [["share of windows"] + [f"{100 * x / tot.sum():.1f}%" for x in tot]]
    for m in tm:
        rows.append([m.replace("RandomForest", "RF")] + [f"{100 * np.mean([R[c][m]['bin_correct'][b] / R[c][m]['bin_total'][b] for c in cells if R[c][m]['bin_total'][b] > 0]):.1f}" for b in range(len(BINS))])
    table(["method"] + BIN_NAMES, rows)

    p("## F14. Analytic cost per decision (kFLOP = extraction + 2 x model MACs + comparisons; mean over the 28 cells; stacked models keep the earlier windows' feature vectors in memory, no recomputation; RF comparisons are not recomputed for stacked forests (depth not stored))")
    p()
    crow = []
    for mo in CLASSICAL:
        for v in STACK_VARIANTS:
            vals = []
            for c in cells:
                part, fset, cfg = c
                ex = G[key_of(part, fset, cfg, "LDA")]["extraction"]["flops"]
                Cn = len(G[key_of(part, fset, cfg, "LDA")]["class_labels"])
                F = int(np.load(FAIR2 / "classical" / (key_of(part, fset, cfg, mo).replace("|", "__") + ".npz"))["n_features"]) * MULT[v]
                if mo == "LDA":
                    mc = model_cost("LDA", F, Cn)
                elif mo == "SVM":
                    mc = model_cost("SVM", F, Cn, n_sv=float(np.mean([i["n_sv"] for i in info[(c, mo)][v]])))
                else:
                    r = G[key_of(part, fset, cfg, "RandomForest")]
                    fi = r["fold_info"]
                    mc = model_cost("RandomForest", F, Cn, trees=int(fi[0]["trees"]), max_depth=int(max(i["max_depth"] for i in fi)))
                vals.append(ex + 2 * mc["macs"] + mc["other_ops"])
            crow.append([SM[mo], v, f"{np.mean(vals) / 1000:.1f}"])
    table(["model", "variant", "kFLOP per decision"], crow)
    for base in bases:
        single, ens = [], []
        for c in cells:
            part, fset, cfg = c
            ex = G[key_of(part, fset, cfg, "LDA")]["extraction"]["flops"]
            Cn = len(G[key_of(part, fset, cfg, "LDA")]["class_labels"])
            m = lstm_meta[(c, base)]
            macs = float(np.mean([T.macs_per_step(m["features"], cf, Cn) + m["features"] for cf in m["cfgs"]]))      # + the input standardisation, as in model_cost
            single.append(ex + 2 * macs)
            ens.append(ex + 2 * macs * T.N_SEEDS)                                                                   # extraction is shared, the model part runs 5 times
        p(f"{base}: mean {np.mean(single) / 1000:.1f} kFLOP per decision (chosen configurations averaged over folds); {base}_ens5: {np.mean(ens) / 1000:.1f}.")
        p()
    if bases:
        p("For comparison the earlier LSTM_tuned (64 units, 1 layer) cost about 196 (set A) / 213 (set W) kFLOP.")
        p()

    if have_lstm:
        p("## F15. Part B: chosen LSTM configurations (counts over 28 cells x 5 outer folds) and epochs")
        p()
        for base in bases:
            ms = [lstm_meta[(c, base)] for c in cells]
            names = collections.Counter(n for m in ms for n in m["chosen"])
            ep = np.array([e for m in ms for e in m["epochs"]])
            p(f"**{base}** (round {ROUND_OF[base]}{', identical to round 1' if ROUND_OF[base] == 2 and round2_reuses_round1() else ''}). Chosen: " + ", ".join(f"{k} {v}" for k, v in names.most_common()))
            p()
            p(f"Epochs of the retrain: median {np.median(ep):.0f}, 5th-95th percentile {np.percentile(ep, 5):.0f}-{np.percentile(ep, 95):.0f}, max {ep.max()}, share at the 400 limit {100 * np.mean(ep >= T.MAX_EPOCHS):.1f}%.")
            p()
        p("## F16. Declared comparisons H7-H9 (LSTM; paired by fold; mean difference in points; LSTM ahead/behind = cells where the LSTM is ahead/behind; res = resolved better/worse). Primary LSTM = LSTM_best (round 2, amendment A2); LSTM_r1 (round 1) is the same comparison, descriptive")
        p()
        for name, cs in scopes:
            p(f"**{name}**")
            p()
            rows = []
            for base in bases:
                for lstm, h in ((base, "H7/H8"), (f"{base}_ens5", "H9")):
                    if lstm == base:
                        rows.append(["H7", f"{lstm} minus LSTM_tuned (earlier)"] + list(cmp_row(cs, R, lstm, "LSTM_tuned", "LSTM better", "earlier LSTM better")))
                    for pn in ("P3", "P6"):
                        for mo in CLASSICAL:
                            rows.append([h, f"{lstm} minus {SM[mo]}/{pn}"] + list(cmp_row(cs, R, lstm, f"{mo}/{pn}", "LSTM better", "classical better")))
                    for pn in ("P1", "P3", "P6"):
                        rows.append([h, f"{lstm} minus best of three/{pn} (post hoc)"] + list(cmp_row(cs, R, lstm, lambda c, R_, pn=pn: best_fold(R_, c, pn), "LSTM better", "classical better")))
            table(["id", "comparison", "mean diff (pts)", "LSTM ahead/behind", "res better/worse", "reading"], rows)
        p("## F17. Part B secondary (parameter-free): the LSTM decisions smoothed by a causal vote over everything since the onset, against the classical pipeline P6 (best of three, post hoc)")
        p()
        rows = []
        for m in lstm_methods:
            rows.append([m] + [f"{mean_acc(R, cs, m):.2f}" for _, cs in scopes] + [cmp_row(scopes[0][1], R, m, lambda c, R_: best_fold(R_, c, "P6"), "LSTM better", "classical better")[0]])
        table(["method"] + [n for n, _ in scopes] + ["minus best-of-three P6 (all 28)"], rows)
        p("## F18. Per patient: LSTM minus the best of three classical models, by pipeline (mean over that patient's cells; cells of one patient are not independent)")
        p()
        rows = []
        for lstm in [x for b in bases for x in (b, b + "_ens5")]:
            for pn in ("P1", "P3", "P6"):
                parts = []
                for pt in SHORT:
                    cs = [c for c in cells if c[0] == pt]
                    mean, n, rp, rn = compare(cs, R, lstm, lambda c, R_, pn=pn: best_fold(R_, c, pn))
                    a, b = signs(cs, R, lstm, lambda c, R_, pn=pn: best_fold(R_, c, pn))
                    parts.append(f"{mean:+.2f} ({a}/{b} of {n})")
                rows.append([lstm, pn] + parts)
        table(["LSTM", "classical pipeline"] + [SHORT[pt] for pt in SHORT], rows)

    # files
    out = {}
    for c in cells:
        part, fset, cfg = c
        for m, r in R[c].items():
            out[key_of(part, fset, cfg, m)] = dict(accuracy=r["acc"], weighted_f1=r["f1"], cohen_kappa=r["kappa"], confusion_matrix=r["cm"], fold_accs=r["fold"].tolist(), bin_correct=r["bin_correct"], bin_total=r["bin_total"])
    json.dump(dict(results=out, choices={f"{key_of(c[0], c[1], c[2], mo)}|{pn}": v for (c, mo, pn), v in choices.items()}), open(FAIR2 / "fair2_results.json", "w"))
    rows = [dict(patient=SHORT[c[0]], set=c[1], rate_hz=c[2][0], channels=c[2][1], bits=c[2][2], pilot=int(tuple(c[2]) == (1000, 8, 16)), method=m, accuracy=r["acc"], weighted_f1=r["f1"], kappa=r["kappa"]) for c in cells for m, r in R[c].items()]
    pd.DataFrame(rows).to_csv(FAIR2 / "fair2_cells.csv", index=False)
    open(FAIR2 / "fair2_tables.md", "w", encoding="utf-8").write("\n".join(LINES) + "\n")
    print("\nFiles: cache/grid/fair2/fair2_tables.md, fair2_results.json, fair2_cells.csv")


if __name__ == "__main__":
    main()
