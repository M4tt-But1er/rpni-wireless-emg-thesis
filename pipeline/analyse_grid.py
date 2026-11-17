"""Analysis of the bitrate grid exactly as declared in docs/grid_declared_protocol.md. Redirect stdout to cache/grid/grid_summary.txt. Also writes cache/grid/grid_cells.csv (one row per cell: metrics, cost counts)
and cache/grid/winners.json (the step-3 winners, rule fixed in the protocol).
Usage: python -W ignore analyse_grid.py > cache/grid/grid_summary.txt"""
import collections
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from bitrate_pareto import best_ratio_point, pareto_frontier
from build_wide_tables import PARTS
from config import CACHE_DIR
from feature_cost import DECISIONS_PER_SECOND, model_cost
from check_elbow_truncation import best_ratio_point_geometric

R = json.load(open(CACHE_DIR / "grid" / "grid_results.json"))
SHORT = {"P1_combined": "P1", "P2_combined": "P2", "Wallace_S1": "S1", "Wallace_S2": "S2"}
MODELS = ["LDA", "SVM", "RandomForest"]
SETS = ["A", "W"]
TCRIT = student_t.ppf(0.975, 4)
REF = (1000, 8, 16)


def parse(key):
    part, rate, nch, bits, fset, model = key.split("|")
    return part, int(rate[:-2]), int(nch[:-2]), int(bits[:-3]), fset, model


def costs(rec, model, rate):
    n_classes = len(rec["class_labels"])
    info = rec["fold_info"]
    kw = {}
    if model == "SVM":
        kw["n_sv"] = float(np.mean([i["n_sv"] for i in info]))
    elif model == "RandomForest":
        kw.update(trees=int(info[0]["trees"]), max_depth=int(max(i["max_depth"] for i in info)))
    mc = model_cost(model, rec["n_features"], n_classes, **kw)
    ex = rec["extraction"]
    total = ex["flops"] + 2 * mc["macs"] + mc["other_ops"]
    return dict(extraction_flops=ex["flops"], extraction_special=ex["special_ops"], model_macs=mc["macs"], model_other_ops=mc["other_ops"], model_special=mc["special_ops"],
                model_ops=mc["macs"] + mc["other_ops"],
                total_flops=total, total_flops_per_s=total * DECISIONS_PER_SECOND)


def table():
    rows = []
    for key, rec in R.items():
        part, rate, nch, bits, fset, model = parse(key)
        fa = np.array(rec["fold_accs"])
        rows.append(dict(patient=SHORT[part], part=part, rate_hz=rate, channels=nch, bits=bits, bitrate_bps=rate * nch * bits, feature_set=fset, model=model,
                         accuracy=rec["accuracy"], weighted_f1=rec["weighted_f1"], kappa=rec["cohen_kappa"], fold_acc_sd=float(fa.std(ddof=1)),
                         n_features=rec["n_features"], n_windows=rec["n_windows"], **costs(rec, model, rate)))
    return pd.DataFrame(rows)


def pick(df, metric, cost_col, geometric=True):
    """(config tuple, cost, metric) at the elbow of the Pareto frontier of `metric` against `cost_col` (log10 axis). geometric=True: the declared max-distance-from-chord rule without truncation (primary);
    geometric=False: bitrate_pareto.best_ratio_point as coded in the earlier grid (elbow truncated to an integer on the log axis), for comparability (Amendment 2 of the protocol)."""
    df = df.sort_values([cost_col, "rate_hz", "channels", "bits"])
    pts = list(zip(df[cost_col].astype(float), df[metric].astype(float)))
    c, m = (best_ratio_point_geometric if geometric else best_ratio_point)(pts)
    row = df[(df[cost_col] == c) & (df[metric] == m)].iloc[0]
    return (int(row.rate_hz), int(row.channels), int(row.bits)), float(c), float(m)


def cheapest_reaching(df, target, cost_col="bitrate_bps"):
    ok = df[df["accuracy"] >= target].sort_values([cost_col, "rate_hz", "channels", "bits"])
    if ok.empty:
        return None
    r = ok.iloc[0]
    return (int(r.rate_hz), int(r.channels), int(r.bits)), float(r[cost_col]), float(r.accuracy)


def fmt(cfg):
    return f"{cfg[0]} Hz x {cfg[1]} ch x {cfg[2]} b"


def main():
    df = table()
    df.to_csv(CACHE_DIR / "grid" / "grid_cells.csv", index=False)
    n_expected = 193 * 2 * 3
    print(f"Cells present: {len(df)} of {n_expected}  (written cache/grid/grid_cells.csv)\n")
    if df.empty:
        return

    print("### 1. Pareto frontier elbows (log10 axis) and cheapest configuration reaching an accuracy target, per patient / feature set / model")
    print("Geometric elbow (primary) of accuracy, weighted F1 and kappa against bit rate, then the AS-CODED elbow of the earlier grid; cheapest bit rate with accuracy >= 80/90/95%; then the geometric elbow of accuracy against total FLOPs per decision and against model operations (MACs + comparisons).")
    winners = {}
    for p in PARTS:
        for fset in SETS:
            for model in MODELS:
                d = df[(df.part == p) & (df.feature_set == fset) & (df.model == model)]
                if d.empty:
                    continue
                parts_ = []
                for metric in ("accuracy", "weighted_f1", "kappa"):
                    cfg, c, m = pick(d, metric, "bitrate_bps")
                    parts_.append(f"{metric} elbow {fmt(cfg)} = {c / 1000:.0f} kb/s ({100 * m:.1f}{'' if metric == 'kappa' else '%'})" if metric != "kappa" else f"kappa elbow {fmt(cfg)} = {c / 1000:.0f} kb/s ({m:.3f})")
                    if metric in ("accuracy", "weighted_f1"):
                        winners.setdefault(f"{p}|{fset}", set()).add(cfg)
                cfg_old, c_old, m_old = pick(d, "accuracy", "bitrate_bps", geometric=False)
                parts_.append(f"AS-CODED (truncated, as in the earlier grid) accuracy elbow {fmt(cfg_old)} = {c_old / 1000:.0f} kb/s ({100 * m_old:.1f}%)")
                winners.setdefault(f"{p}|{fset}", set()).add(cfg_old)
                tgt = []
                for target in (0.80, 0.90, 0.95):
                    r = cheapest_reaching(d, target)
                    tgt.append(f">={int(target * 100)}%: " + (f"{fmt(r[0])} = {r[1] / 1000:.0f} kb/s" if r else "not reached"))
                cfg_f, c_f, m_f = pick(d, "accuracy", "total_flops")
                cfg_m, c_m, m_m = pick(d, "accuracy", "model_ops")
                print(f"{SHORT[p]} set {fset} {model}: " + "; ".join(parts_) + " | " + "; ".join(tgt)
                      + f" | FLOPs elbow {fmt(cfg_f)} = {c_f / 1000:.0f} kFLOP/decision ({100 * m_f:.1f}%); model-ops elbow {fmt(cfg_m)} = {c_m / 1000:.1f} kOps ({100 * m_m:.1f}%)")

    print("\n### 2. Main effects (mean accuracy %, over the other settings; per patient, feature set and model)")
    for axis in ("channels", "rate_hz", "bits"):
        print(f"\nBy {axis}")
        g = df.groupby(["patient", "feature_set", "model", axis]).accuracy.mean().unstack(axis) * 100
        print(g.round(1).to_string())

    print("\n### 3. W against A (paired by fold, same configuration and model; OPTIMISTIC: W was chosen on Stage 1 data)")
    rows = []
    for key, rec in R.items():
        part, rate, nch, bits, fset, model = parse(key)
        if fset != "W":
            continue
        a = R.get(key.replace("|W|", "|A|"))
        if a is None:
            continue
        d = np.array(rec["fold_accs"]) - np.array(a["fold_accs"])
        h = TCRIT * d.std(ddof=1) / np.sqrt(len(d))
        rows.append(dict(patient=SHORT[part], model=model, rate=rate, channels=nch, bits=bits, diff=100 * d.mean(), lo=100 * (d.mean() - h), hi=100 * (d.mean() + h)))
    w = pd.DataFrame(rows)
    if not w.empty:
        w["resolved_better"] = w.lo > 0
        w["resolved_worse"] = w.hi < 0
        print(w.groupby(["patient", "model"]).agg(mean_diff=("diff", "mean"), configs=("diff", "size"), resolved_better=("resolved_better", "sum"), resolved_worse=("resolved_worse", "sum")).round(2).to_string())
        print(f"\nAll cells: mean W - A {w['diff'].mean():+.2f} points over {len(w)} cells; resolved better {int(w.resolved_better.sum())}, worse {int(w.resolved_worse.sum())}")
        print("\nBy channel count (mean W - A):\n" + w.groupby(["channels"])["diff"].mean().round(2).to_string())
        print("\nBy sample rate (mean W - A):\n" + w.groupby(["rate"])["diff"].mean().round(2).to_string())

    print("\n### 3b. Models: mean accuracy by model over the grid (%, per patient and feature set), and best model count")
    print((df.groupby(["patient", "feature_set", "model"]).accuracy.mean().unstack("model") * 100).round(2).to_string())

    print("\n### 3c. Channel count, paired by fold: mean accuracy difference between channel counts at the same rate, bits, set and model (points)")
    ch_rows = []
    for key, rec in R.items():
        part, rate, nch, bits, fset, model = parse(key)
        for small, big in ((2, 5), (5, 8), (8, 12)):
            if nch != big:
                continue
            o = R.get(cell_key := f"{part}|{rate}Hz|{small}ch|{bits}bit|{fset}|{model}")
            if o is None:
                continue
            d = np.array(rec["fold_accs"]) - np.array(o["fold_accs"])
            ch_rows.append(dict(patient=SHORT[part], step=f"{small}->{big}", diff=100 * d.mean()))
    if ch_rows:
        print(pd.DataFrame(ch_rows).groupby(["patient", "step"])["diff"].mean().round(2).unstack("step").to_string())

    print("\n### 4. Reference configuration (1000 Hz x 8 ch x 16 b) against the accuracy elbow: accuracy, F1, kappa, cost")
    for p in PARTS:
        for fset in SETS:
            for model in MODELS:
                d = df[(df.part == p) & (df.feature_set == fset) & (df.model == model)]
                ref = d[(d.rate_hz == REF[0]) & (d.channels == REF[1]) & (d.bits == REF[2])]
                if d.empty or ref.empty:
                    continue
                cfg, c, m = pick(d, "accuracy", "bitrate_bps")
                el = d[(d.rate_hz == cfg[0]) & (d.channels == cfg[1]) & (d.bits == cfg[2])].iloc[0]
                r = ref.iloc[0]
                print(f"{SHORT[p]} {fset} {model:12s} ref: acc {100 * r.accuracy:.1f} F1 {100 * r.weighted_f1:.1f} kappa {r.kappa:.3f} {r.bitrate_bps / 1000:.0f} kb/s, {r.total_flops / 1000:.0f} kFLOP/dec (model {r.model_ops / 1000:.1f} kOps) | "
                      f"elbow {fmt(cfg)}: acc {100 * el.accuracy:.1f} F1 {100 * el.weighted_f1:.1f} kappa {el.kappa:.3f} {el.bitrate_bps / 1000:.0f} kb/s, {el.total_flops / 1000:.0f} kFLOP/dec (model {el.model_ops / 1000:.1f} kOps)")

    print("\n### 5. Step-3 winners (rule fixed in the protocol): elbows of accuracy and F1 for LDA/SVM/RF, cheapest config reaching 95% mean accuracy over the three models, reference")
    out = {}
    for p in PARTS:
        for fset in SETS:
            cfgs = set(winners.get(f"{p}|{fset}", set()))
            m3 = df[(df.part == p) & (df.feature_set == fset)].groupby(["rate_hz", "channels", "bits", "bitrate_bps"]).accuracy.agg(["mean", "size"]).reset_index()
            m3 = m3[m3["size"] == len(MODELS)]
            ok = m3[m3["mean"] >= 0.95].sort_values(["bitrate_bps", "rate_hz", "channels", "bits"])
            if not ok.empty:
                r = ok.iloc[0]
                cfgs.add((int(r.rate_hz), int(r.channels), int(r.bits)))
            cfgs.add(REF)
            out[f"{p}|{fset}"] = sorted(cfgs)
            print(f"{SHORT[p]} set {fset}: " + "; ".join(fmt(c) for c in sorted(cfgs)))
    json.dump(out, open(CACHE_DIR / "grid" / "winners.json", "w"), indent=1)

    print("\n### 6. Cost of the cells (sum of core-minutes)")
    sec = collections.Counter()
    for key, rec in R.items():
        sec[parse(key)[5]] += rec["seconds"]
    print("  " + ", ".join(f"{m} {s / 60:.0f} min" for m, s in sec.items()))


if __name__ == "__main__":
    main()
