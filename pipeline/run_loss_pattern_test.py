"""X3 of docs/three_experiments_declared_protocol.md: burst loss against an equal amount of isolated loss.

Operating points (all channels, set W): 500 Hz x 8 bit and 1 kHz x 16 bit. 240-byte packets numbered from the start of each recording / run.
Loss levels p = 2, 5, 10, 20 % of packets; patterns isolated (no two lost packets adjacent), short bursts (~200 ms), long bursts (~1 s); 3 random
placements each. The receiver zero-fills lost samples; segments and windows come from the clean signal; features are recomputed for every segment
that contains a lost sample. LDA, SVM, RF trained on the CLEAN training folds of the grid table, tested on the lossy held-out fold; per-window,
causal hard vote (all since onset) and vote with damaged windows skipped.

Output: cache/loss_pattern/results.json keyed "<part>|<rate>|<bits>" (resumable); summarise with --summarise.
Usage: python -W ignore -u run_loss_pattern_test.py [--workers N] [--parts ...] [--check]   |   --summarise"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bitrate_sweep import build_segments_variable_rate, quantize
from build_wide_tables import PARTS, channels_of
from config import CACHE_DIR, REST_CUE
from fair_vote import causal_vote, segment_starts
from families import leaf_of
from grid_slices import slice_matrix
from grid_tables import decimated_sources, table_path, wallace_ranges
from models import compute_cv_folds
from run_bitrate_search import get_ranking_and_grid
from vaskov_features import sliding_windows
from wallace_bitrate import HOP_MS, WINDOW_MS, quantize_fixed
from wide_features import segment_columns

OUT_DIR = CACHE_DIR / "loss_pattern"
RES = OUT_DIR / "results.json"
W_PATH = CACHE_DIR / "wide_set_W.json"
GRID_RES = CACHE_DIR / "grid" / "grid_results.json"
CONFIGS = [(500, 8), (1000, 16)]
LEVELS = [0.02, 0.05, 0.10, 0.20]
PATTERNS = {"isolated": None, "burst_200ms": 200.0, "burst_1s": 1000.0}
REPEATS = 3
PAYLOAD = 240
MODELS = ["LDA", "SVM", "RandomForest"]


# ------------------------------------------------------------------ loss patterns
def place_runs(n, n_runs, L, rng):
    """Start packets of n_runs runs of L lost packets in n packets, uniformly at random, never overlapping or touching (>= 1 received packet between runs)."""
    if n_runs <= 0:
        return np.array([], dtype=int)
    free = n - n_runs * (L + 1) + 1
    assert free >= 0, (n, n_runs, L)
    v = np.sort(rng.choice(free + n_runs, n_runs, replace=False))
    return v + np.arange(n_runs) * L


def packet_loss(n_packets, p, run_len, rng):
    """Boolean lost-packet vector. run_len = 1: isolated; > 1: bursts of run_len packets. K = round(p n) lost packets (bursts: round(K / L) runs)."""
    K = int(round(p * n_packets))
    n_runs = K if run_len == 1 else int(round(K / run_len))
    lost = np.zeros(n_packets, dtype=bool)
    for s in place_runs(n_packets, n_runs, run_len, rng):
        lost[s:s + run_len] = True
    return lost


def run_length(pattern, packet_ms):
    dur = PATTERNS[pattern]
    return 1 if dur is None else max(2, int(round(dur / packet_ms)))


# ------------------------------------------------------------------ features
def quantised_sources(part, rate, bits):
    srcs = decimated_sources(part, rate)
    channels = channels_of(part)
    ranges = wallace_ranges(srcs, channels) if part.startswith("Wallace") else None
    out = []
    for data_f, cue, time_s, prefix, participant, label in srcs:
        if part.startswith("P"):
            data = {ch: quantize(data_f[ch], bits) for ch in channels}
        else:
            data = {ch: quantize_fixed(data_f[ch], bits, *ranges[ch]) for ch in channels}
        segs = [s for s in build_segments_variable_rate(cue, data, rate) if s[0] != REST_CUE]
        out.append((data, cue, segs, prefix))
    return out


def seg_features(data, channels, rate, trimmed, end, win, hop):
    xws = [sliding_windows(data[ch][trimmed:end], win, hop) for ch in channels]
    if xws[0].shape[0] == 0:
        return None
    cols = segment_columns(xws, channels, rate)
    return pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def run_cell(part, rate, bits, check_only=False):
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    import run_grid as G
    from sklearn.preprocessing import StandardScaler   # noqa: F401 (used through G)
    t_start = time.time()
    channels = channels_of(part)
    C = len(channels)
    fpp = PAYLOAD // (C * bits // 8)
    packet_ms = 1000.0 * fpp / rate
    win = int(round(WINDOW_MS * rate / 1000))
    hop = max(1, int(round(HOP_MS * rate / 1000)))
    Wl = set(json.load(open(W_PATH))["kept_leaves"])

    grid = pd.read_parquet(table_path(part, rate, bits))
    allcols = list(grid.columns[5:])
    wcols = [c for c in allcols if leaf_of(c) in Wl]
    keep = get_ranking_and_grid(part)[0][:C]
    X_clean, names = slice_matrix(grid, wcols, channels, keep)
    y = grid["cue"].to_numpy()
    folds = compute_cv_folds(grid)
    seg_ids = grid["segment_id"].to_numpy()
    labels = sorted(np.unique(y).tolist())
    c2i = {c: i for i, c in enumerate(labels)}
    yi = np.array([c2i[c] for c in y])

    # rows of the grid table per (source, segment): the grid's segment ids are "<prefix>_<seg_id>", rows in build order
    srcs = quantised_sources(part, rate, bits)
    row_of = {}
    b = segment_starts(seg_ids)
    for i, j in zip(b[:-1], b[1:]):
        row_of[seg_ids[i]] = (i, j)

    # check: recomputed clean features equal the grid table (W columns, exactly), segment by segment
    n_checked, max_diff = 0, 0.0
    for data, cue, segs, prefix in srcs:
        for k, (cv, s, e, tr) in enumerate(segs):
            sid = f"{prefix}_{k}"
            if sid not in row_of:
                continue
            f = seg_features(data, channels, rate, tr, e, win, hop)
            Xs, _ = slice_matrix(pd.concat([grid.iloc[row_of[sid][0]:row_of[sid][1], :5].reset_index(drop=True), f], axis=1), wcols, channels, keep)
            max_diff = max(max_diff, float(np.abs(Xs - X_clean[row_of[sid][0]:row_of[sid][1]]).max()))
            n_checked += 1
            if check_only and n_checked >= 5:
                break
        if check_only and n_checked >= 5:
            break
    assert max_diff == 0.0, f"{part} {rate} {bits}: recomputed clean features differ from the grid table by {max_diff}"
    if check_only:
        return f"{part}|{rate}|{bits}", {"check_segments": n_checked, "max_diff": max_diff, "fpp": fpp, "packet_ms": packet_ms}

    # models fitted once per fold on the clean training rows
    fitted = {}
    with threadpool_limits(limits=1):
        for m in MODELS:
            for f in range(folds.max() + 1):
                tr = folds != f
                need, est = __import__("models").make_model_pipeline(m)
                if m == "RandomForest":
                    est.set_params(n_jobs=2)
                sc = None
                if need:
                    from sklearn.preprocessing import StandardScaler
                    sc = StandardScaler().fit(X_clean[tr])
                est.fit(sc.transform(X_clean[tr]) if sc is not None else X_clean[tr], y[tr])
                fitted[(m, f)] = (sc, est)

        def predict(m, X, rows):
            out = np.empty(len(rows), dtype=object)
            for f in range(folds.max() + 1):
                sel = folds[rows] == f
                if sel.any():
                    sc, est = fitted[(m, f)]
                    Xf = X[sel]
                    out[sel] = est.predict(sc.transform(Xf) if sc is not None else Xf)
            return out

        all_rows = np.arange(len(y))
        clean_pred = {m: predict(m, X_clean, all_rows) for m in MODELS}

    def scores(pred, damaged):
        pi = np.array([c2i[c] for c in pred])
        ok = pi == yi
        vote = causal_vote(pi, seg_ids, None, len(labels))
        skip = vote_skip(pi, damaged, seg_ids, len(labels))
        res = {}
        for name, hit in (("window", ok), ("vote", vote == yi), ("vote_skip", skip == yi)):
            res[name] = dict(accuracy=float(hit.mean()), fold_accs=[float(hit[folds == f].mean()) for f in range(folds.max() + 1)])
        res["window_intact_accuracy"] = float(ok[~damaged].mean()) if (~damaged).any() else None
        return res

    out = {"fpp": fpp, "packet_ms": packet_ms, "n_windows": int(len(y)), "check_max_diff": max_diff, "check_segments": n_checked,
           "clean": {m: scores(clean_pred[m], np.zeros(len(y), dtype=bool)) for m in MODELS}, "conditions": {}}
    pi_idx = {"isolated": 0, "burst_200ms": 1, "burst_1s": 2}
    part_idx = PARTS.index(part)
    cfg_idx = CONFIGS.index((rate, bits))
    for li, p in enumerate(LEVELS):
        for pattern in PATTERNS:
            L = run_length(pattern, packet_ms)
            for rep in range(REPEATS):
                X = X_clean.copy()
                changed = np.zeros(len(y), dtype=bool)
                lost_pk, total_pk, lost_scored, total_scored = 0, 0, 0, 0
                for si, (data, cue, segs, prefix) in enumerate(srcs):
                    n_frames = len(cue)
                    n_pk = -(-n_frames // fpp)
                    rng = np.random.default_rng([part_idx, cfg_idx, li, pi_idx[pattern], rep, si])
                    lost = packet_loss(n_pk, p, L, rng)
                    lost_pk += int(lost.sum()); total_pk += n_pk
                    mask = np.repeat(lost, fpp)[:n_frames]
                    lossy = {ch: np.where(mask, np.float32(0), data[ch]) for ch in channels}
                    for k, (cv, s, e, tr) in enumerate(segs):
                        sid = f"{prefix}_{k}"
                        if sid not in row_of:
                            continue
                        a, bnd = row_of[sid]
                        lost_scored += int(mask[tr:e].sum()); total_scored += e - tr
                        if not mask[tr:e].any():
                            continue
                        f = seg_features(lossy, channels, rate, tr, e, win, hop)
                        Xs, _ = slice_matrix(pd.concat([grid.iloc[a:bnd, :5].reset_index(drop=True), f], axis=1), wcols, channels, keep)
                        X[a:bnd] = Xs
                damaged = (X != X_clean).any(axis=1)
                rows = np.flatnonzero(damaged)
                rec = {"run_len_packets": L, "run_ms": L * packet_ms, "lost_packets": lost_pk, "total_packets": total_pk,
                       "loss_realised": lost_pk / total_pk, "loss_in_scored_segments": lost_scored / max(total_scored, 1),
                       "damaged_share": float(damaged.mean())}
                with threadpool_limits(limits=1):
                    for m in MODELS:
                        pred = clean_pred[m].copy()
                        if len(rows):
                            pred[rows] = predict(m, X[rows], rows)
                        rec[m] = scores(pred, damaged)
                out["conditions"][f"{p:.2f}|{pattern}|{rep}"] = rec
                print(f"  {part} {rate}Hz {bits}bit p={100 * p:.0f}% {pattern:12s} rep {rep}: lost {100 * rec['loss_realised']:.1f}% "
                      f"(scored {100 * rec['loss_in_scored_segments']:.1f}%), damaged {100 * rec['damaged_share']:.1f}%, "
                      + "  ".join(f"{m[:3]} {100 * rec[m]['window']['accuracy']:.1f}/{100 * rec[m]['vote']['accuracy']:.1f}" for m in MODELS), flush=True)
    out["seconds"] = time.time() - t_start
    return f"{part}|{rate}|{bits}", out


def vote_skip(pi, damaged, seg_ids, n_classes):
    """Causal hard vote over the INTACT decisions since the onset; a damaged window shows the current vote; -1 if no intact decision yet in the segment."""
    res = np.full(len(pi), -1, dtype=int)
    b = segment_starts(seg_ids)
    for i, j in zip(b[:-1], b[1:]):
        idx = np.arange(i, j)
        ok = idx[~damaged[i:j]]
        if len(ok) == 0:
            continue
        v = causal_vote(pi[ok], np.zeros(len(ok), dtype=int), None, n_classes)
        pos = np.searchsorted(ok, idx, side="right") - 1          # latest intact window at or before each window
        has = pos >= 0
        res[idx[has]] = v[pos[has]]
    return res


def main():
    args = sys.argv[1:]
    if "--summarise" in args:
        return summarise()
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 2
    parts = PARTS
    if "--parts" in args:
        parts = []
        for a in args[args.index("--parts") + 1:]:
            if a.startswith("--"):
                break
            parts.append(a)
    check = "--check" in args
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = json.load(open(RES)) if RES.exists() and not check else {}
    tasks = [(p, r, b) for p in parts for r, b in CONFIGS if check or f"{p}|{r}|{b}" not in results]
    print(f"{len(tasks)} cells, {workers} worker(s)", flush=True)
    for key, rec in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_cell)(*t, check) for t in tasks):
        if check:
            print(key, rec, flush=True)
            continue
        results[key] = rec
        with open(RES, "w") as fh:
            json.dump(results, fh)
        g = json.load(open(GRID_RES))
        part, rate, bits = key.split("|")
        nch = len(channels_of(part))
        same = {m: abs(g[f"{part}|{rate}Hz|{nch}ch|{bits}bit|W|{m}"]["accuracy"] - rec["clean"][m]["window"]["accuracy"]) < 1e-12 for m in MODELS}
        print(f"  {key}: clean vs grid {same}  ({rec['seconds']:.0f}s)", flush=True)
    print("All done")


T4 = 2.7764451051977987          # t(0.975, 4 df)


def summarise():
    """Declared analysis: burst minus isolated accuracy at equal loss, paired by fold (fold accuracy averaged over the 3 repeats), 95 % t-interval;
    the thesis rule over the cells participant x model x level. Output cache/loss_pattern/summary.md and summary.json."""
    R = json.load(open(RES))
    modes = ["window", "vote", "vote_skip"]
    pats = list(PATTERNS)
    lines, out = [], {}
    for rate, bits in CONFIGS:
        keys = [k for k in R if k.endswith(f"|{rate}|{bits}")]
        parts = [k.split("|")[0] for k in keys]
        lines.append(f"\n## {rate} Hz x {bits} bit, all channels, set W ({len(keys)} participants)\n")
        # table 1: mean accuracy over participants and models
        lines.append("Mean accuracy (%) over participants and the three models; damaged = share of windows whose features changed; loss = realised share of samples lost inside the scored segments.\n")
        lines.append("| Loss | Pattern | run (ms) | loss in scored segments | damaged windows | per window | vote | vote, damaged skipped | per window, undamaged windows only |\n|---|---|---|---|---|---|---|---|---|")
        clean = {m: np.mean([R[k]["clean"][mo][m]["accuracy"] for k in keys for mo in MODELS]) for m in modes}
        lines.append("| 0 | clean | | 0 | 0 | " + " | ".join(f"{100 * clean[m]:.1f}" for m in modes) + " | |")
        for p in LEVELS:
            for pat in pats:
                cs = [R[k]["conditions"][f"{p:.2f}|{pat}|{r}"] for k in keys for r in range(REPEATS)]
                acc = {m: np.mean([c[mo][m]["accuracy"] for c in cs for mo in MODELS]) for m in modes}
                intact = np.mean([c[mo]["window_intact_accuracy"] for c in cs for mo in MODELS if c[mo]["window_intact_accuracy"] is not None])
                runms = sorted({round(c["run_ms"]) for c in cs})
                lines.append(f"| {100 * p:.0f}% | {pat} | {'/'.join(map(str, runms))} | {100 * np.mean([c['loss_in_scored_segments'] for c in cs]):.1f}% | "
                             f"{100 * np.mean([c['damaged_share'] for c in cs]):.1f}% | " + " | ".join(f"{100 * acc[m]:.1f}" for m in modes) + f" | {100 * intact:.1f} |")
        # table 2: declared comparison
        lines.append("\nBurst minus isolated (points), paired by fold. Cells = participant x model x loss level. 'worse' / 'better' = 95 % interval below / above zero.\n")
        lines.append("| Mode | Comparison | mean | cells burst worse (by sign) | cells resolved worse | cells resolved better | reading |\n|---|---|---|---|---|---|---|")
        for m in modes:
            for pat in pats[1:]:
                diffs, worse_sign, res_worse, res_better = [], 0, 0, 0
                per_level = {}
                for k in keys:
                    for mo in MODELS:
                        for p in LEVELS:
                            fb = np.mean([R[k]["conditions"][f"{p:.2f}|{pat}|{r}"][mo][m]["fold_accs"] for r in range(REPEATS)], axis=0)
                            fi = np.mean([R[k]["conditions"][f"{p:.2f}|isolated|{r}"][mo][m]["fold_accs"] for r in range(REPEATS)], axis=0)
                            d = 100 * (fb - fi)
                            mu, half = d.mean(), T4 * d.std(ddof=1) / np.sqrt(len(d))
                            diffs.append(mu)
                            per_level.setdefault(p, []).append(mu)
                            worse_sign += mu < 0
                            res_worse += (mu + half) < 0
                            res_better += (mu - half) > 0
                n = len(diffs)
                mean = float(np.mean(diffs))
                if abs(mean) < 1:
                    reading = "about equal"
                elif mean < 0 and res_worse >= n / 3:
                    reading = "bursts worse"
                elif mean > 0 and res_better >= n / 3:
                    reading = "bursts better"
                else:
                    reading = "unresolved"
                lines.append(f"| {m} | {pat} - isolated | {mean:+.2f} | {worse_sign}/{n} | {res_worse} | {res_better} | {reading} |")
                out[f"{rate}|{bits}|{m}|{pat}"] = dict(mean=mean, n=n, worse_sign=int(worse_sign), resolved_worse=int(res_worse), resolved_better=int(res_better),
                                                      reading=reading, mean_by_level={f"{p:.2f}": float(np.mean(v)) for p, v in per_level.items()})
        lines.append("\nMean difference by loss level (points):\n")
        lines.append("| Mode | Comparison | " + " | ".join(f"{100 * p:.0f}%" for p in LEVELS) + " |\n|---|---|" + "---|" * len(LEVELS))
        for m in modes:
            for pat in pats[1:]:
                bl = out[f"{rate}|{bits}|{m}|{pat}"]["mean_by_level"]
                lines.append(f"| {m} | {pat} - isolated | " + " | ".join(f"{bl[f'{p:.2f}']:+.2f}" for p in LEVELS) + " |")
        # per participant at 10 %
        lines.append("\nPer participant, 10 % loss, accuracy (%) per window / vote, mean of three models and three repeats:\n")
        lines.append("| Participant | clean | " + " | ".join(pats) + " |\n|---|---|" + "---|" * len(pats))
        for k, part in zip(keys, parts):
            cl = np.mean([R[k]["clean"][mo]["window"]["accuracy"] for mo in MODELS]), np.mean([R[k]["clean"][mo]["vote"]["accuracy"] for mo in MODELS])
            cells = []
            for pat in pats:
                cs = [R[k]["conditions"][f"0.10|{pat}|{r}"] for r in range(REPEATS)]
                cells.append(f"{100 * np.mean([c[mo]['window']['accuracy'] for c in cs for mo in MODELS]):.1f} / {100 * np.mean([c[mo]['vote']['accuracy'] for c in cs for mo in MODELS]):.1f}")
            lines.append(f"| {part} | {100 * cl[0]:.1f} / {100 * cl[1]:.1f} | " + " | ".join(cells) + " |")
    (OUT_DIR / "summary.md").write_text("# Burst against isolated loss (X3), generated by run_loss_pattern_test.py --summarise\n" + "\n".join(lines) + "\n", encoding="utf-8")
    json.dump(out, open(OUT_DIR / "summary.json", "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
