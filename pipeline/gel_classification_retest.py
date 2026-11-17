"""Re-test of the current classifiers on data received through the gel phantom (polarised orientation).
Protocol, declared before any result: docs/gel_retest_declared_protocol.md.

For every gel run that saved its received stream (20 streams, 16 participant x configuration cells), the sample = movement segments wholly
inside the transmitted slice. Set-W features of those windows are recomputed from the decoded SENT and RECEIVED streams with the grid's own
feature function, and every model is trained per fold on the offline grid table (grouped 5-fold CV of the grid) and asked to classify the sample
windows of its test fold three ways: offline (grid table), sent, received.

Models: LDA, SVM, RF (+ causal hard vote over all windows since the onset), tuned RBFNN, tuned LSTM (grid step-3 procedures).
Output: cache/link/gel_retest/<cell>.json (resumable) and summary.md / summary.json (--summarise).

Usage:
  python -W ignore gel_classification_retest.py --check            # V-a, V-b, sample sizes, channel match; no models
  python -W ignore -u gel_classification_retest.py --workers 3     # all models, all cells
  python -W ignore gel_classification_retest.py --summarise
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "esp32_ble_link" / "results"
OUT = HERE / "cache" / "link" / "gel_retest"
W_PATH = HERE / "cache" / "wide_set_W.json"
GRID_RESULTS = HERE / "cache" / "grid" / "grid_results.json"

STREAM_DIRS = [d for d in (sorted((RES / "phantom_campaign_20261104_141740").glob("*kbps")) +
                            sorted((RES / "phantom_campaign_20261104_160046").glob("*kbps")) +
                            [RES / "20261104_134557_stream_vaskov-P1_1of12"])
               if any(d.glob("received_stream_*dBm.bin"))]          # the 960 kb/s runs were not recorded
OUT_POLARISED = OUT


def unpolarised_dirs():
    """Stream folders of the campaigns whose note marks the UNPOLARISED (worst-case) orientation (9 Nov 2026 onwards)."""
    out = []
    for camp in sorted(RES.glob("phantom_campaign_*")):
        try:
            m = json.load(open(camp / "manifest.json"))
        except (OSError, ValueError):
            continue
        notes = [str((r.get("result") or {}).get("placement_note", "")) for r in m.get("runs", {}).values()]
        if any("UNPOLARISED" in n for n in notes):
            out += [d for d in sorted(camp.glob("*kbps")) if any(d.glob("received_stream_*dBm.bin"))]
    return out


def use_orientation(name):
    """Switch the module to the unpolarised streams; results go to their own folder, the polarised results stay untouched."""
    global STREAM_DIRS, OUT
    if name == "unpolarised":
        STREAM_DIRS = unpolarised_dirs()
        OUT = HERE / "cache" / "link" / "gel_retest_unpolarised"


# ------------------------------------------------------------------ decoding (same formula as the grid's quantiser)
def unpack_bits(data, bits, n):
    a = np.frombuffer(data, dtype=np.uint8)
    if bits == 8:
        return a[:n].astype(np.uint16)
    if bits == 16:
        return np.frombuffer(data[: 2 * n], dtype="<u2").astype(np.uint16)
    if bits == 12:
        if len(a) % 3:
            a = np.append(a, np.zeros(3 - len(a) % 3, dtype=np.uint8))
        b0, b1, b2 = a[0::3].astype(np.uint16), a[1::3].astype(np.uint16), a[2::3].astype(np.uint16)
        x, y = b0 | ((b1 & 0x0F) << 8), (b1 >> 4) | (b2 << 4)
        return np.stack([x, y], axis=1).reshape(-1)[:n]
    raise ValueError(bits)


def decode(data, meta):
    F, C, bits = meta["n_frames"], meta["channels"], meta["bits"]
    codes = unpack_bits(data, bits, F * C).astype(np.float64).reshape(F, C)
    lo, hi = np.array(meta["lo"]), np.array(meta["hi"])
    step = (hi - lo) / (2 ** bits - 1)
    return (codes * step + lo).astype(np.float32)          # = quantize()/quantize_fixed(): round(...) * step + lo, cast to float32


def load_stream(d):
    meta = json.load(open(d / "stream_meta.json"))
    rec = sorted(d.glob("received_stream_*dBm.bin"))[0]
    sent, got = (d / "sent_stream.bin").read_bytes(), rec.read_bytes()
    miss = np.load(d / ("received_missing_" + rec.stem.split("_")[-1] + ".npy"))
    part = f"{meta['participant']}_combined" if meta["kind"] == "vaskov" else meta["participant"]
    return dict(dir=d, name=f"{d.parent.name}/{d.name}" if d.parent.name.startswith("phantom") else d.name, meta=meta, part=part,
                power=rec.stem.split("_")[-1], sent=sent, got=got, bytes_missing=int(miss.sum()),
                bit_exact=(sent == got) and not miss.any(), x_sent=decode(sent, meta), x_recv=decode(got, meta))


def cell_of(s):
    m = s["meta"]
    return f"{s['part']}|{m['fs']}Hz|{m['bits']}bit|" + "+".join(sorted(m["channel_names"]))


# ------------------------------------------------------------------ table, sample and stream features
def load_cell_table(part, fs, bits, chans):
    from build_wide_tables import channels_of
    from families import leaf_of
    from grid_slices import slice_matrix
    from grid_tables import table_path
    from models import compute_cv_folds
    W = set(json.load(open(W_PATH))["kept_leaves"])
    t = pd.read_parquet(table_path(part, fs, bits))
    cols = list(t.columns[5:])
    X, names = slice_matrix(t, [c for c in cols if leaf_of(c) in W], channels_of(part), list(chans))
    return t, X, names, compute_cv_folds(t)


def sample_and_features(t, names, s):
    """Rows of the sample (segments wholly inside the slice) and their features from the sent and received streams, in table row order."""
    from build_wide_tables import channels_of
    from vaskov_features import sliding_windows
    from wide_features import segment_columns
    m = s["meta"]
    fs, i0, nf = m["fs"], m["i0d"], m["n_frames"]
    win, hop = int(round(200 * fs / 1000)), max(1, int(round(50 * fs / 1000)))
    widx = np.rint(t["window_time_s"].to_numpy() * fs).astype(int)
    seg = t["segment_id"].to_numpy()
    in_file = t["segment_id"].str.startswith(m["file_key"] + "_").to_numpy()
    inside = in_file & (widx >= i0) & (widx + win <= i0 + nf)
    segs_in_file = pd.unique(seg[in_file])
    sample_segs = [g for g in segs_in_file if inside[seg == g].all()]
    rows = np.flatnonzero(np.isin(seg, sample_segs))
    order = [c for c in channels_of(s["part"]) if c in m["channel_names"]]          # table channel order (pair names follow it)
    pos = [m["channel_names"].index(c) for c in order]
    feats = {}
    for which, x in (("sent", s["x_sent"]), ("received", s["x_recv"])):
        blocks = []
        for g in sample_segs:
            r = np.flatnonzero(seg == g)
            st = widx[r] - i0
            assert np.all(np.diff(st) == hop), (g, np.diff(st)[:5])
            xws = [sliding_windows(x[st[0]: st[-1] + win, p], win, hop) for p in pos]
            assert xws[0].shape[0] == len(r), (g, xws[0].shape, len(r))
            cols = segment_columns(xws, order, fs)
            blocks.append(np.column_stack([np.asarray(cols[n], dtype=np.float64) for n in names]))
        F = np.vstack(blocks) if blocks else np.empty((0, len(names)))
        feats[which] = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return rows, sample_segs, feats


# ------------------------------------------------------------------ models
def classical(name, X, y, folds, sample_rows, Xs_by_stream):
    from models import make_model_pipeline
    from sklearn.preprocessing import StandardScaler
    pooled = np.empty_like(y)
    out = {k: np.empty(len(v["rows"]), dtype=y.dtype) for k, v in Xs_by_stream.items()}
    outs = {k: np.empty(len(v["rows"]), dtype=y.dtype) for k, v in Xs_by_stream.items()}
    for f in sorted(np.unique(folds)):
        tr, te = folds != f, folds == f
        need, est = make_model_pipeline(name)
        sc = StandardScaler().fit(X[tr]) if need else None
        tf = (lambda a: sc.transform(a)) if need else (lambda a: a)
        est.fit(tf(X[tr]), y[tr])
        pooled[te] = est.predict(tf(X[te]))
        for k, v in Xs_by_stream.items():
            sel = te[v["rows"]]
            if sel.any():
                out[k][sel] = est.predict(tf(v["received"][sel]))
                outs[k][sel] = est.predict(tf(v["sent"][sel]))
    return pooled, out, outs


def rbfnn_tuned(X, y, groups, folds, Xs_by_stream, fixed=None):
    """fixed = per-fold settings chosen earlier (list of dicts); then no search is run (unpolarised re-test)."""
    from models import RBFNetClassifier
    from run_rbfnn_tuning import tune_rbfnn
    from sklearn.preprocessing import StandardScaler
    pooled = np.empty_like(y)
    out = {k: np.empty(len(v["rows"]), dtype=y.dtype) for k, v in Xs_by_stream.items()}
    outs = {k: np.empty(len(v["rows"]), dtype=y.dtype) for k, v in Xs_by_stream.items()}
    chosen = []
    for f in sorted(np.unique(folds)):
        tr, te = folds != f, folds == f
        if fixed:
            d = fixed[int(f)]
            n_c, s, C = d["n_centers"], d["sigma_scale"], d["C"]
        else:
            (n_c, s, C), inner, _ = tune_rbfnn(X[tr], y[tr], groups[tr])
        sc = StandardScaler().fit(X[tr])
        clf = RBFNetClassifier(n_centers=n_c, C=C, sigma_scale=s).fit(sc.transform(X[tr]), y[tr])
        pooled[te] = clf.predict(sc.transform(X[te]))
        for k, v in Xs_by_stream.items():
            sel = te[v["rows"]]
            if sel.any():
                out[k][sel] = clf.predict(sc.transform(v["received"][sel]))
                outs[k][sel] = clf.predict(sc.transform(v["sent"][sel]))
        chosen.append(dict(n_centers=int(n_c), sigma_scale=float(s), C=float(C)))
    return pooled, out, outs, chosen


def lstm_tuned(t, X, names, folds, Xs_by_stream, fixed=None):
    """Grid step-3 procedure (run_grid_networks.lstm_tuned); returns window predictions on the sample, offline / sent / received, in table row order."""
    from sklearn.model_selection import StratifiedShuffleSplit
    import run_lstm_training_check as L
    from lstm_model import build_segment_sequences
    meta = t.iloc[:, :5].reset_index(drop=True)
    df = pd.concat([meta, pd.DataFrame(X, columns=names)], axis=1)
    _, feats, classes, c2i, by_id, seg_fold = L.prepare(df, names, folds)
    i2c = {i: c for c, i in c2i.items()}
    n_classes = len(classes)
    alt = {}
    for k, v in Xs_by_stream.items():                           # sequences of the sample segments built from the stream features
        for which in ("sent", "received"):
            d = df.iloc[v["rows"]].copy()
            d[names] = v[which]
            alt[(k, which)] = {sid: seq for sid, seq in ((q[0], q) for q in build_segment_sequences(d, names))}
    off_true, off_pred = [], []
    res = {k: {w: {} for w in ("offline", "sent", "received")} for k in Xs_by_stream}
    info = []
    for fold in sorted(seg_fold.unique()):
        tr, te, mean, std = L.fold_data(by_id, seg_fold, fold)
        if fixed:
            cname, epoch = fixed[int(fold)]["config"], fixed[int(fold)]["epochs"]
        else:
            cues = np.array([c for _, _, c in tr])
            val_size = max(n_classes, int(np.ceil(0.2 * len(tr))))
            itr_idx, iva_idx = next(StratifiedShuffleSplit(1, test_size=val_size, random_state=L.SEED).split(np.zeros(len(tr)), cues))
            itr, iva = [tr[i] for i in itr_idx], [tr[i] for i in iva_idx]
            best = None
            for cname, cfg in L.CONFIGS.items():
                _, hist = L.train(itr, {"val": iva}, c2i, n_classes, len(feats), L.MAX_EPOCHS, eval_every=5, **cfg)
                r_best = min([r for r in hist if "val_loss" in r], key=lambda r: r["val_loss"])
                if best is None or r_best["val_loss"] < best[2]:
                    best = (cname, r_best["epoch"], r_best["val_loss"])
            cname, epoch, _ = best
        model, _ = L.train(tr, {}, c2i, n_classes, len(feats), epoch, **L.CONFIGS[cname])
        _, _, pred, true = L.evaluate(model, te, c2i, n_classes)
        off_true.append(true)
        off_pred.append(pred)
        te_ids = {q[0]: q for q in te}
        for k in Xs_by_stream:
            ids = [sid for sid in alt[(k, "sent")] if sid in te_ids]
            for sid in ids:
                for which, seqs in (("offline", [te_ids[sid]]),
                                    ("sent", L.scaled([alt[(k, "sent")][sid]], mean, std)),
                                    ("received", L.scaled([alt[(k, "received")][sid]], mean, std))):
                    _, _, p, _ = L.evaluate(model, seqs, c2i, n_classes)
                    res[k][which][sid] = np.array([i2c[i] for i in p])
        info.append(dict(config=cname, epochs=int(epoch)))
    y_true = np.array([i2c[i] for i in np.concatenate(off_true)])
    y_pred = np.array([i2c[i] for i in np.concatenate(off_pred)])
    return y_true, y_pred, res, info


def vote(pred, seg_ids, classes):
    from fair_vote import causal_vote
    c2i = {c: i for i, c in enumerate(classes)}
    v = causal_vote(np.array([c2i[c] for c in pred]), seg_ids, None, len(classes))
    return np.array([classes[i] for i in v])


# ------------------------------------------------------------------ one cell
NET_CELLS = ("P1_combined|250Hz|8bit", "P2_combined|1000Hz|16bit", "Wallace_S1|1000Hz|16bit", "Wallace_S2|1000Hz|16bit")   # protocol Amendment 1


def run_cell(cell, stream_dirs, threads, with_nets=True, fixed_nets=None):
    warnings.filterwarnings("ignore")
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(threads)
    t0 = time.time()
    with threadpool_limits(limits=threads):
        streams = [load_stream(Path(d)) for d in stream_dirs]
        m = streams[0]["meta"]
        part, fs, bits, chans = streams[0]["part"], m["fs"], m["bits"], m["channel_names"]
        t, X, names, folds = load_cell_table(part, fs, bits, chans)
        y = t["cue"].to_numpy()
        seg = t["segment_id"].to_numpy()
        classes = sorted(np.unique(y).tolist())
        S = {}
        for s in streams:
            rows, segs, feats = sample_and_features(t, names, s)
            S[s["name"]] = dict(rows=rows, segs=segs, sent=feats["sent"], received=feats["received"], s=s)
        rec = {"cell": cell, "part": part, "fs": fs, "bits": bits, "channels": chans, "n_features": len(names), "n_windows": int(len(t)),
               "models": {}, "streams": {}}
        for k, v in S.items():
            s = v["s"]
            off = X[v["rows"]]
            rec["streams"][k] = dict(power=s["power"], bit_exact=bool(s["bit_exact"]), bytes_missing=s["bytes_missing"], stream_bytes=len(s["sent"]),
                                     sample_segments=len(v["segs"]), sample_windows=int(len(v["rows"])),
                                     sample_classes=sorted(np.unique(y[v["rows"]]).tolist()),
                                     vb_max_abs_diff_sent_vs_table=float(np.abs(v["sent"] - off).max()) if len(off) else None,
                                     vb_max_rel_diff=float((np.abs(v["sent"] - off) / (np.abs(off) + 1e-12)).max()) if len(off) else None,
                                     features_received_equal_sent=bool(np.array_equal(v["received"], v["sent"])), models={})

        def record(model, pooled, recv, sent, extra=None):
            rec["models"][model] = {"cv_accuracy_offline": float((pooled == y).mean()), **(extra or {})}
            for k, v in S.items():
                yo, ps = y[v["rows"]], pooled[v["rows"]]
                rec["streams"][k]["models"][model] = dict(
                    acc_offline=float((ps == yo).mean()), acc_sent=float((sent[k] == yo).mean()), acc_received=float((recv[k] == yo).mean()),
                    agree_received_vs_offline=float((recv[k] == ps).mean()), agree_received_vs_sent=float((recv[k] == sent[k]).mean()))

        grid = json.load(open(GRID_RESULTS))
        from run_bitrate_search import get_ranking_and_grid
        same_as_grid = set(get_ranking_and_grid(part)[0][:len(chans)]) == set(chans)
        rec["channels_equal_grid_subset"] = bool(same_as_grid)
        for name in ("LDA", "SVM", "RandomForest"):
            pooled, recv, sent = classical(name, X, y, folds, None, S)
            gk = f"{part}|{fs}Hz|{len(chans)}ch|{bits}bit|W|{name}"
            record(name, pooled, recv, sent, {"grid_accuracy": grid[gk]["accuracy"] if (same_as_grid and gk in grid) else None})
            # causal vote over all windows since the onset, applied within each segment
            pv = vote(pooled, seg, classes)
            rv, sv = {}, {}
            for k, v in S.items():
                rv[k] = vote(recv[k], seg[v["rows"]], classes)
                sv[k] = vote(sent[k], seg[v["rows"]], classes)
            record(name + "+vote", pv, rv, sv)
        rec["networks_run"] = bool(with_nets)
        if not with_nets:
            rec["seconds"] = time.time() - t0
            return cell, rec
        fixed_nets = fixed_nets or {}
        rec["network_settings_reused_from_polarised"] = bool(fixed_nets)
        pooled, recv, sent, chosen = rbfnn_tuned(X, y, t["segment_id"].to_numpy(), folds, S, fixed_nets.get("RBFNN_tuned"))
        record("RBFNN_tuned", pooled, recv, sent, {"fold_info": chosen})
        y_true, y_pred, res, info = lstm_tuned(t, X, names, folds, S, fixed_nets.get("LSTM_tuned"))
        rec["models"]["LSTM_tuned"] = {"cv_accuracy_offline": float((y_true == y_pred).mean()), "fold_info": info}
        for k, v in S.items():
            ids = v["segs"]
            got = {w: np.concatenate([res[k][w][sid] for sid in ids]) for w in ("offline", "sent", "received")}
            yo = np.concatenate([y[seg == sid] for sid in ids])
            rec["streams"][k]["models"]["LSTM_tuned"] = dict(
                acc_offline=float((got["offline"] == yo).mean()), acc_sent=float((got["sent"] == yo).mean()),
                acc_received=float((got["received"] == yo).mean()),
                agree_received_vs_offline=float((got["received"] == got["offline"]).mean()),
                agree_received_vs_sent=float((got["received"] == got["sent"]).mean()))
        rec["seconds"] = time.time() - t0
    return cell, rec


def cells():
    by = {}
    for d in STREAM_DIRS:
        s = load_stream(d)
        by.setdefault(cell_of(s), []).append(str(d))
    return by


def check():
    print(f"{len(STREAM_DIRS)} streams")
    by = cells()
    print(f"{len(by)} cells")
    from run_bitrate_search import get_ranking_and_grid
    for cell, dirs in by.items():
        s0 = load_stream(Path(dirs[0]))
        m = s0["meta"]
        t, X, names, folds = load_cell_table(s0["part"], m["fs"], m["bits"], m["channel_names"])
        same = set(get_ranking_and_grid(s0["part"])[0][:len(m["channel_names"])]) == set(m["channel_names"])
        for d in dirs:
            s = load_stream(Path(d))
            rows, segs, feats = sample_and_features(t, names, s)
            off = X[rows]
            diff = float(np.abs(feats["sent"] - off).max()) if len(rows) else float("nan")
            print(f"{s['name']:52s} {s['power']:>6s} bit-exact {s['bit_exact']!s:5s} grid-channels {same!s:5s} F={len(names):3d} "
                  f"sample {len(segs):2d} segs {len(rows):4d} win, classes {len(np.unique(t['cue'].to_numpy()[rows])):2d}, "
                  f"V-b max|sent-table| {diff:.3g}, recv==sent features {np.array_equal(feats['received'], feats['sent'])}")


def summarise():
    recs = [json.load(open(p)) for p in sorted(OUT.glob("*.json")) if p.name != "summary.json"]
    models = ["LDA", "SVM", "RandomForest", "LDA+vote", "SVM+vote", "RandomForest+vote", "RBFNN_tuned", "LSTM_tuned"]
    lines = ["# Gel re-test: offline vs received (sample windows); protocol docs/gel_retest_declared_protocol.md", "",
             "| Stream | Setting | Bit-exact | Segments | Windows | " + " | ".join(models) + " |",
             "|---|---|---|---|---|" + "---|" * len(models)]
    tot = {mm: [0, 0, 0, 0] for mm in models}        # windows, agree, correct offline, correct received
    allw = 0
    for r in recs:
        for k, s in r["streams"].items():
            cells_ = []
            for mm in models:
                if mm not in s["models"]:
                    cells_.append("--")
                    continue
                q = s["models"][mm]
                n = s["sample_windows"]
                tot[mm][0] += n
                tot[mm][1] += round(q["agree_received_vs_offline"] * n)
                tot[mm][2] += round(q["acc_offline"] * n)
                tot[mm][3] += round(q["acc_received"] * n)
                cells_.append(f"{100 * q['acc_offline']:.1f} / {100 * q['acc_received']:.1f} ({100 * q['agree_received_vs_offline']:.0f}%)")
            allw += s["sample_windows"]
            lines.append(f"| {k} | {s['power']} | {s['bit_exact']} | {s['sample_segments']} | {s['sample_windows']} | " + " | ".join(cells_) + " |")
    lines += ["", "Cells: accuracy offline / received on the sample windows (agreement of received with offline predictions).", "",
              "| Model | Sample | Agreement | Accuracy offline | Accuracy received |", "|---|---|---|---|---|"]
    summ = {}
    for mm in models:
        n, a, co, cr = tot[mm]
        nstreams = sum(1 for r in recs for s in r["streams"].values() if mm in s["models"])
        summ[mm] = dict(windows=n, streams=nstreams, agreement=a / n, acc_offline=co / n, acc_received=cr / n)
        lines.append(f"| {mm} | {nstreams} streams, {n} windows | {100 * a / n:.2f}% | {100 * co / n:.1f}% | {100 * cr / n:.1f}% |")
    checks = []
    for r in recs:
        for mm in ("LDA", "SVM", "RandomForest"):
            g = r["models"][mm].get("grid_accuracy")
            if g is not None:
                checks.append(abs(g - r["models"][mm]["cv_accuracy_offline"]))
    vb = [s["vb_max_abs_diff_sent_vs_table"] for r in recs for s in r["streams"].values() if s["vb_max_abs_diff_sent_vs_table"] is not None]
    lines += ["", f"V-a bit-exact: {sum(s['bit_exact'] for r in recs for s in r['streams'].values())} of {sum(len(r['streams']) for r in recs)} streams",
              f"V-b largest |sent features - table features|: {max(vb):.3g}",
              f"V-c LDA/SVM/RF CV accuracy vs grid_results: {len(checks)} comparisons, largest difference {max(checks) if checks else float('nan'):.3g}"]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump({"per_model": summ, "streams": sum(len(r["streams"]) for r in recs), "sample_windows": allw}, open(OUT / "summary.json", "w"), indent=1)
    print("\n".join(lines))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    args = sys.argv[1:]
    orientation = args[args.index("--orientation") + 1] if "--orientation" in args else "polarised"
    use_orientation(orientation)
    print(f"orientation: {orientation}; {len(STREAM_DIRS)} streams; output {OUT}", flush=True)
    if "--check" in args:
        return check()
    if "--summarise" in args:
        return summarise()
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 1
    OUT.mkdir(parents=True, exist_ok=True)
    by = cells()
    keyfile = {c: OUT / (c.replace("|", "__").replace("+", "-") + ".json") for c in by}
    todo = [(c, d) for c, d in by.items() if not keyfile[c].exists()]
    if "--smoke" in args:
        todo = todo[:1]
    threads = max(1, 15 // workers)
    nets = {c: any(c.startswith(n + "|") for n in NET_CELLS) for c, _ in todo}
    todo.sort(key=lambda cd: not nets[cd[0]])                 # network cells first (the long ones)
    print(f"{len(todo)} cells to run ({sum(nets.values())} with networks), {workers} worker(s), {threads} thread(s) each", flush=True)
    fixed = {}
    if orientation == "unpolarised":                      # same network settings as the polarised re-test of the same cell: no new search
        for c, _ in todo:
            pf = OUT_POLARISED / (c.replace("|", "__").replace("+", "-") + ".json")
            if nets[c] and pf.exists():
                pm = json.load(open(pf))["models"]
                if "RBFNN_tuned" in pm and "LSTM_tuned" in pm:
                    fixed[c] = {m: pm[m]["fold_info"] for m in ("RBFNN_tuned", "LSTM_tuned")}
        print(f"network settings reused from the polarised re-test for {len(fixed)} cell(s)", flush=True)
    from joblib import Parallel, delayed
    for cell, rec in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_cell)(c, d, threads, nets[c], fixed.get(c)) for c, d in todo):
        json.dump(rec, open(keyfile[cell], "w"), indent=1)
        a = {k.split("/")[-1]: [round(v["models"][m]["agree_received_vs_offline"], 4) for m in v["models"]] for k, v in rec["streams"].items()}
        print(f"  {cell[:40]} done in {rec['seconds']:.0f}s; agreement per stream, all models {a}", flush=True)
    print("All done")


if __name__ == "__main__":
    main()
