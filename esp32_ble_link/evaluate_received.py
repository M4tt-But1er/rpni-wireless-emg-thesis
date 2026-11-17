#!/usr/bin/env python3
"""evaluate_received.py -- feed the samples that came out of the BLE link back into the thesis classifiers.

Input: a results folder written by ble_link_test.py (stream mode), which contains
  sent_stream.bin, received_stream_<P>dBm.bin, received_missing_<P>dBm.npy, stream_meta.json

What it does
  1. Decodes the bytes back to samples (undoing the bit packing and the quantiser) for the stream that was SENT
     and the stream that was RECEIVED. Samples in packets that never arrived are filled (--fill zero|hold).
  2. Builds the training set with the thesis pipeline at the SAME configuration (sample rate, channels, bit depth),
     from the whole dataset EXCEPT every movement segment that overlaps the transmitted slice, so the model never
     sees the data it is tested on.
  3. Trains the classifiers (Tier 1 + Tier 2 feature selection, then RF / SVM with the project's default settings).
  4. Classifies the windows of the transmitted slice three ways and compares:
       offline    features from the offline pipeline (what the bitrate sweeps used)
       sent       features from the decoded sent stream  (checks my decoding against the offline pipeline)
       received   features from the decoded received stream (what the link delivered)
     The windows are the ones the offline pipeline found (its onset detection and segment labels), the same
     convention as the packet-loss test.

Usage:  python evaluate_received.py results/<run folder> [--power 9] [--models RandomForest SVM] [--fill zero]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))


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


def decode(data, missing_bytes, meta, fill):
    """bytes -> (frames x channels float32 signal, bool mask of samples that were never received)."""
    F, C, bits = meta["n_frames"], meta["channels"], meta["bits"]
    n = F * C
    codes = unpack_bits(data, bits, n).astype(np.float64).reshape(F, C)
    idx = np.arange(n)
    miss = (missing_bytes[(idx * bits) // 8] | missing_bytes[((idx + 1) * bits - 1) // 8]).reshape(F, C)
    lo, hi = np.array(meta["lo"]), np.array(meta["hi"])
    step = np.where(hi > lo, (hi - lo) / (2 ** bits - 1), 1.0)
    x = codes * step + lo
    if fill == "zero":                       # zero in the filtered (zero-mean) domain, as in the packet-loss test
        x[miss] = 0.0
    else:                                    # hold the last good sample
        x = pd.DataFrame(x).mask(miss).ffill().fillna(0.0).to_numpy()
    return x.astype(np.float32), miss


def window_features(x, starts, fs, chans, win):
    from features import extract_channel_features
    cols = {}
    for j, ch in enumerate(chans):
        xw = np.stack([x[s:s + win, j] for s in starts])
        for name, arr in extract_channel_features(xw, fs=float(fs)).items():
            cols[f"{ch}_{name}"] = arr
    return pd.DataFrame(cols)


def clean(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--power", type=int, help="which transmit-power recording to use (default: the first one)")
    ap.add_argument("--models", nargs="+", default=["RandomForest", "SVM"])
    ap.add_argument("--fill", choices=["zero", "hold"], default="zero")
    args = ap.parse_args()

    run = Path(args.run_dir)
    meta = json.load(open(run / "stream_meta.json"))
    if meta.get("kind") not in ("vaskov", "wallace") or meta.get("raw"):
        sys.exit("This run has no dataset behind it (synthetic or --raw), so there is nothing to classify.")
    recs = sorted(run.glob("received_stream_*dBm.bin"))
    if args.power is not None:
        recs = [r for r in recs if r.name == f"received_stream_{args.power}dBm.bin"]
    if not recs:
        sys.exit("No received_stream_*dBm.bin in that folder (was the run made with --no-record, or burst mode?)")
    rec = recs[0]
    pw = rec.stem.split("_")[-1]
    sent = (run / "sent_stream.bin").read_bytes()
    got = rec.read_bytes()
    miss_b = np.load(run / f"received_missing_{pw}.npy")
    fs, bits, chans = meta["fs"], meta["bits"], meta["channel_names"]
    print(f"{meta['source']}: {meta['channels']} ch x {fs} Hz x {bits} bit, {meta['seconds']:.1f} s starting at {meta['start_s']:.1f} s; "
          f"recording {rec.name}: {int(miss_b.sum())} of {len(miss_b)} bytes missing")

    # ---- decode both streams
    x_sent, _ = decode(sent, np.zeros(len(sent), bool), meta, args.fill)
    x_recv, miss = decode(got, miss_b, meta, args.fill)
    print(f"decoded: {int(miss.any(axis=1).sum())} of {len(miss)} sample frames are affected by lost packets")

    # ---- training windows from the offline pipeline at the same configuration
    print("building the offline pipeline features for this configuration (a few minutes the first time)...")
    if meta["kind"] == "vaskov":
        from bitrate_sweep import build_one_config
        from run_bitrate_search import DATASET_SPECS
        raws = DATASET_SPECS[f"{meta['participant']}_combined"]["raw_names"]
        df = pd.concat([build_one_config(r, fs, chans, bits, "eval", snr_db=None) for r in raws], ignore_index=True)
    else:
        from wallace_bitrate import build_wallace_config
        rng = {c: (lo, hi) for c, lo, hi in zip(chans, meta["lo"], meta["hi"])}
        df = build_wallace_config(meta["participant"], fs, chans, bits, quant_ranges=rng)
    from bitrate_sweep import WINDOW_MS_SWEEP
    win = int(round(WINDOW_MS_SWEEP * fs / 1000))
    i0, n_frames = meta["i0d"], meta["n_frames"]
    widx = np.rint(df["window_time_s"].to_numpy() * fs).astype(int)
    in_file = df["segment_id"].str.startswith(meta["file_key"] + "_").to_numpy()
    test = in_file & (widx >= i0) & (widx + win <= i0 + n_frames)
    touches = in_file & (widx + win > i0) & (widx < i0 + n_frames)
    if not test.any():
        sys.exit("The transmitted slice contains no movement windows; pass --start <seconds> to ble_link_test.py.")
    train = ~df["segment_id"].isin(set(df.loc[touches, "segment_id"]))
    test_classes = sorted(df.loc[test, "cue"].unique().tolist())
    missing_classes = sorted(set(test_classes) - set(df.loc[train, "cue"].unique().tolist()))
    print(f"training windows {int(train.sum())}, test windows {int(test.sum())} in {df.loc[test, 'segment_id'].nunique()} segments, "
          f"classes in test: {test_classes}")
    if missing_classes:
        # Every segment of these classes overlaps the transmitted slice, so none is left to train on. The models can never predict
        # them and the accuracy is 0 whatever the link does: it is an artefact of the hold-out, not a result.
        print(f"WARNING: classes {missing_classes} have NO training windows once the slice is left out; accuracy on them is not "
              f"meaningful. Only the agreement between the sent and received predictions is (it measures what the link changed).")

    from run_packet_loss_test import reduce_features
    from models import make_model_pipeline
    from sklearn.preprocessing import StandardScaler
    reduced = reduce_features(df[train])
    df_test = df[test].reset_index(drop=True)
    y = df_test["cue"].to_numpy()
    starts = widx[test] - i0
    damaged = np.array([miss[s:s + win].any() for s in starts])
    X = {
        "offline": clean(df_test[reduced]),
        "sent": clean(window_features(x_sent, starts, fs, chans, win)[reduced]),
        "received": clean(window_features(x_recv, starts, fs, chans, win)[reduced]),
    }
    covered = ~np.isin(y, missing_classes)                  # test windows whose class has training data
    results = {"run": str(run), "power": pw, "source": meta["source"], "fs": fs, "channels": len(chans), "bits": bits,
               "fill": args.fill, "test_windows": int(test.sum()), "damaged_windows": int(damaged.sum()),
               "bytes_missing": int(miss_b.sum()), "n_features": len(reduced), "test_classes": [int(c) for c in test_classes],
               "classes_missing_from_training": [int(c) for c in missing_classes],
               "accuracy_meaningful": bool(len(missing_classes) == 0), "windows_with_trainable_class": int(covered.sum()),
               "models": {}}
    print(f"\nfeatures used: {len(reduced)}; windows touched by lost data: {int(damaged.sum())} of {len(damaged)}\n")
    print(f"{'model':14}{'offline':>10}{'sent':>10}{'received':>10}{'sent=recv':>11}{'acc (damaged)':>15}{'acc (intact)':>14}")
    for name in args.models:
        scale, est = make_model_pipeline(name)
        Xtr, ytr = clean(df.loc[train, reduced]), df.loc[train, "cue"].to_numpy()
        sc = StandardScaler().fit(Xtr) if scale else None
        tf = (lambda a: sc.transform(a)) if scale else (lambda a: a)
        est.fit(tf(Xtr), ytr)
        pred = {k: est.predict(tf(v)) for k, v in X.items()}
        acc = {k: float((p == y).mean()) for k, p in pred.items()}
        agree = float((pred["sent"] == pred["received"]).mean())
        ok = pred["received"] == y
        a_dam = float(ok[damaged].mean()) if damaged.any() else None
        a_ok = float(ok[~damaged].mean()) if (~damaged).any() else None
        results["models"][name] = dict(accuracy=acc, agreement_sent_vs_received=agree, acc_received_damaged=a_dam,
                                       acc_received_intact=a_ok,
                                       accuracy_on_trainable_classes=({k: float((p[covered] == y[covered]).mean()) for k, p in pred.items()}
                                                                      if covered.any() and missing_classes else None))
        fmt = lambda v: "   n/a" if v is None else f"{v * 100:6.1f}%"
        print(f"{name:14}{acc['offline'] * 100:9.1f}%{acc['sent'] * 100:9.1f}%{acc['received'] * 100:9.1f}%{agree * 100:10.1f}%"
              f"{fmt(a_dam):>15}{fmt(a_ok):>14}")
    out = run / f"evaluation_{pw}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved {out}")
    print("Reading it: 'sent' should match 'offline' (same pipeline, my decoder). 'received' differs from 'sent' only through\n"
          "packets the link lost. A test slice has only tens of windows, so single-digit differences are noise.")


if __name__ == "__main__":
    main()
