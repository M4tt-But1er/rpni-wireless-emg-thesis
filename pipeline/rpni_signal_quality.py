"""EXPLORATORY (docs/rpni_vs_muscle_declared_protocol.md), test 3: signal quality per channel from the raw 30 kHz recordings.
Per source file / run: in-band (100-499 Hz, the pipeline's filter) and broadband (20 Hz-7 kHz) zero-phase 4th-order Butterworth; the pipeline's
own segmentation (movement segments from the detected onset, rest segments trimmed by 250 ms at each end); per-segment RMS; Welch PSD
(Hann, 3000 samples = 0.1 s) of the broadband signal per segment. Output: cache/rpni_vs_muscle/signal_quality.json with per channel
the per-segment RMS (by cue) for both bands and the length-weighted mean PSD of movement and rest segments.
Usage: python -W ignore -u rpni_signal_quality.py [--workers N]"""
import json
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import welch

from bitrate_sweep import build_segments_variable_rate, variable_rate_filter_downsample
from build_wide_tables import PARTS, channels_of
from config import CACHE_DIR, REST_CUE
from run_bitrate_search import DATASET_SPECS
from wallace_data import PARTICIPANTS as WALLACE

FS = 30000
NPERSEG = 3000
OUT = CACHE_DIR / "rpni_vs_muscle" / "signal_quality.json"


def sources():
    out = []
    for part in PARTS:
        if part.startswith("P"):
            out += [(part, n) for n in DATASET_SPECS[part]["raw_names"]]
        else:
            out += [(part, r) for r in WALLACE[part]["run_for_task"].values()]
    return out


def one_source(part, name):
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits
    t0 = time.time()
    with threadpool_limits(limits=1):
        if part.startswith("P"):
            from data_io import load_raw
            df = load_raw(name)
        else:
            from wallace_data import load_wallace_run
            df = load_wallace_run(part, name)
        cue = df["cue"].to_numpy()
        chans = channels_of(part)
        inband = {ch: variable_rate_filter_downsample(df[ch].to_numpy(), FS)[0] for ch in chans}
        segs = build_segments_variable_rate(cue, inband, FS)
        res = {}
        for ch in chans:
            broad = variable_rate_filter_downsample(df[ch].to_numpy(), FS, band=(20.0, 7000.0))[0].astype(np.float64)
            x_in = inband[ch].astype(np.float64)
            rows, psd = [], {"move": [np.zeros(NPERSEG // 2 + 1), 0.0], "rest": [np.zeros(NPERSEG // 2 + 1), 0.0]}
            for cue_val, start, end, tstart in segs:
                a = start if cue_val == REST_CUE else tstart
                if end - a < NPERSEG:
                    continue
                rows.append((int(cue_val), float(np.sqrt(np.mean(x_in[a:end] ** 2))), float(np.sqrt(np.mean(broad[a:end] ** 2))), int(end - a)))
                f, p = welch(broad[a:end], fs=FS, window="hann", nperseg=NPERSEG)
                k = "rest" if cue_val == REST_CUE else "move"
                psd[k][0] += p * (end - a)
                psd[k][1] += end - a
            res[ch] = dict(segments=rows, psd_move_sum=psd["move"][0].tolist(), n_move=psd["move"][1],
                           psd_rest_sum=psd["rest"][0].tolist(), n_rest=psd["rest"][1])
    return part, name, res, f.tolist(), round(time.time() - t0, 1)


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = {"freqs": None, "sources": {}}
    t0 = time.time()
    for part, name, res, f, s in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(one_source)(*a) for a in sources()):
        out["freqs"] = f
        out["sources"][f"{part}|{name}"] = res
        print(f"  {part} {name}: {s}s  (total {time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(OUT, "w"))
    print("All done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
