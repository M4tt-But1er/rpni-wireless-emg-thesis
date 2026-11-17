"""Re-summary of the bitrate grid with the wide set W only (user, 11 Nov 2026: W for everything except the feature-set comparisons).
Same cells and the same rules as analyse_grid.py; nothing is re-run. Output: cache/grid/grid_summary_W.txt (stdout).
Usage: python -W ignore analyse_grid_W.py > cache/grid/grid_summary_W.txt"""
import numpy as np
import pandas as pd

import analyse_grid as G


def main():
    df = G.table()
    w = df[df.feature_set == "W"]
    print(f"W cells: {len(w)} (expected 193 x 3 = 579)\n")

    print("### Main effects with W (mean accuracy %, averaged over the other two variables and over LDA, SVM and RF)")
    for axis in ("channels", "rate_hz", "bits"):
        g = w.groupby(["patient", axis]).accuracy.mean().unstack(axis) * 100
        print(f"\nBy {axis}\n" + g.round(1).to_string())
        print("all participants: " + ", ".join(f"{k}: {100 * v:.1f}" for k, v in w.groupby(axis).accuracy.mean().items()))

    print("\n### Channel count with W, paired by fold (same rate, bits and model): mean difference in points")
    rows = []
    for key, rec in G.R.items():
        part, rate, nch, bits, fset, model = G.parse(key)
        if fset != "W":
            continue
        for small, big in ((2, 5), (5, 8), (8, 12)):
            if nch != big:
                continue
            o = G.R.get(f"{part}|{rate}Hz|{small}ch|{bits}bit|W|{model}")
            if o is None:
                continue
            d = np.array(rec["fold_accs"]) - np.array(o["fold_accs"])
            rows.append(dict(patient=G.SHORT[part], step=f"{small}->{big}", diff=100 * d.mean()))
    print(pd.DataFrame(rows).groupby(["patient", "step"])["diff"].mean().round(2).unstack("step").to_string())

    print("\n### Sample rate with W, per patient (mean over channels, bits and models; %)")
    print((w.groupby(["patient", "rate_hz"]).accuracy.mean().unstack("rate_hz") * 100).round(1).to_string())
    print("\n### Bit depth with W, per patient (mean over rates, channels and models; %)")
    print((w.groupby(["patient", "bits"]).accuracy.mean().unstack("bits") * 100).round(1).to_string())
    print("\n### Models with W: mean accuracy over the grid (%)")
    print((w.groupby(["patient", "model"]).accuracy.mean().unstack("model") * 100).round(2).to_string())

    print("\n### Cost with W (kFLOP per decision), all channels")
    for rate in sorted(w.rate_hz.unique()):
        d = w[(w.rate_hz == rate) & (w.channels.isin([8, 12])) & (w.bits == 16)]
        ex = d.groupby("patient").extraction_flops.mean() / 1000
        print(f"{rate} Hz: extraction " + ", ".join(f"{p} {v:.0f}" for p, v in ex.items()))
    for model in G.MODELS:
        d = w[w.model == model]
        print(f"{model}: model ops {d.model_ops.min() / 1000:.1f} to {d.model_ops.max() / 1000:.1f} k over the grid")
    ref = w[(w.rate_hz == 1000) & (w.channels == 8) & (w.bits == 16)]
    print("\n### Reference stream (1 kHz x 8 ch x 16 bit) with W: model operations (k) and total kFLOP per decision, range over participants")
    for model in G.MODELS:
        d = ref[ref.model == model]
        print(f"{model}: model ops {d.model_ops.min() / 1000:.1f} to {d.model_ops.max() / 1000:.1f} k; total {d.total_flops.min() / 1000:.0f} to {d.total_flops.max() / 1000:.0f} kFLOP")
    tot = ref.pivot_table(index="patient", columns="model", values="total_flops")
    print("SVM total / LDA total: " + ", ".join(f"{p} {v:.1f}" for p, v in (tot["SVM"] / tot["LDA"]).items()))
    acc = ref.pivot_table(index="patient", columns="model", values="accuracy") * 100
    print("SVM minus LDA accuracy (points): " + ", ".join(f"{p} {v:+.2f}" for p, v in (acc["SVM"] - acc["LDA"]).items()))


if __name__ == "__main__":
    main()
