"""EXPLORATORY summary of docs/rpni_vs_muscle_declared_protocol.md (tests 1-4). Reads cache/rpni_vs_muscle/{evals.json, signal_quality.json}
and cache/channel_importance/evals.json (reproduction check); writes cache/rpni_vs_muscle/{summary.md, summary.json, fig_rpni_vs_muscle.pdf/png}.
Usage: python -W ignore analyse_rpni_vs_muscle.py"""
import itertools
import json

import numpy as np

from build_wide_tables import PARTS, channels_of
from config import CACHE_DIR
from run_channel_importance import RPNI, skey
from run_rpni_vs_muscle import MODELS, T30, TABLES, muscle_subsets

D = CACHE_DIR / "rpni_vs_muscle"
SHORT = {"P1_combined": "P1", "P2_combined": "P2", "Wallace_S1": "S1", "Wallace_S2": "S2"}
MOVE = {2: "Thumb flex", 3: "Index flex", 4: "Middle flex", 5: "Ring flex", 6: "Small flex", 7: "Fist", 8: "Pinch", 9: "Point",
        10: "Finger ext", 11: "Wrist flex", 13: "Abduction", 14: "Adduction", 18: "Thumb opp"}
GROUPS = {"thumb/finger flexion": [2, 3, 4, 5, 6], "grasps": [7, 8, 9, 18], "extension, wrist, hand": [10, 11, 13, 14]}
rng = np.random.default_rng(0)


def perm_test(vals, is_rp):
    """Exact within-participant permutation test of mean(RPNI) - mean(muscle); returns (diff, p two-sided)."""
    vals, is_rp = np.asarray(vals, float), np.asarray(is_rp, bool)
    k, n = is_rp.sum(), len(vals)
    obs = vals[is_rp].mean() - vals[~is_rp].mean()
    null = []
    for c in itertools.combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(c)] = True
        null.append(vals[m].mean() - vals[~m].mean())
    null = np.array(null)
    return float(obs), float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def combined_perm(per_part, n=200000):
    """Sum over participants of mean(RPNI) - mean(muscle), Monte-Carlo null with labels permuted within each participant."""
    obs = sum(v[np.asarray(r, bool)].mean() - v[~np.asarray(r, bool)].mean() for v, r in per_part)
    tot = np.zeros(n)
    for v, r in per_part:
        v, r = np.asarray(v, float), np.asarray(r, bool)
        k = r.sum()
        idx = np.argsort(rng.random((n, len(v))), axis=1)[:, :k]
        s_r = v[idx].mean(axis=1)
        tot += s_r - (v.sum() - v[idx].sum(axis=1)) / (len(v) - k)
    return float(obs), float(np.mean(np.abs(tot) >= abs(obs) - 1e-12))


def main():
    E = json.load(open(D / "evals.json"))
    CI = json.load(open(CACHE_DIR / "channel_importance" / "evals.json"))

    def acc(part, table, subset, model=None):
        ms = [model] if model else MODELS
        return 100 * float(np.mean([E[f"{part}|{table}|{m}|{skey(part, subset)}"]["accuracy"] for m in ms]))

    def recall(part, ch):
        r = np.mean([E[f"{part}|{T30}|{m}|{ch}"]["recall"] for m in MODELS], axis=0) * 100
        return dict(zip(E[f"{part}|{T30}|LDA|{ch}"]["labels"], r))

    L, S = [], {}
    # reproduction check
    bad = []
    for part in PARTS:
        for sub in [[c] for c in channels_of(part)] + [RPNI[part]]:
            for m in MODELS:
                a, b = E[f"{part}|{T30}|{m}|{skey(part, sub)}"]["accuracy"], CI[f"{part}|{m}|{skey(part, sub)}"]["accuracy"]
                if a != b:
                    bad.append((part, m, sub, a, b))
    L.append("# EXPLORATORY: RPNI vs muscle channels (tests 1-4)\n")
    L.append(f"Reproduction check (single channels and RPNI-only at 30 kHz vs channel_importance/evals.json): "
             f"{'all identical' if not bad else f'{len(bad)} DIFFER: {bad[:3]}'}.\n")

    # ---------- Test 1
    L.append("## Test 1. RPNI channels only vs every equal-size set of muscle channels only (30 kHz, mean of LDA/SVM/RF, %)\n")
    L.append("| | RPNIs | RPNI-only | muscle-only mean | muscle-only range | RPNI rank | verdict | per model RPNI vs muscle mean (LDA, SVM, RF) |")
    L.append("|---|---|---|---|---|---|---|---|")
    S["test1"] = {}
    for part in PARTS:
        rp = acc(part, T30, RPNI[part])
        mu = np.array([acc(part, T30, s) for s in muscle_subsets(part)])
        rank = int(np.sum(mu > rp)) + 1
        verdict = "more" if rp > mu.max() else ("less" if rp < mu.min() else "within range")
        pm = ", ".join(f"{acc(part, T30, RPNI[part], m):.1f} vs {np.mean([acc(part, T30, s, m) for s in muscle_subsets(part)]):.1f}" for m in MODELS)
        L.append(f"| {SHORT[part]} | {len(RPNI[part])} | {rp:.1f} | {mu.mean():.1f} | {mu.min():.1f}-{mu.max():.1f} | {rank} of {len(mu) + 1} | {verdict} | {pm} |")
        S["test1"][part] = dict(rpni=rp, muscle_mean=float(mu.mean()), muscle_min=float(mu.min()), muscle_max=float(mu.max()), rank=rank,
                                n=len(mu) + 1, verdict=verdict)
    L.append("")

    # ---------- Test 2
    L.append("## Test 2. Which movements each channel separates (30 kHz, channel alone, recall %, mean of LDA/SVM/RF)\n")
    S["test2"] = {}
    spec_count = {g: 0 for g in GROUPS}
    for part in PARTS:
        ch = channels_of(part)
        R = {c: recall(part, c) for c in ch}
        cues = sorted(R[ch[0]])
        L.append(f"### {SHORT[part]}\n")
        L.append("| Movement | best channel | type | recall | best RPNI | best muscle | RPNI - muscle | mean RPNI | mean muscle |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        rows = {}
        for q in cues:
            best = max(ch, key=lambda c: R[c][q])
            br = max(RPNI[part], key=lambda c: R[c][q])
            bm = max([c for c in ch if c not in RPNI[part]], key=lambda c: R[c][q])
            d = R[br][q] - R[bm][q]
            mr = np.mean([R[c][q] for c in RPNI[part]])
            mm = np.mean([R[c][q] for c in ch if c not in RPNI[part]])
            rows[q] = dict(best=best, best_is_rpni=best in RPNI[part], best_rpni=br, best_muscle=bm, diff=float(d), mean_rpni=float(mr), mean_muscle=float(mm))
            L.append(f"| {MOVE[q]} | {best} | {'RPNI' if best in RPNI[part] else 'muscle'} | {R[best][q]:.0f} | {br} {R[br][q]:.0f} | {bm} {R[bm][q]:.0f} | "
                     f"{d:+.0f} | {mr:.0f} | {mm:.0f} |")
        L.append("")
        grp = {}
        for g, qs in GROUPS.items():
            qs = [q for q in qs if q in rows]
            wins = sum(rows[q]["diff"] > 5 for q in qs)
            spec = wins > len(qs) / 2
            spec_count[g] += spec
            grp[g] = dict(n=len(qs), rpni_wins_gt5=wins, mean_diff=float(np.mean([rows[q]["diff"] for q in qs])),
                          mean_rpni=float(np.mean([rows[q]["mean_rpni"] for q in qs])), mean_muscle=float(np.mean([rows[q]["mean_muscle"] for q in qs])),
                          specialises=bool(spec))
        nbest = sum(r["best_is_rpni"] for r in rows.values())
        L.append(f"Best single channel is an RPNI for {nbest} of {len(rows)} movements (RPNIs are {len(RPNI[part])} of {len(ch)} channels).\n")
        L.append("| Movement group | movements | best RPNI beats best muscle by >5 | mean (best RPNI - best muscle) | mean RPNI | mean muscle | specialises? |")
        L.append("|---|---|---|---|---|---|---|")
        for g, v in grp.items():
            L.append(f"| {g} | {v['n']} | {v['rpni_wins_gt5']} | {v['mean_diff']:+.1f} | {v['mean_rpni']:.1f} | {v['mean_muscle']:.1f} | {'yes' if v['specialises'] else 'no'} |")
        L.append("")
        S["test2"][part] = dict(movements=rows, groups=grp, n_best_rpni=nbest)
    L.append("Consistency across participants (specialisation in >= 3 of 4): " +
             "; ".join(f"{g}: {n} of 4 ({'consistent' if n >= 3 else 'not consistent'})" for g, n in spec_count.items()) + "\n")
    S["test2"]["consistency"] = spec_count

    # ---------- Test 3
    sq_path = D / "signal_quality.json"
    if sq_path.exists():
        Q = json.load(open(sq_path))
        f = np.array(Q["freqs"])
        L.append("## Test 3. Signal quality from the raw 30 kHz recordings\n")
        L.append("Movement-to-rest ratio = 20 log10(median movement-segment RMS / median rest-segment RMS). Spectral measures are of the movement-related "
                 "excess power (movement PSD minus rest PSD, 20 Hz-7 kHz).\n")
        S["test3"] = {}
        metrics = ["inband_db", "broad_db", "best_move_db", "n_moves_gt6db", "mdf", "mnf", "f95", "frac_above_500"]
        perpart = {m: [] for m in metrics}
        for part in PARTS:
            ch = channels_of(part)
            srcs = [k for k in Q["sources"] if k.split("|")[0] == part]
            L.append(f"### {SHORT[part]}\n")
            L.append("| Channel | type | ratio in band (dB) | ratio broadband (dB) | best movement (dB) | movements > 6 dB | median freq (Hz) | mean freq (Hz) | 95 % power below (Hz) | excess power above 500 Hz |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            vals = {m: [] for m in metrics}
            for c in ch:
                segs = [s for k in srcs for s in Q["sources"][k][c]["segments"]]
                rest_in = np.median([s[1] for s in segs if s[0] == 1])
                rest_bb = np.median([s[2] for s in segs if s[0] == 1])
                mv = [s for s in segs if s[0] != 1]
                inband = 20 * np.log10(np.median([s[1] for s in mv]) / rest_in)
                broad = 20 * np.log10(np.median([s[2] for s in mv]) / rest_bb)
                per_q = {q: 20 * np.log10(np.median([s[1] for s in mv if s[0] == q]) / rest_in) for q in sorted({s[0] for s in mv})}
                pm = sum(np.array(Q["sources"][k][c]["psd_move_sum"]) for k in srcs) / sum(Q["sources"][k][c]["n_move"] for k in srcs)
                pr = sum(np.array(Q["sources"][k][c]["psd_rest_sum"]) for k in srcs) / sum(Q["sources"][k][c]["n_rest"] for k in srcs)
                ex = np.clip(pm - pr, 0, None)
                band = (f >= 20) & (f <= 7000)
                fe, pe = f[band], ex[band]
                cum = np.cumsum(pe) / pe.sum()
                row = dict(inband_db=inband, broad_db=broad, best_move_db=max(per_q.values()), n_moves_gt6db=sum(v > 6 for v in per_q.values()),
                           mdf=float(fe[np.searchsorted(cum, 0.5)]), mnf=float((fe * pe).sum() / pe.sum()), f95=float(fe[np.searchsorted(cum, 0.95)]),
                           frac_above_500=float(pe[fe > 500].sum() / pe.sum()))
                for m in metrics:
                    vals[m].append(float(row[m]))
                L.append(f"| {c} | {'RPNI' if c in RPNI[part] else 'muscle'} | {row['inband_db']:.1f} | {row['broad_db']:.1f} | {row['best_move_db']:.1f} | "
                         f"{row['n_moves_gt6db']} of {len(per_q)} | {row['mdf']:.0f} | {row['mnf']:.0f} | {row['f95']:.0f} | {100 * row['frac_above_500']:.0f} % |")
            is_rp = [c in RPNI[part] for c in ch]
            L.append("")
            L.append("| Measure | mean RPNI | mean muscle | RPNI - muscle | permutation p |")
            L.append("|---|---|---|---|---|")
            S["test3"][part] = {"channels": {c: {m: vals[m][i] for m in metrics} for i, c in enumerate(ch)}, "compare": {}}
            for m in metrics:
                v = np.array(vals[m])
                dlt, p = perm_test(v, is_rp)
                perpart[m].append((v, is_rp))
                S["test3"][part]["compare"][m] = dict(rpni=float(v[is_rp].mean()), muscle=float(v[~np.array(is_rp)].mean()), diff=dlt, p=p)
                L.append(f"| {m} | {v[is_rp].mean():.2f} | {v[~np.array(is_rp)].mean():.2f} | {dlt:+.2f} | {p:.2f} |")
            L.append("")
        L.append("Combined over participants (sum of per-participant RPNI - muscle differences, labels permuted within participants):\n")
        L.append("| Measure | sum of differences | p | participants with RPNI > muscle |")
        L.append("|---|---|---|---|")
        S["test3"]["combined"] = {}
        for m in metrics:
            obs, p = combined_perm(perpart[m])
            npos = sum(v[np.asarray(r)].mean() > v[~np.asarray(r)].mean() for v, r in perpart[m])
            S["test3"]["combined"][m] = dict(sum_diff=obs, p=p, n_rpni_higher=int(npos))
            L.append(f"| {m} | {obs:+.2f} | {p:.3f} | {npos} of 4 |")
        L.append("")
    else:
        L.append("## Test 3: signal_quality.json not found (not run yet).\n")

    # ---------- Test 4
    L.append("## Test 4. Single-channel accuracy under compression (mean of LDA/SVM/RF, %)\n")
    have = all(f"{p}|{t}|{m}|{c}" in E for p in PARTS for t in TABLES for m in MODELS for c in channels_of(p))
    if not have:
        L.append("Incomplete: not all grid-table cells are present yet.\n")
    else:
        losses = {"rate 1k-250 (16 bit)": ("1000Hz_16bit", "250Hz_16bit"), "rate 1k-500 (16 bit)": ("1000Hz_16bit", "500Hz_16bit"),
                  "bits 16-8 (1 kHz)": ("1000Hz_16bit", "1000Hz_8bit"), "combined 30k/16 - 250/8": (T30, "250Hz_8bit")}
        S["test4"] = {}
        per_loss = {k: [] for k in losses}
        per_rel = {k: [] for k in losses}
        for part in PARTS:
            ch = channels_of(part)
            ncls = len(E[f"{part}|{T30}|LDA|{ch[0]}"]["labels"])
            chance = 100 / ncls
            L.append(f"### {SHORT[part]} (chance {chance:.1f} %)\n")
            show = [T30, "10000Hz_16bit", "1000Hz_16bit", "1000Hz_8bit", "500Hz_16bit", "500Hz_8bit", "250Hz_16bit", "250Hz_8bit"]
            L.append("| Channel | type | " + " | ".join(show) + " |")
            L.append("|---|---|" + "---|" * len(show))
            A = {c: {t: acc(part, t, [c]) for t in TABLES} for c in ch}
            for c in ch:
                L.append(f"| {c} | {'RPNI' if c in RPNI[part] else 'muscle'} | " + " | ".join(f"{A[c][t]:.1f}" for t in show) + " |")
            L.append("")
            is_rp = np.array([c in RPNI[part] for c in ch])
            L.append("| Loss (points) | mean RPNI | mean muscle | RPNI - muscle | p | relative loss RPNI | relative loss muscle | RPNI - muscle (relative) | p |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            S["test4"][part] = {"acc": A, "loss": {}}
            for k, (hi, lo) in losses.items():
                v = np.array([A[c][hi] - A[c][lo] for c in ch])
                rel = np.array([(A[c][hi] - A[c][lo]) / max(A[c][hi] - chance, 1e-9) for c in ch])
                d1, p1 = perm_test(v, is_rp)
                d2, p2 = perm_test(rel, is_rp)
                per_loss[k].append((v, is_rp))
                per_rel[k].append((rel, is_rp))
                S["test4"][part]["loss"][k] = dict(rpni=float(v[is_rp].mean()), muscle=float(v[~is_rp].mean()), diff=d1, p=p1,
                                                   rel_rpni=float(rel[is_rp].mean()), rel_muscle=float(rel[~is_rp].mean()), rel_diff=d2, rel_p=p2)
                L.append(f"| {k} | {v[is_rp].mean():.1f} | {v[~is_rp].mean():.1f} | {d1:+.1f} | {p1:.2f} | {rel[is_rp].mean():.2f} | {rel[~is_rp].mean():.2f} | {d2:+.2f} | {p2:.2f} |")
            L.append("")
        L.append("Rule check (RPNI loss > muscle loss by more than 2 points in >= 3 of 4 participants) and combined test:\n")
        L.append("| Loss | participants RPNI - muscle > 2 | verdict | sum of differences | p | sum (relative) | p (relative) |")
        L.append("|---|---|---|---|---|---|---|")
        S["test4"]["rule"] = {}
        for k in losses:
            n2 = sum(S["test4"][p]["loss"][k]["diff"] > 2 for p in PARTS)
            obs, p = combined_perm(per_loss[k])
            obs2, p2 = combined_perm(per_rel[k])
            verdict = "finer detail" if n2 >= 3 else "not shown"
            S["test4"]["rule"][k] = dict(n_gt2=int(n2), verdict=verdict, sum_diff=obs, p=p, sum_rel=obs2, p_rel=p2)
            L.append(f"| {k} | {n2} of 4 | {verdict} | {obs:+.1f} | {p:.3f} | {obs2:+.2f} | {p2:.3f} |")
        L.append("")
        make_figure(S, E, acc)

    (D / "summary.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(S, open(D / "summary.json", "w"), indent=1, default=float)
    print("\n".join(L))


def make_figure(S, E, acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RP, MU = "#1f5fa8", "#8c8c8c"
    rates = [250, 500, 1000, 3000, 10000]
    fig = plt.figure(figsize=(10, 4.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1], hspace=0.45, wspace=0.32)
    ax = fig.add_subplot(gs[:, 0])
    for i, part in enumerate(PARTS):
        t = S["test1"][part]
        mu = [acc(part, T30, s) for s in muscle_subsets(part)]
        ax.scatter(np.full(len(mu), i) + rng.uniform(-0.12, 0.12, len(mu)), mu, s=14, color=MU, alpha=0.7, zorder=2,
                   label="muscle channels only (every set)" if i == 0 else None)
        ax.scatter([i], [t["rpni"]], s=60, marker="D", color=RP, zorder=3, label="RPNI channels only" if i == 0 else None)
    ax.set_xticks(range(4), [f"{SHORT[p]}\n({len(RPNI[p])} ch)" for p in PARTS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(55, 95)
    ax.set_title("a  Equal channel count (30 kHz)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    x = np.arange(len(rates))
    for j, part in enumerate(PARTS):
        a = fig.add_subplot(gs[j // 2, 1 + j % 2])
        ch = channels_of(part)
        for typ, col in (("muscle", MU), ("RPNI", RP)):
            cs = [c for c in ch if (c in RPNI[part]) == (typ == "RPNI")]
            Y = np.array([[acc(part, f"{r}Hz_16bit", [c]) for r in rates] for c in cs])
            for y in Y:
                a.plot(x, y, color=col, lw=0.6, alpha=0.35)
            a.plot(x, Y.mean(axis=0), color=col, lw=2.2, marker="o", ms=3.5, label=f"{typ} (mean)")
        a.set_xticks(x, ["250", "500", "1k", "3k", "10k"], fontsize=8)
        a.tick_params(axis="y", labelsize=8)
        a.set_ylim(10, 80)
        a.set_title(("b  " if j == 0 else "") + SHORT[part], loc="left", fontsize=10)
        a.grid(axis="y", color="#e6e6e6", lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
        if j % 2 == 0:
            a.set_ylabel("Single channel (%)", fontsize=8)
        if j >= 2:
            a.set_xlabel("Sample rate (Hz), 16 bit", fontsize=8)
        if j == 1:
            a.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(D / "fig_rpni_vs_muscle.pdf", bbox_inches="tight")
    fig.savefig(D / "fig_rpni_vs_muscle.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
