"""Declared selection rule B3 (docs/fair_followup_declared_protocol.md) and, for --round 2, amendment A2, applied to the pilot: rank the random configurations by the mean validation accuracy at the loss-optimal
epoch over the 40 (cell, fold) pilot pairs (ties by lower mean validation loss); final candidate set = base, lr3, reg + the top 6 random configurations; inner scheme for the final run = 3-fold unless the
projected time (mean seconds per pilot run of the 9 selected configurations x 9 configs x 3 inner folds x 140 cell-folds x 1.2 / workers) exceeds 3.5 h, then the single old split.
Round 1: the 32 random configurations of lstm_tuning.random_configs -> cache/grid/fair2/lstm_candidates.json, lstm_pilot_summary.md.
Round 2: the 32 new configurations of lstm_tuning.random_configs_r2 plus the 6 round-1 carry-overs (their round-1 pilot numbers are reused) -> lstm_candidates_r2.json, lstm_pilot_summary_r2.md.
Uses inner-validation numbers only. Usage: python -W ignore select_lstm_candidates.py [--round 1|2] [--workers N]"""
import json
import sys
from collections import defaultdict

import numpy as np

import lstm_tuning as T
from config import CACHE_DIR
from run_lstm_pilot import OUT, OUT_R2

FAIR2 = CACHE_DIR / "grid" / "fair2"
HOURS_LIMIT = 3.5
N_TOP = 6
CELL_FOLDS = 140


def pilot_stats(directory, names):
    """Per configuration name: mean validation accuracy / loss / best epoch, share of best epochs at the epoch limit, mean seconds per run, over the 40 (cell, fold) pairs of a pilot directory."""
    files = sorted(directory.glob("*.json"))
    assert len(files) == 40, f"expected 40 pilot task files in {directory}, found {len(files)}"
    per = defaultdict(lambda: dict(acc=[], loss=[], ep=[], sec=[]))
    for f in files:
        for name, r in json.load(open(f))["results"].items():
            if name in names:
                p = per[name]
                p["acc"].append(r["val_acc"]); p["loss"].append(r["best_val_loss"]); p["ep"].append(r["best_epoch"]); p["sec"].append(r["seconds"])
    assert set(per) == set(names) and all(len(v["acc"]) == 40 for v in per.values())
    return {n: dict(acc=float(np.mean(v["acc"])), loss=float(np.mean(v["loss"])), ep=float(np.mean(v["ep"])), at_limit=float(np.mean(np.array(v["ep"]) >= T.MAX_EPOCHS)), sec=float(np.mean(v["sec"])))
            for n, v in per.items()}


def main():
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 6
    rnd = int(args[args.index("--round") + 1]) if "--round" in args else 1
    if rnd == 1:
        cfgs, space = T.all_configs(), T.SPACE
        stat = pilot_stats(OUT, set(cfgs))
        rand = [n for n in cfgs if n.startswith("r")]
        tag, cand_file, sum_file, carry = "", "lstm_candidates.json", "lstm_pilot_summary.md", []
    else:
        new = T.random_configs_r2()
        c1 = json.load(open(FAIR2 / "lstm_candidates.json"))
        carry = [c["name"] for c in c1["final"] if c["name"].startswith("r")]                 # the 6 round-1 random configurations of the final set
        cfgs = {**T.OLD, **{c["name"]: c["cfg"] for c in c1["final"] if c["name"] in carry}, **new}
        stat = {**pilot_stats(OUT, set(carry) | set(T.OLD)), **pilot_stats(OUT_R2, set(new))}
        rand = carry + list(new)
        space = T.SPACE_R2
        tag, cand_file, sum_file = "_r2", "lstm_candidates_r2.json", "lstm_pilot_summary_r2.md"
    ranked = sorted(rand, key=lambda n: (-stat[n]["acc"], stat[n]["loss"]))
    final = list(T.OLD) + ranked[:N_TOP]
    s = float(np.mean([stat[n]["sec"] for n in final]))
    proj3 = s * 9 * 3 * CELL_FOLDS * 1.2 / workers / 3600
    proj1 = s * 9 * 1 * CELL_FOLDS * 1.2 / workers / 3600
    scheme = "3fold" if proj3 <= HOURS_LIMIT else "single"
    shown = list(T.OLD) + rand
    cand = dict(round=rnd, final=[dict(name=n, cfg=cfgs[n]) for n in final], inner_scheme=scheme, projected_hours_3fold=proj3, projected_hours_single=proj1, workers_assumed=workers,
                mean_seconds_per_run_selected=s, carry_overs=carry, new_in_top6=[n for n in ranked[:N_TOP] if n not in carry], pilot_stats={n: stat[n] for n in shown})
    json.dump(cand, open(FAIR2 / cand_file, "w"), indent=1)

    lines = [f"# LSTM pilot summary, round {rnd} (8 reference cells x 5 folds = 40 inner validations per configuration; inner-validation numbers only)", "",
             f"Final candidate set: {', '.join(final)}. Inner scheme for the final run: **{scheme}** (projected {proj3:.2f} h with the 3-fold scheme, {proj1:.2f} h with the single split, {workers} workers; limit {HOURS_LIMIT} h).", ""]
    if rnd == 2:
        lines += [f"Round-1 carry-overs ranked together with the 32 new configurations: {', '.join(carry)}. New configurations in the top 6: {', '.join(cand['new_in_top6']) or 'none'}.", ""]
    lines += ["| config | hidden | layers | bottleneck | dropout | wd | lr | noise | batch | mean val acc % | mean val loss | mean best epoch | share at 400 | mean s/run |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in sorted(shown, key=lambda n: (-stat[n]["acc"], stat[n]["loss"])):
        c, st = cfgs[n], stat[n]
        lines.append(f"| {n}{' *' if n in final else ''} | {c['hidden']} | {c['layers']} | {c['bottleneck']} | {c['dropout']} | {c['wd']} | {c['lr']} | {c['noise']} | {c['batch']} | {100 * st['acc']:.2f} | {st['loss']:.3f} | {st['ep']:.0f} | {st['at_limit']:.2f} | {st['sec']:.1f} |")
    lines += ["", f"(* = in the final set.) Marginal mean validation accuracy (%) of the {len(rand)} random configurations by value of each setting (a rough guide, the factors are confounded):", ""]
    for k, vals in space.items():
        parts = []
        for v in vals:
            m = [stat[n]["acc"] for n in rand if cfgs[n][k] == v]
            parts.append(f"{v}: {100 * np.mean(m):.2f} (n={len(m)})" if m else f"{v}: -")
        lines.append(f"- {k}: " + "; ".join(parts))
    top = ranked[:N_TOP]
    lines += ["", "Edge check for the top 6 random configurations (value on each axis; the lowest and highest value of an axis are its edges):", ""]
    for k, vals in space.items():
        lines.append(f"- {k} (edges {vals[0]} / {vals[-1]}): " + ", ".join(str(cfgs[n][k]) for n in top))
    open(FAIR2 / sum_file, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
