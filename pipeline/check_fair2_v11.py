"""V11a and V11d of docs/fair_followup_declared_protocol.md for the classical follow-up run (cache/grid/fair2/classical/ against cache/grid/fair/classical/).
V11a: variant 'none' with hard_k1 equals the earlier raw per-window predictions for all 84 (cell, model) pairs, the inner hard-vote accuracies equal the earlier inner accuracies, and the nested hard-k choice
equals the earlier chosen k. V11d: shapes, folds and labels equal the earlier files, probabilities are valid, every variant is present. Log: cache/grid/fair2/check_v11ad.log"""
import json
import sys

import numpy as np

import run_fair_classical as R1
import run_fair_classical2 as R2
from fair_history import CANDIDATES, STACK_VARIANTS
from fair_vote import K_GRID

LOG = R2.OUT.parent / "check_v11ad.log"


def main():
    lines = []
    winners = json.load(open(R1.WIN))
    n = bad = 0
    n_sv_eq = 0
    for model in R1.MODELS:
        for pf, cfgs in winners.items():
            part, fset = pf.split("|")
            for rate, nch, bits in cfgs:
                n += 1
                tag = R1.key_of(part, rate, nch, bits, fset, model)
                old = np.load(R1.task_path(part, rate, nch, bits, fset, model), allow_pickle=False)
                new = np.load(R2.task_path(part, rate, nch, bits, fset, model), allow_pickle=False)
                problems = []
                labels = list(old["labels"])
                if list(new["labels"]) != labels:
                    problems.append("labels differ")
                if not (old["folds"] == new["folds"]).all():
                    problems.append("folds differ")
                c2i = {c: i for i, c in enumerate(labels)}
                old_hard = np.array([c2i[c] for c in old["pooled"]])
                if not (old_hard == new["hard__none"]).all():
                    problems.append(f"raw predictions differ in {(old_hard != new['hard__none']).sum()} rows")
                inner = new["inner__none"]
                if inner.shape != (5, len(CANDIDATES)):
                    problems.append(f"inner shape {inner.shape}")
                elif not np.array_equal(inner[:, :6], old["inner_acc"]):
                    problems.append("inner hard accuracies differ from the earlier ones")
                else:
                    k_new = [0 if K_GRID[int(np.argmax(r[:6]))] is None else K_GRID[int(np.argmax(r[:6]))] for r in inner]
                    if k_new != list(old["chosen_k"]):
                        problems.append(f"chosen k differs {k_new} vs {list(old['chosen_k'])}")
                for v in STACK_VARIANTS:
                    h, P = new[f"hard__{v}"], new[f"proba__{v}"]
                    if h.shape[0] != len(old["folds"]) or P.shape != (len(h), len(labels)):
                        problems.append(f"{v}: shape")
                    elif not np.allclose(P.sum(axis=1), 1.0, atol=1e-4) or P.min() < -1e-9:
                        problems.append(f"{v}: probabilities not valid")
                    elif model != "SVM" and not (P.argmax(axis=1) == h).all() and (P.argmax(axis=1) != h).mean() > 0.002:
                        problems.append(f"{v}: argmax of probabilities differs from predict() in {(P.argmax(axis=1) != h).mean():.4f} of rows")
                    if new[f"inner__{v}"].shape != (5, len(CANDIDATES)):
                        problems.append(f"{v}: inner shape")
                if problems:
                    bad += 1
                    lines.append(f"FAIL {tag}: " + "; ".join(problems))
    lines.append(f"V11a/V11d: {n - bad} of {n} (cell, model) pairs pass" + ("" if bad == 0 else f", {bad} FAIL"))
    LOG.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
