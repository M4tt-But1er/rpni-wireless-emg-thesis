"""LSTM final nested run (docs/fair_followup_declared_protocol.md, B4). Per (cell, outer fold): every configuration of cache/grid/fair2/lstm_candidates.json is trained with early stopping on the inner
validation scheme chosen by select_lstm_candidates.py (3-fold, or the single old split) using ONLY the outer-training sequences; score = mean validation accuracy at the loss-optimal epoch (ties: lower
mean validation loss, then earlier candidate); epochs = round(mean best epoch), at least 5; the chosen configuration is retrained on all outer-training sequences with seeds 0..4 and scored on the
untouched outer fold: LSTM_best = seed 0, LSTM_best_ens5 = mean softmax of the 5. One task = (cell, outer fold) -> cache/grid/fair2/lstm/<cell>__f<k>.npz (rows of the table in sequence order, class-index
predictions single and ensemble, ensemble probabilities, chosen configuration, inner scores); resumable.
Usage: python -W ignore -u run_lstm_final.py [--workers N] [--smoke]"""
import json
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

import lstm_tuning as T
import run_grid_networks as R
import run_lstm_training_check as L
from config import CACHE_DIR
from fair_vote import segment_starts
from run_fair_classical import assert_table_order
from run_lstm_pilot import inner_split

OUT = CACHE_DIR / "grid" / "fair2" / "lstm"                          # round 1 (LSTM_r1 in the analysis)
OUT_R2 = CACHE_DIR / "grid" / "fair2" / "lstm_r2"                    # round 2 (LSTM_best in the analysis, amendment A2)
CAND = CACHE_DIR / "grid" / "fair2" / "lstm_candidates.json"
CAND_R2 = CACHE_DIR / "grid" / "fair2" / "lstm_candidates_r2.json"


def task_path(part, rate, nch, bits, fset, fold, rnd=1):
    return (OUT if rnd == 1 else OUT_R2) / (R.key_of(part, rate, nch, bits, fset, "LSTM_best").replace("|", "__") + f"__f{fold}.npz")


def run_task(part, rate, nch, bits, fset, fold, threads, rnd=1):
    warnings.filterwarnings("ignore")
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(threads)
    cand = json.load(open(CAND if rnd == 1 else CAND_R2))
    with threadpool_limits(limits=threads):
        t0 = time.time()
        t, X, names, keep, folds = R.prepare_cell(part, rate, nch, bits, fset)
        assert_table_order(t)
        df, feats, classes, c2i, by_id, seg_fold = R.lstm_loaded(t, X, names, folds)
        C, F = len(classes), len(feats)
        tr, te, _, _ = L.fold_data(by_id, seg_fold, fold)
        if cand["inner_scheme"] == "3fold":
            cues = np.array([c for _, _, c in tr])
            splits = [([tr[i] for i in a], [tr[i] for i in b]) for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(np.zeros(len(tr)), cues)]
        else:
            splits = [inner_split(tr, C)]
        scores = {}
        for c in cand["final"]:
            runs = [T.train_cfg(itr, c2i, C, F, c["cfg"], val_seqs=iva)[1] for itr, iva in splits]
            scores[c["name"]] = dict(val_acc=float(np.mean([r["val_acc"] for r in runs])), val_loss=float(np.mean([r["best_val_loss"] for r in runs])),
                                     epochs=int(max(5, round(np.mean([r["best_epoch"] for r in runs])))), best_epochs=[r["best_epoch"] for r in runs])
        order = [c["name"] for c in cand["final"]]
        chosen = min(order, key=lambda n: (-scores[n]["val_acc"], scores[n]["val_loss"], order.index(n)))
        cfg = next(c["cfg"] for c in cand["final"] if c["name"] == chosen)
        ep = scores[chosen]["epochs"]
        probas, true = [], None
        for seed in range(T.N_SEEDS):
            model, _ = T.train_cfg(tr, c2i, C, F, cfg, epochs=ep, seed=seed)
            P, true = T.predict_proba(model, te, c2i, C)
            probas.append(P)
        ens = np.mean(probas, axis=0)
        seg = t["segment_id"].to_numpy()
        b = segment_starts(seg)
        rows_of = {seg[i]: np.arange(i, j) for i, j in zip(b[:-1], b[1:])}
        rows = np.concatenate([rows_of[s] for s in seg_fold.index if seg_fold[s] == fold])
        yi = np.array([c2i[c] for c in t["cue"].to_numpy()])
        assert len(rows) == len(true) and (yi[rows] == true).all(), "labels do not match the table rows (V12b)"
        assert (folds[rows] == fold).all(), "rows do not belong to the outer fold (V12b)"
    np.savez_compressed(task_path(part, rate, nch, bits, fset, fold, rnd), rows=rows, true=true, pred_single=probas[0].argmax(1), pred_ens=ens.argmax(1), proba_ens=ens.astype(np.float32),
                        proba_single=probas[0].astype(np.float32), chosen=chosen, chosen_cfg=json.dumps(cfg), epochs=ep, scores=json.dumps(scores), n_features=F, inner_scheme=cand["inner_scheme"],
                        seconds=time.time() - t0)
    print(f"  {part} {rate}Hz {nch}ch {bits}bit set {fset} fold {fold}: chose {chosen} for {ep} epochs (val acc {scores[chosen]['val_acc']:.3f}); outer acc single {100 * (probas[0].argmax(1) == true).mean():.2f}"
          f" ens5 {100 * (ens.argmax(1) == true).mean():.2f} ({time.time() - t0:.0f}s)", flush=True)
    return fold


def all_tasks():
    w = json.load(open(R.WIN))
    return [(pf.split("|")[0], rate, nch, bits, pf.split("|")[1], f) for pf, cfgs in w.items() for rate, nch, bits in cfgs for f in range(5)]


def main():
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 1
    rnd = int(args[args.index("--round") + 1]) if "--round" in args else 1
    cand_path = CAND if rnd == 1 else CAND_R2
    if rnd == 2 and {c["name"] for c in json.load(open(CAND))["final"]} == {c["name"] for c in json.load(open(CAND_R2))["final"]}:
        print("Round-2 candidate set equals round 1's: nothing to run, the round-1 files are the round-2 result (amendment A2).")
        return
    (OUT if rnd == 1 else OUT_R2).mkdir(parents=True, exist_ok=True)
    tasks = [x for x in all_tasks() if not task_path(*x, rnd).exists()]
    if "--smoke" in args:
        tasks = tasks[:1]
    print(f"round {rnd}: {len(tasks)} tasks ({len(all_tasks())} total), {workers} worker(s), scheme {json.load(open(cand_path))['inner_scheme']}", flush=True)
    for _ in Parallel(n_jobs=workers, return_as="generator_unordered")(delayed(run_task)(*t, 1, rnd) for t in tasks):
        pass
    print("All done")


if __name__ == "__main__":
    main()
