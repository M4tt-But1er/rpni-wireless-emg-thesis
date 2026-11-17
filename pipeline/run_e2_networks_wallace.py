"""Fills the empty S1/S2 cells of the E2 table (engineered cascade set, 1 kHz stream; appendix table tab:res_e1): tuned RBFNN,
LSTM (headline setting) and tuned LSTM on Wallace S1 and S2, with the code that produced the P1/P2 values, imported unchanged:

  RBFNN, tuned : run_rbfnn_tuning.tune_rbfnn, nested per outer fold, on the Stage-3 feature set each fold already used for the
                 default RBFNN (cache/cascade_baseline_wallace_lda_rbfnn.json), exactly as run_rbfnn_tuning.main does for P1/P2.
  LSTM         : lstm_model.evaluate_lstm_with_folds_with_predictions on the Stage-2 reduced set (cache/feature_selection_results_wallace.json),
                 40 epochs, 64 units, batch 16, exactly as run_models_norest.main does for P1/P2.
  LSTM, tuned  : run_lstm_training_check.part2 (configs base / lr3 / reg, early stopping on an inner 20 % split), same Stage-2 set.

Same folds (models.compute_cv_folds), same metrics (run_models_norest.compute_metrics). Reproduction check: the same three runs on P1,
with P1's own inputs, must give the published 96.76 / 94.58 / 97.40 %.
Each tuned-RBFNN outer fold, each LSTM and each tuned LSTM is one task, run in parallel processes (one thread each) and saved to
cache/e2_networks_wallace/<task>.json (resumable); the pooled results go to the NEW file cache/e2_networks_wallace.json keyed
"<participant>|<method>". Nothing existing is changed.
Usage: python -W ignore -u run_e2_networks_wallace.py [--workers N]"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import run_lstm_training_check as L
from config import CACHE_DIR
from lstm_model import evaluate_lstm_with_folds_with_predictions
from models import RBFNetClassifier, compute_cv_folds
from run_models_norest import compute_metrics
from run_rbfnn_tuning import check_equivalence, tune_rbfnn

warnings.filterwarnings("ignore")
OUT = CACHE_DIR / "e2_networks_wallace.json"
TASK_DIR = CACHE_DIR / "e2_networks_wallace"                      # one file per task (resumable)
CASCADE = json.load(open(CACHE_DIR / "cascade_baseline_wallace_lda_rbfnn.json"))
FS_W = json.load(open(CACHE_DIR / "feature_selection_results_wallace.json"))
FS_P = json.load(open(CACHE_DIR / "feature_selection_results_norest.json"))
UNTUNED_P = json.load(open(CACHE_DIR / "model_results_norest.json"))
PUBLISHED_P1 = {"RBFNN_tuned": 0.9676, "LSTM": 0.9458, "LSTM_tuned": 0.9740}


def inputs(part):
    """(table, Stage-2 reduced set, per-fold Stage-3 sets of the default RBFNN)."""
    if part.startswith("Wallace"):
        df = pd.read_parquet(CACHE_DIR / f"{part}_noRest.parquet")
        return df, FS_W[part]["reduced_features"], CASCADE[part]["RBFNN"]["nested_per_fold_features"]
    name = f"{part}_noRest"
    df = pd.read_parquet(CACHE_DIR / f"{name}.parquet")
    return df, FS_P[name]["reduced_features"], UNTUNED_P[name]["RBFNN"]["nested_per_fold_features"]


def rbfnn_fold(part, fold):
    """One outer fold of run_rbfnn_tuning.main (same steps, in a separate process)."""
    df, _, per_fold_features = inputs(part)
    y, groups = df["cue"].to_numpy(), df["segment_id"].to_numpy()
    fold_ids = compute_cv_folds(df)
    if fold == 0:
        check_equivalence(df[per_fold_features[0]].to_numpy(), y, groups)
    tr, te = fold_ids != fold, fold_ids == fold
    X = df[per_fold_features[fold]].to_numpy()
    (n_c, s, C), inner, _ = tune_rbfnn(X[tr], y[tr], groups[tr])
    scaler = StandardScaler().fit(X[tr])
    clf = RBFNetClassifier(n_centers=n_c, C=C, sigma_scale=s).fit(scaler.transform(X[tr]), y[tr])
    pred = clf.predict(scaler.transform(X[te]))
    acc = float((pred == y[te]).mean())
    print(f"  {part} RBFNN_tuned fold {fold}: {len(per_fold_features[fold])} features, centres={n_c}, width x{s}, C={C}, inner={inner:.4f}, outer={acc:.4f}", flush=True)
    return dict(n_centers=n_c, sigma_scale=s, C=C, inner_cv_acc=inner, outer_test_acc=acc, n_features=len(per_fold_features[fold]),
                test_index=np.flatnonzero(te).tolist(), pred=pred.tolist())


def rbfnn_assemble(part, folds):
    df, _, _ = inputs(part)
    y = df["cue"].to_numpy()
    y_pred = np.empty_like(y)
    for f in folds:
        y_pred[np.array(f["test_index"])] = np.array(f["pred"], dtype=y.dtype)
    m = compute_metrics(y, y_pred, sorted(df["cue"].unique().tolist()))
    per_fold = [{k: v for k, v in f.items() if k not in ("test_index", "pred")} for f in folds]
    return dict(accuracy=m["accuracy"], weighted_f1=m["weighted_f1"], cohen_kappa=m["cohen_kappa"], per_fold=per_fold)


def lstm_headline(part):
    """LSTM block of run_models_norest.main for one participant."""
    df, reduced, _ = inputs(part)
    fold_ids = compute_cv_folds(df)
    accs, yt, yp = evaluate_lstm_with_folds_with_predictions(df, reduced, fold_ids, epochs=40, hidden_size=64, batch_size=16)
    m = compute_metrics(yt, yp, sorted(df["cue"].unique().tolist()))
    return dict(accuracy=m["accuracy"], weighted_f1=m["weighted_f1"], cohen_kappa=m["cohen_kappa"], nested_fold_accs=[float(a) for a in accs])


def lstm_tuned(part):
    """run_lstm_training_check.part2 for one participant."""
    df, reduced, _ = inputs(part)
    r = L.part2(part, loaded=L.prepare(df, reduced, compute_cv_folds(df)))
    return dict(accuracy=r["tuned_accuracy"], weighted_f1=r["tuned_weighted_f1"], cohen_kappa=r["tuned_kappa"], per_fold=r["per_fold"])


def task_file(task):
    return TASK_DIR / ("__".join(str(t) for t in task) + ".json")


def run_task(task):
    """One task in its own process, one thread (so 14 tasks share the 16 cores). Writes its own file; resumable."""
    import torch
    from threadpoolctl import threadpool_limits
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    if task_file(task).exists():
        return task, "exists"
    t0 = time.time()
    with threadpool_limits(limits=1):
        kind, part = task[0], task[1]
        r = rbfnn_fold(part, task[2]) if kind == "RBFNN_tuned" else lstm_headline(part) if kind == "LSTM" else lstm_tuned(part)
    r["seconds"] = round(time.time() - t0, 1)
    json.dump(r, open(task_file(task), "w"))
    if kind != "RBFNN_tuned":
        print(f"  == {part} {kind}: {100 * r['accuracy']:.2f}%, {r['seconds']:.0f}s", flush=True)
    return task, f"{r['seconds']:.0f}s"


PARTS = ["Wallace_S1", "Wallace_S2", "P1_combined"]


def main():
    from joblib import Parallel, delayed
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 14
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [("LSTM_tuned", p) for p in PARTS] + [("RBFNN_tuned", p, f) for p in PARTS for f in range(5)] + [("LSTM", p) for p in PARTS]
    todo = [t for t in tasks if not task_file(t).exists()]
    print(f"{len(todo)} of {len(tasks)} tasks to run on {workers} workers", flush=True)
    Parallel(n_jobs=workers, verbose=0)(delayed(run_task)(t) for t in todo)
    res = {}
    for p in PARTS:
        res[f"{p}|RBFNN_tuned"] = rbfnn_assemble(p, [json.load(open(task_file(("RBFNN_tuned", p, f)))) for f in range(5)])
        res[f"{p}|LSTM"] = json.load(open(task_file(("LSTM", p))))
        res[f"{p}|LSTM_tuned"] = json.load(open(task_file(("LSTM_tuned", p))))
    json.dump(res, open(OUT, "w"), indent=1)
    for k, r in res.items():
        part, method = k.split("|")
        extra = f" (published {100 * PUBLISHED_P1[method]:.2f}%)" if part == "P1_combined" else ""
        print(f"== {k}: accuracy {100 * r['accuracy']:.2f}%, weighted F1 {100 * r['weighted_f1']:.2f}%, kappa {r['cohen_kappa']:.3f}{extra}", flush=True)
    print("All done ->", OUT)


if __name__ == "__main__":
    main()
