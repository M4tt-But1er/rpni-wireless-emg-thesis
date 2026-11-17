"""MACs/FLOPs (exact, platform-independent operation counts) and a 3-part
latency breakdown (signal capture time + feature-extraction time + model
inference time) for every saved movement-only model.

Why 3 parts, not just model-inference time (which is all the earlier
benchmark_deployment.py measured): a live system can't make a prediction the
instant it wants one -- it first has to wait for a full 200ms window of
signal to actually arrive (capture time), then compute that window's
features (feature-extraction time), and only then run the trained model
(inference time). For every model in this pipeline, model inference turns
out to be microseconds -- utterly dwarfed by the fixed 200ms capture time,
which matters far more for real responsiveness than shaving inference time
further ever could.

MACs = multiply-accumulate operations (one multiply + one add, the basic
unit of most ML math). Reported separately from "other operations" (e.g.
comparisons for Random Forest, which does no MACs at all; exp() calls for
RBF-kernel SVM and RBFNN, which are transcendental, not simple MACs).
"""
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch

from config import CACHE_DIR, PIPELINE_DIR, WINDOW_MS
from features import extract_channel_features, _sliding_windows, WINDOW_SAMPLES
from save_models import SAVED_MODELS_DIR, load_model
from data_io import load_raw
from preprocessing import preprocess_dataframe

PARTICIPANT_DATASETS = ["P1_combined_noRest", "P2_combined_noRest"]
SKLEARN_MODELS = ["LDA", "RandomForest", "SVM", "RBFNN"]
N_REPEATS = 2000


def load_raw_pickle(dataset_name, model_name):
    path = SAVED_MODELS_DIR / f"{dataset_name}_{model_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def complexity_lda(est, n_features):
    n_classes = est.coef_.shape[0] if est.coef_.ndim > 1 else 1
    n_classes = max(n_classes, 2)
    macs = n_features * n_classes
    return {"macs": int(macs), "other_ops": 0, "notes": "one matrix-vector product (features x classes)"}


def complexity_random_forest(est, n_features):
    n_trees = len(est.estimators_)
    depths = [t.tree_.max_depth for t in est.estimators_]
    node_counts = [t.tree_.node_count for t in est.estimators_]
    max_total_comparisons = n_trees * max(depths)
    mean_total_comparisons = n_trees * (sum(depths) / n_trees)
    return {
        "macs": 0,
        "other_ops": int(max_total_comparisons),
        "notes": f"pure threshold comparisons, no multiply-adds. {n_trees} trees, "
                 f"worst-case {max(depths)} comparisons/tree (max depth), "
                 f"typical ~{mean_total_comparisons:.0f} total (mean depth {sum(depths)/n_trees:.1f}). "
                 f"Total decision nodes across the forest: {sum(node_counts)}.",
    }


def complexity_svm(est, n_features):
    n_sv = est.support_vectors_.shape[0]
    macs = n_sv * n_features  # one dot-product-equivalent per support vector
    exp_calls = n_sv  # RBF kernel: one exp() per support vector
    return {
        "macs": int(macs), "other_ops": int(exp_calls),
        "notes": f"{n_sv} support vectors x {n_features} features for the kernel distance calc, "
                 f"plus {n_sv} exp() calls (RBF kernel, each a transcendental function -- "
                 f"more expensive per-call than a plain multiply-add on typical embedded hardware).",
    }


def complexity_rbfnn(est, n_features):
    n_centers = est.centers_.shape[0]
    n_classes = est.clf_.coef_.shape[0] if est.clf_.coef_.ndim > 1 else 1
    n_classes = max(n_classes, 2)
    macs = n_centers * n_features + n_centers * n_classes
    exp_calls = n_centers
    return {
        "macs": int(macs), "other_ops": int(exp_calls),
        "notes": f"{n_centers} centers x {n_features} features (distance calc) + "
                 f"{n_centers} x {n_classes} (logistic-regression readout), "
                 f"plus {n_centers} exp() calls (Gaussian basis functions).",
    }


def complexity_lstm(payload):
    input_size = payload["input_size"]
    hidden_size = payload["hidden_size"]
    n_classes = payload["n_classes"]
    # 4 gates (input, forget, cell candidate, output), each: W_x@x + W_h@h + b
    macs_per_timestep = 4 * hidden_size * (input_size + hidden_size)
    macs_output_layer = hidden_size * n_classes
    total_macs = macs_per_timestep + macs_output_layer
    # each gate also applies a sigmoid or tanh nonlinearity, elementwise -- not a MAC,
    # but a real, non-negligible per-timestep cost, counted separately.
    activation_calls = 4 * hidden_size + 2 * hidden_size  # 4 gates + cell-state tanh + hidden-state combine
    return {
        "macs": int(total_macs), "other_ops": int(activation_calls),
        "notes": f"per NEW timestep (hidden state carried forward, not reprocessing the "
                 f"whole sequence): 4 gates x {hidden_size} hidden units x "
                 f"({input_size} inputs + {hidden_size} hidden) + {hidden_size}x{n_classes} output layer. "
                 f"Plus ~{activation_calls} sigmoid/tanh activation calls (transcendental, not MACs).",
    }


def measure_feature_extraction_time(dataset_name, n_repeats=200):
    """Time to compute the full 192-feature superset for ONE window, across
    all 8 channels -- using real signal data, not synthetic. This is the
    step every model in this project needs before it can predict at all,
    regardless of which features it ultimately uses."""
    raw_name = "P1_1of12" if dataset_name.startswith("P1") else "P2_1of10"
    df = load_raw(raw_name)
    dfp = preprocess_dataframe(df)
    channels = [c for c in dfp.columns if c not in ("time_s", "cue")]

    # grab one real window per channel (first full window in the recording)
    # .copy() -- pywt needs a writable buffer, a raw pandas-backed slice is read-only
    windows = {ch: dfp[ch].to_numpy()[:WINDOW_SAMPLES].reshape(1, -1).copy() for ch in channels}

    # warm-up
    for ch in channels:
        extract_channel_features(windows[ch])

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        for ch in channels:
            extract_channel_features(windows[ch])
    total = time.perf_counter() - t0
    per_window_us = (total / n_repeats) * 1e6
    return per_window_us, len(channels)


def measure_sklearn_inference_latency(dataset_name, model_name, n_repeats=N_REPEATS):
    payload = load_raw_pickle(dataset_name, model_name)
    est = payload["estimator"]
    scaler = payload["scaler"]
    features = payload["feature_list"]

    if model_name == "RandomForest":
        est.n_jobs = 1  # see benchmark_deployment.py -- realistic single-core embedded number

    df = pd.read_parquet(CACHE_DIR / f"{dataset_name}.parquet")
    X = df[features].to_numpy()
    if scaler is not None:
        X = scaler.transform(X)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(X), size=n_repeats)
    samples = X[idx]

    for i in range(20):
        est.predict(samples[i:i + 1])
    t0 = time.perf_counter()
    for i in range(n_repeats):
        est.predict(samples[i:i + 1])
    total = time.perf_counter() - t0
    return (total / n_repeats) * 1e6


def measure_lstm_inference_latency(dataset_name, n_repeats=N_REPEATS):
    path = SAVED_MODELS_DIR / f"{dataset_name}_LSTM.pt"
    payload = torch.load(path, weights_only=False)
    from lstm_model import SequenceLSTM
    model = SequenceLSTM(payload["input_size"], payload["hidden_size"], payload["n_classes"])
    model.load_state_dict(payload["state_dict"])
    model.eval()

    df = pd.read_parquet(CACHE_DIR / f"{dataset_name}.parquet")
    features = payload["feature_list"]
    x_raw = df[features].to_numpy()[:1]
    x_scaled = ((x_raw - payload["scale_mean"]) / payload["scale_std"]).astype(np.float32)
    x_single = torch.from_numpy(x_scaled).unsqueeze(0)

    with torch.no_grad():
        h = None
        for _ in range(20):
            out, h = model.lstm(x_single, h)
            _ = model.fc(out)
        h = None
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            out, h = model.lstm(x_single, h)
            _ = model.fc(out)
        total = time.perf_counter() - t0
    return (total / n_repeats) * 1e6


def main():
    results = {}
    for dataset_name in PARTICIPANT_DATASETS:
        results[dataset_name] = {}
        feat_extract_us, n_channels = measure_feature_extraction_time(dataset_name)
        print(f"[{dataset_name}] feature extraction (192 features x {n_channels} channels): "
              f"{feat_extract_us:.1f} us/window")

        for model_name in SKLEARN_MODELS:
            payload = load_raw_pickle(dataset_name, model_name)
            est = payload["estimator"]
            n_features = len(payload["feature_list"])

            if model_name == "LDA":
                complexity = complexity_lda(est, n_features)
            elif model_name == "RandomForest":
                complexity = complexity_random_forest(est, n_features)
            elif model_name == "SVM":
                complexity = complexity_svm(est, n_features)
            elif model_name == "RBFNN":
                complexity = complexity_rbfnn(est, n_features)

            inference_us = measure_sklearn_inference_latency(dataset_name, model_name)
            total_latency_ms = WINDOW_MS + feat_extract_us / 1000 + inference_us / 1000

            print(f"  {model_name}: {complexity['macs']} MACs, {complexity['other_ops']} other ops, "
                  f"inference={inference_us:.2f}us, total_latency={total_latency_ms:.2f}ms")

            results[dataset_name][model_name] = {
                "n_features": n_features,
                "macs": complexity["macs"],
                "other_ops": complexity["other_ops"],
                "complexity_notes": complexity["notes"],
                "capture_time_ms": WINDOW_MS,
                "feature_extraction_us": round(feat_extract_us, 2),
                "model_inference_us": round(inference_us, 3),
                "total_latency_ms": round(total_latency_ms, 3),
            }

        lstm_payload = torch.load(SAVED_MODELS_DIR / f"{dataset_name}_LSTM.pt", weights_only=False)
        complexity = complexity_lstm(lstm_payload)
        inference_us = measure_lstm_inference_latency(dataset_name)
        total_latency_ms = WINDOW_MS + feat_extract_us / 1000 + inference_us / 1000
        print(f"  LSTM: {complexity['macs']} MACs, {complexity['other_ops']} other ops, "
              f"inference={inference_us:.2f}us, total_latency={total_latency_ms:.2f}ms")
        results[dataset_name]["LSTM"] = {
            "n_features": lstm_payload["input_size"],
            "macs": complexity["macs"],
            "other_ops": complexity["other_ops"],
            "complexity_notes": complexity["notes"],
            "capture_time_ms": WINDOW_MS,
            "feature_extraction_us": round(feat_extract_us, 2),
            "model_inference_us": round(inference_us, 3),
            "total_latency_ms": round(total_latency_ms, 3),
        }

    out_path = CACHE_DIR / "complexity_latency_norest.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print("\nsaved", out_path)


if __name__ == "__main__":
    main()
