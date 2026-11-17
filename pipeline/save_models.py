"""Persist trained models to disk so future requests can load a ready model
and get byte-identical results, instead of retraining from scratch every
time -- nothing in this pipeline did this before now (every earlier script
trained a fresh model, used it once for scoring/benchmarking, and discarded
it; only the resulting *numbers* were ever saved).

Saved to pipeline/saved_models/. Sklearn-style models (LDA, RandomForest,
SVM, RBFNN) are pickled directly along with their scaler (if any) and
metadata. The LSTM is saved via torch's own state_dict mechanism (the
standard way to persist a PyTorch model) plus the same metadata.

Two versions of a model are stored per model name -- see load_model()'s
`suffix` parameter -- because model.py's RBFNetClassifier and the sklearn
estimators are ordinary picklable Python objects; the only special case is
the LSTM, which needs torch's loader instead of raw pickle.
"""
import pickle

import numpy as np
import torch

from config import PIPELINE_DIR
from lstm_model import build_segment_sequences, train_lstm_final

SAVED_MODELS_DIR = PIPELINE_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(exist_ok=True)


def save_model(dataset_name, model_name, fitted_estimator, scaler, feature_list, metrics):
    """Pickle a fitted sklearn-style estimator (LDA/RandomForest/SVM/RBFNN)
    plus everything needed to use it again: which features to extract in
    what order, whether/how to scale them first, and the metrics it scored
    when it was saved (so a future run can confirm nothing silently drifted)."""
    payload = {
        "dataset": dataset_name,
        "model_name": model_name,
        "estimator": fitted_estimator,
        "scaler": scaler,
        "feature_list": feature_list,
        "metrics_at_save_time": metrics,
    }
    out_path = SAVED_MODELS_DIR / f"{dataset_name}_{model_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  saved {out_path}")
    return out_path


def save_lstm_model(dataset_name, feature_list, df, fold_ids, metrics, hidden_size=64,
                     epochs=40, batch_size=16):
    """Trains one final LSTM on the FULL dataset (no held-out fold -- the
    nested-fold LSTMs used for scoring were never meant to be the deployed
    model, same reasoning as the sklearn models' final_fit_for_deployment)
    and saves its weights + everything needed to reload and use it."""
    classes = sorted(df["cue"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    sequences = build_segment_sequences(df, feature_list)
    all_x = np.concatenate([x for _, x, _ in sequences], axis=0)
    mean = all_x.mean(axis=0)
    std = all_x.std(axis=0)
    std[std == 0] = 1.0
    scaled_sequences = [(sid, ((x - mean) / std).astype(np.float32), c) for sid, x, c in sequences]

    model = train_lstm_final(scaled_sequences, n_classes, class_to_idx, input_size=len(feature_list),
                              hidden_size=hidden_size, epochs=epochs, batch_size=batch_size)

    payload = {
        "dataset": dataset_name,
        "model_name": "LSTM",
        "state_dict": model.state_dict(),
        "input_size": len(feature_list),
        "hidden_size": hidden_size,
        "n_classes": n_classes,
        "class_to_idx": class_to_idx,
        "feature_list": feature_list,
        "scale_mean": mean,
        "scale_std": std,
        "metrics_at_save_time": metrics,
    }
    out_path = SAVED_MODELS_DIR / f"{dataset_name}_LSTM.pt"
    torch.save(payload, out_path)
    print(f"  saved {out_path}")
    return out_path


def load_model(dataset_name, model_name):
    """Loads a previously saved model, ready to call .predict(X) (sklearn-
    style) or .predict_sequence(X_seq) (LSTM) without retraining anything."""
    if model_name == "LSTM":
        path = SAVED_MODELS_DIR / f"{dataset_name}_LSTM.pt"
        payload = torch.load(path, weights_only=False)
        from lstm_model import SequenceLSTM
        model = SequenceLSTM(payload["input_size"], payload["hidden_size"], payload["n_classes"])
        model.load_state_dict(payload["state_dict"])
        model.eval()

        idx_to_class = {i: c for c, i in payload["class_to_idx"].items()}

        class LoadedLSTM:
            def predict_sequence(self, X_seq):
                """X_seq: (n_timesteps, n_features) raw feature values (NOT
                pre-scaled -- scaling is applied here using the saved
                training-time mean/std)."""
                x_scaled = ((X_seq - payload["scale_mean"]) / payload["scale_std"]).astype(np.float32)
                with torch.no_grad():
                    logits = model(torch.from_numpy(x_scaled).unsqueeze(0))
                    pred_idx = logits.argmax(dim=-1).squeeze(0).numpy()
                return np.array([idx_to_class[i] for i in pred_idx])

        loaded = LoadedLSTM()
        loaded.feature_list = payload["feature_list"]
        loaded.metrics_at_save_time = payload["metrics_at_save_time"]
        return loaded

    path = SAVED_MODELS_DIR / f"{dataset_name}_{model_name}.pkl"
    with open(path, "rb") as f:
        payload = pickle.load(f)

    class LoadedSklearnModel:
        def predict(self, X):
            """X: (n_samples, n_features) raw feature values in the exact
            order of self.feature_list -- scaling (if any) is applied here."""
            Xs = payload["scaler"].transform(X) if payload["scaler"] is not None else X
            return payload["estimator"].predict(Xs)

    loaded = LoadedSklearnModel()
    loaded.feature_list = payload["feature_list"]
    loaded.metrics_at_save_time = payload["metrics_at_save_time"]
    return loaded
