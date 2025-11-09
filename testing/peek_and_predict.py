# testing/peek_and_predict.py
import argparse, json, os
import numpy as np
import torch

from src.utils.data_loader import EEGFeatureLoader
from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP


DEFAULT_PIPELINE = "data/weights/feature_pipeline.joblib"
DEFAULT_WEIGHTS  = "data/weights/feature_mlp_pipeline.pth"
DEFAULT_CONFIG   = "data/weights/runtime_config.json"


def _infer_hidden_dims(state_dict):
    """
    Infer hidden layer sizes from a checkpoint produced by EEGFeatureMLP.
    Looks for Sequential keys: net.0 (first Linear), net.4 (second Linear).
    """
    hidden = []
    if "net.0.weight" in state_dict:            # [hidden1, input_dim]
        hidden.append(int(state_dict["net.0.weight"].shape[0]))
    if "net.4.weight" in state_dict:            # [hidden2, hidden1]
        hidden.append(int(state_dict["net.4.weight"].shape[0]))
    # Fallback if something unexpected:
    return hidden if hidden else [64, 64]


def main(args):
    # ---- 1) Load a small Left/Right dataset slice (same as training) ----
    loader = EEGFeatureLoader()
    X, y = loader.load_dataset(subjects=[1], tmin=0.0, tmax=2.0, fmin=8, fmax=30)
    n, C, T = X.shape
    uniq, cnt = np.unique(y, return_counts=True)
    cls_dist = {int(k): int(v) for k, v in zip(uniq, cnt)}

    print("\nDataset peek:")
    print(f"  Trials: {n}, Channels: {C}, Samples/window: {T}")
    print(f"  Class distribution: {cls_dist}  (0=left,1=right)")

    idx = int(np.clip(args.index, 0, n - 1))
    x = X[idx]  # (C, T)
    if args.show_raw:
        ch0_head = np.round(x[0, :5], 4).tolist()
        print(f"  Example window [first 5 samples of ch0]: {ch0_head}")

    # ---- 2) Load feature pipeline and transform one window ----
    fp = FeaturePipeline.load(args.pipeline_path)
    feat = fp.transform(x[None, ...])  # (1, feature_dim)
    feature_dim = int(feat.shape[1])

    # ---- 3) Build model to MATCH the checkpoint, then load weights ----
    sd = torch.load(args.weights_path, map_location="cpu")
    hidden_dims = _infer_hidden_dims(sd)
    print(f"\nInferred hidden_dims from checkpoint: {hidden_dims}")

    model = EEGFeatureMLP(
        input_dim=feature_dim,
        n_classes=2,
        hidden_dims=hidden_dims,  # ensures shapes match
        dropout=0.2,
    )
    model.load_state_dict(sd)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(feat).float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

    print("\nSingle-window prediction:")
    print(f"  True label: {int(y[idx])}  (0=left,1=right)")
    print(f"  Pred probs: {np.round(probs, 6).tolist()}")
    print(f"  Pred class: {pred}")

    # ---- 4) Optional: write runtime config JSON for realtime adapter ----
    if args.write_config:
        cfg = {
            "sampling_rate_hz": 250,
            "bandpass_hz": [8, 30],
            "window_seconds": 2.0,
            "window_samples": 500,
            "n_channels_expected": 22,
            "features": {
                "csp_components": 4,
                "bands": {"mu": [8, 12], "beta": [13, 30]},
                "feature_dim": feature_dim,
                "scaler": "StandardScaler"
            },
            "model": {
                "arch": "EEGFeatureMLP",
                "hidden_dims": hidden_dims,
                "dropout": 0.2,
                "n_classes": 2,
                "class_map": {"0": "left", "1": "right"},
                "weights_path": os.path.abspath(args.weights_path),
            },
            "pipeline_path": os.path.abspath(args.pipeline_path),
            "notes": "Feed a (22,500) float32 window (8–30 Hz bandpassed @250Hz) → pipeline → MLP."
        }
        os.makedirs(os.path.dirname(args.config_out), exist_ok=True)
        with open(args.config_out, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"\nRuntime config written to: {args.config_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=5, help="Which trial to inspect/predict")
    ap.add_argument("--show_raw", action="store_true", help="Print a few raw samples for ch0")
    ap.add_argument("--pipeline_path", default=DEFAULT_PIPELINE)
    ap.add_argument("--weights_path", default=DEFAULT_WEIGHTS)
    ap.add_argument("--write_config", action="store_true")
    ap.add_argument("--config_out", default=DEFAULT_CONFIG)
    args = ap.parse_args()
    main(args)
