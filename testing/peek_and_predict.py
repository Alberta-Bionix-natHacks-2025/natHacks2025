# testing/peek_and_predict.py
import argparse, json, os
import numpy as np
import torch

from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP

# 4-ch defaults (match your new trainer outputs)
DEFAULT_PIPELINE_4CH = "data/weights/feature_pipeline_4ch.joblib"
DEFAULT_WEIGHTS_4CH  = "data/weights/feature_mlp_4ch.pth"
DEFAULT_CONFIG_4CH   = "data/weights/runtime_config_4ch.json"

# legacy 22-ch defaults (if you want to use them)
DEFAULT_PIPELINE_22  = "data/weights/feature_pipeline.joblib"
DEFAULT_WEIGHTS_22   = "data/weights/feature_mlp_pipeline.pth"
DEFAULT_CONFIG_22    = "data/weights/runtime_config.json"


def _unpack_dataset(ret):
    # Accept (X,y) or (X,y,meta)
    if isinstance(ret, tuple) and len(ret) >= 2:
        return ret[0], ret[1]
    raise RuntimeError("Unexpected return from loader.load_dataset()")


def _infer_hidden_dims(state_dict):
    """Infer hidden sizes from saved EEGFeatureMLP checkpoint."""
    # Strip 'module.' if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    hidden = []
    if "net.0.weight" in state_dict:  # first Linear
        hidden.append(int(state_dict["net.0.weight"].shape[0]))
    if "net.4.weight" in state_dict:  # second Linear
        hidden.append(int(state_dict["net.4.weight"].shape[0]))
    return hidden if hidden else [64, 64], state_dict


def main(args):
    # ---------- Load a small dataset slice like training ----------
    if args.fourch:
        from src.utils.data_loader_4ch import EEGFeatureLoader4Ch
        loader = EEGFeatureLoader4Ch(subjects=[1], tmin=0.5, tmax=2.5)
        pipeline_path = args.pipeline_path or DEFAULT_PIPELINE_4CH
        weights_path  = args.weights_path  or DEFAULT_WEIGHTS_4CH
        n_expected_ch = 4
        runtime_order = ["C4", "Fp2", "Fp1", "C3"]
        print("Using 4-ch BNCI proxies: [C4, FC2, FC1, C3]")
    else:
        from src.utils.data_loader import EEGFeatureLoader
        loader = EEGFeatureLoader()
        pipeline_path = args.pipeline_path or DEFAULT_PIPELINE_22
        weights_path  = args.weights_path  or DEFAULT_WEIGHTS_22
        n_expected_ch = 22
        runtime_order = None
        print("Using 22-ch BNCI (original).")

    X, y = _unpack_dataset(loader.load_dataset())
    n, C, T = X.shape
    uniq, cnt = np.unique(y, return_counts=True)
    cls_dist = {int(k): int(v) for k, v in zip(uniq, cnt)}

    print("\nDataset peek:")
    print(f"  Trials: {n}, Channels: {C}, Samples/window: {T}")
    print(f"  Class distribution: {cls_dist}  (0=left,1=right)")

    idx = int(np.clip(args.index, 0, n - 1))
    x = X[idx]  # (C, T)
    if args.show_raw:
        print(f"  Example window [first 5 samples of ch0]: {np.round(x[0,:5], 4).tolist()}")

    # ---------- Load pipeline and make features ----------
    fp: FeaturePipeline = FeaturePipeline.load(pipeline_path)
    feat = fp.transform(x[None, ...])         # (1, feat_dim)
    feat_dim = int(feat.shape[1])

    # ---------- Build model to match checkpoint ----------
    sd_raw = torch.load(weights_path, map_location="cpu")
    hidden_dims, sd = _infer_hidden_dims(sd_raw)
    print(f"\nInferred hidden_dims from checkpoint: {hidden_dims}")

    model = EEGFeatureMLP(
        input_dim=feat_dim,
        hidden_dims=tuple(hidden_dims),
        n_classes=2,
        dropout=0.2,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(feat).float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

    print("\nSingle-window prediction:")
    print(f"  True label: {int(y[idx])}  (0=left,1=right)")
    print(f"  Pred probs: {np.round(probs, 6).tolist()}")
    print(f"  Pred class: {pred}")

    # ---------- Optional: write runtime config ----------
    if args.write_config:
        cfg_out = args.config_out or (DEFAULT_CONFIG_4CH if args.fourch else DEFAULT_CONFIG_22)
        bands = getattr(fp, "bands", ((8,12),(13,30)))
        mu, beta = bands[0], bands[-1]
        csp_comps = getattr(getattr(fp, "csp", None), "n_components", 2 if args.fourch else 4)

        cfg = {
            "sampling_rate_hz": getattr(fp, "fs", 250),
            "bandpass_hz": [8, 30],
            "window_seconds": 2.0,
            "window_samples": 500,
            "n_channels_expected": n_expected_ch,
            "channel_order_runtime": runtime_order,
            "features": {
                "csp_components": int(csp_comps),
                "bands": {"mu": list(mu), "beta": list(beta)},
                "feature_dim": feat_dim,
                "scaler": "StandardScaler"
            },
            "model": {
                "arch": "EEGFeatureMLP",
                "hidden_dims": hidden_dims,
                "dropout": 0.2,
                "n_classes": 2,
                "class_map": {"0": "left", "1": "right"},
                "weights_path": os.path.abspath(weights_path),
            },
            "pipeline_path": os.path.abspath(pipeline_path),
            "notes": "Feed a window (C,T) -> FeaturePipeline -> MLP. For 4-ch runtime order: C4,Fp2,Fp1,C3."
        }
        os.makedirs(os.path.dirname(cfg_out), exist_ok=True)
        with open(cfg_out, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"\nRuntime config written to: {cfg_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fourch", action="store_true",
                    help="Use 4-channel loader & defaults.")
    ap.add_argument("--index", type=int, default=5,
                    help="Which trial to inspect/predict")
    ap.add_argument("--show_raw", action="store_true",
                    help="Print a few raw samples of ch0")
    ap.add_argument("--pipeline_path", default="",
                    help="Override pipeline path")
    ap.add_argument("--weights_path", default="",
                    help="Override weights path")
    ap.add_argument("--write_config", action="store_true",
                    help="Write a runtime config JSON next to weights/pipeline")
    ap.add_argument("--config_out", default="",
                    help="Optional explicit config output path")
    args = ap.parse_args()
    main(args)
