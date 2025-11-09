"""
Quick training demo for the feature MLP.
Run from project root:
    python -m testing.train_feature_mlp_demo
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.utils.data_loader import EEGFeatureLoader
from src.utils.training_loop import fit_feature_mlp


def main():
    # 1) Load features (already standardized by the loader)
    loader = EEGFeatureLoader(
        sampling_rate=250,
        n_csp_components=4,
        channel_pairs=[(0, 1)],
    )
    X, y = loader.load_features()

    print(f"Loaded features: X={X.shape}, y={y.shape}")

    # 2) Train MLP on features
    # NOTE: use_scaler=False because loader already standardized features.
    result = fit_feature_mlp(
        X, y,
        save_dir="data/weights",
        run_name="feature_mlp",
        hidden_dim=64,
        batch_size=64,
        epochs=35,
        lr=1e-3,
        weight_decay=1e-4,
        val_size=0.2,
        seed=42,
        use_scaler=False,     # keep False unless you move scaling into training
        device_str="cuda" if False else "cpu",  # change to "cuda" if you want GPU locally
    )

    print("\n=== Training result ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
