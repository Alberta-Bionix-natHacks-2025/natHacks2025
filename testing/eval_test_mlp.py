"""
Evaluate a saved feature-MLP model with its FeaturePipeline.
Run from project root:
    python -m testing.eval_feature_mlp
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch

from src.utils.data_loader import EEGFeatureLoader
from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP


def main():
    pipeline_path = "data/weights/feature_pipeline.joblib"
    weights_path = "data/weights/feature_mlp_pipeline.pth"

    # 1) Load RAW windows
    raw_loader = EEGFeatureLoader(sampling_rate=250)
    X_raw, y = raw_loader.load_dataset()
    X_tr, X_va, y_tr, y_va = train_test_split(X_raw, y, test_size=0.2, random_state=123, stratify=y)

    # 2) Load pipeline and transform val
    fp = FeaturePipeline.load(pipeline_path)
    X_va_feat = fp.transform(X_va)

    # 3) Build model & load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGFeatureMLP(input_dim=X_va_feat.shape[1], hidden_dim=64, n_classes=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X_va_feat, dtype=torch.float32, device=device))
        y_pred = logits.argmax(dim=1).cpu().numpy()

    acc = (y_pred == y_va).mean()
    cm = confusion_matrix(y_va, y_pred)
    print(f"Val accuracy: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print("\nClassification report:\n", classification_report(y_va, y_pred, target_names=['left','right']))


if __name__ == "__main__":
    main()
