import numpy as np
import torch
import torch.nn as nn

try:
    import joblib
except Exception:
    joblib = None

from src.models.mlp import EEGFeatureMLP
from src.pipeline.feature_pipeline import FeaturePipeline


class RealTimeClassifier:
    """
    Load a trained MLP + FeaturePipeline and run predictions on incoming windows.

    Expected window shape: (C, T) or (N, C, T)
    """

    def __init__(self, pipeline_path: str, weights_path: str, input_dim: int, n_classes: int, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # load pipeline
        self.pipeline: FeaturePipeline = FeaturePipeline.load(pipeline_path)

        # build model and load weights
        self.model = EEGFeatureMLP(input_dim=input_dim, hidden_dim=64, n_classes=n_classes).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def predict_proba(self, X_window: np.ndarray) -> np.ndarray:
        """
        X_window: (C, T) or (N, C, T)
        Returns probs: (N, n_classes)
        """
        if X_window.ndim == 2:
            X_window = X_window[np.newaxis, ...]  # (1, C, T)

        # feature transform
        X_feat = self.pipeline.transform(X_window)  # (N, D)
        with torch.no_grad():
            logits = self.model(torch.tensor(X_feat, dtype=torch.float32, device=self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X_window: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_window)
        return probs.argmax(axis=1)
