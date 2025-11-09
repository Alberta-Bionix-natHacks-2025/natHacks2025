"""
Train feature MLP using a fitted FeaturePipeline (CSP+PSD+Asym+Scaler).
Run from project root:
    python -m testing.train_feature_mlp_with_pipeline
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.data_loader import EEGFeatureLoader
from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP


def main():
    # 1) Load RAW windows (not pre-featured)
    #    Reuse the MOABB branch inside loader via its load_dataset()
    raw_loader = EEGFeatureLoader(sampling_rate=250)  # we will call load_dataset directly
    X_raw, y = raw_loader.load_dataset()  # (N, C, T), (N,)

    # 2) Split raw windows
    X_tr, X_va, y_tr, y_va = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)

    # 3) Fit pipeline ONLY on train, transform both
    fp = FeaturePipeline(fs=250, n_csp=4, channel_pairs=[(0, 1)])
    X_tr_feat = fp.fit_transform(X_tr, y_tr)
    X_va_feat = fp.transform(X_va)

    print("Train features:", X_tr_feat.shape, "Val features:", X_va_feat.shape)

    # 4) Torch setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    hidden_dim = 64
    n_classes = int(np.max(y) + 1)

    model = EEGFeatureMLP(input_dim=X_tr_feat.shape[1], hidden_dim=hidden_dim, n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    tr_ds = TensorDataset(torch.tensor(X_tr_feat, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(X_va_feat, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    # 5) Train
    best_acc = -1.0
    os.makedirs("data/weights", exist_ok=True)
    weights_path = "data/weights/feature_mlp_pipeline.pth"
    pipeline_path = "data/weights/feature_pipeline.joblib"

    for epoch in range(1, 31):
        model.train()
        running = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        # val
        model.eval()
        correct, total, vloss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                vloss += float(loss.item())
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += yb.size(0)
        va_acc = correct / max(1, total)
        print(f"[{epoch:03d}] train_loss={running/len(tr_loader):.4f} | val_loss={vloss/len(va_loader):.4f} val_acc={va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), weights_path)
            fp.save(pipeline_path)

    print(f"Best val_acc={best_acc:.4f}")
    print(f"Saved model:    {weights_path}")
    print(f"Saved pipeline: {pipeline_path}")


if __name__ == "__main__":
    main()
