# testing/train_feature_with_pipeline_test_mlp.py
import os, json, time
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP

def _unpack_dataset(ret):
    # Accept (X,y) or (X,y,meta)
    if isinstance(ret, tuple) and len(ret) >= 2:
        return ret[0], ret[1]
    raise RuntimeError("Unexpected return from loader.load_dataset()")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fourch", action="store_true",
                    help="Use 4-channel BNCI proxies (C4,FC2,FC1,C3).")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    # ---------------- Load data ----------------
    if args.fourch:
        # 4-ch pipeline training
        from src.utils.data_loader_4ch import EEGFeatureLoader4Ch
        print("Loading BNCI 2014-001 (4-ch proxies: C4,FC2,FC1,C3)…")
        loader = EEGFeatureLoader4Ch(subjects=[1], tmin=0.5, tmax=2.5)
        X, y = _unpack_dataset(loader.load_dataset())    # (N,4,500), 0/1
        # Feature pipeline for 4-ch
        pipe = FeaturePipeline(
            fs=250,
            n_csp=2,
            bands=((8,12),(13,30)),
            # indices in training order [C4,FC2,FC1,C3] => (C3↔C4 = (3,0), Fp1↔Fp2 proxy = (2,1))
            asym_pairs=((3,0),(2,1)),
            use_fbcsp=True,
            fb_bands=((8,10),(10,12),(12,14),(14,18),(18,26),(26,30)),
            shrinkage="lw",
        )
        suffix = "_4ch"
    else:
        # 22-ch pipeline training (original)
        from src.utils.data_loader import EEGFeatureLoader
        print("Loading BNCI 2014-001 via MOABB (Left/Right, 2s, 8–30 Hz)…")
        loader = EEGFeatureLoader()
        X, y = loader.load_dataset()                    # (N,22,500), 0/1
        # If you have channel names available, you can add asymmetry pairs here.
        pipe = FeaturePipeline(
            fs=250,
            n_csp=4,
            bands=((8,12),(13,30)),
            asym_pairs=(),              # add (idxC3, idxC4) etc. if you track names
            use_fbcsp=True,
            fb_bands=((8,12),(12,16),(16,26)),
            shrinkage="lw",
        )
        suffix = ""

    print(f"Trials: {X.shape[0]}, Channels: {X.shape[1]}, Samples: {X.shape[2]}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ---------------- Split ----------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------- Features ----------------
    Xtr = pipe.fit_transform(X_tr, y_tr)
    Xva = pipe.transform(X_val)
    print(f"Train features: {Xtr.shape}  Val features: {Xva.shape}")

    # ---------------- Model ----------------
    model = EEGFeatureMLP(input_dim=Xtr.shape[1], hidden_dims=(64,64), n_classes=2, dropout=0.2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    # ---------------- Train ----------------
    best_acc, wait, patience = 0.0, 0, 12
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        xb = torch.tensor(Xtr, dtype=torch.float32)
        yb = torch.tensor(y_tr, dtype=torch.long)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward(); opt.step()

        # val
        model.eval()
        with torch.no_grad():
            v_logits = model(torch.tensor(Xva, dtype=torch.float32))
            v_loss = loss_fn(v_logits, torch.tensor(y_val, dtype=torch.long)).item()
            v_pred = v_logits.argmax(1).numpy()
            v_acc  = float((v_pred == y_val).mean())

        print(f"[{epoch:03d}] train_loss={float(loss.item()):.4f} | val_loss={v_loss:.4f} val_acc={v_acc:.4f}")

        if v_acc > best_acc + 1e-4:
            best_acc, wait = v_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # ---------------- Save ----------------
    model.load_state_dict(best_state)
    os.makedirs("data/weights", exist_ok=True)
    model_path = f"data/weights/feature_mlp_pipeline{suffix}.pth"
    pipe_path  = f"data/weights/feature_pipeline{suffix}.joblib"
    torch.save(model.state_dict(), model_path)
    pipe.save(pipe_path)
    print("Saved model:   ", model_path)
    print("Saved pipeline:", pipe_path)

    # quick report
    with torch.no_grad():
        v_logits = model(torch.tensor(Xva, dtype=torch.float32))
        v_pred = v_logits.argmax(1).numpy()
    print("\nVal accuracy:", (v_pred==y_val).mean())
    print("Confusion:\n", confusion_matrix(y_val, v_pred))
    print("\nReport:\n", classification_report(y_val, v_pred, target_names=["left","right"]))

if __name__ == "__main__":
    main()
