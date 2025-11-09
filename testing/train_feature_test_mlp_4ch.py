# testing/train_feature_test_mlp_4ch.py
import os, json, time, traceback, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.utils.data_loader_4ch import EEGFeatureLoader4Ch
from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_device(x: np.ndarray, device):
    return torch.from_numpy(x).float().to(device)


def train_one_epoch(model, optimizer, loss_fn, xb, yb, batch_size=64, device="cpu", scheduler=None):
    model.train()
    n = xb.shape[0]
    idx = np.arange(n); np.random.shuffle(idx)
    xb = xb[idx]; yb = yb[idx]
    total_loss, total_correct = 0.0, 0
    for start in range(0, n, batch_size):
        end = start + batch_size
        x = torch.from_numpy(xb[start:end]).float().to(device)
        y = torch.from_numpy(yb[start:end]).long().to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_loss += float(loss.item()) * x.size(0)
            total_correct += int((preds == y).sum().item())
    return total_loss / n, total_correct / n

def time_jitter(X, max_shift=25):  # 25 samples ≈ 0.1 s @ 250Hz
    Xj = X.copy()
    N, C, T = Xj.shape
    sel = np.random.rand(N) < 0.5
    shifts = np.random.randint(-max_shift, max_shift+1, size=sel.sum())
    idx = np.where(sel)[0]
    for k, s in zip(idx, shifts):
        Xj[k] = np.roll(Xj[k], s, axis=1)
    return Xj


@torch.no_grad()
def eval_epoch(model, loss_fn, xb, yb, batch_size=256, device="cpu"):
    model.eval()
    n = xb.shape[0]
    total_loss, total_correct = 0.0, 0
    for start in range(0, n, batch_size):
        end = start + batch_size
        x = _to_device(xb[start:end], device)
        y = torch.from_numpy(yb[start:end]).long().to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((preds == y).sum().item())
    return total_loss / n, total_correct / n


def main():
    print("[trainer-4ch] start", flush=True)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Load raw 4-ch windows ----------
    loader = EEGFeatureLoader4Ch(subjects=[1], tmin=0.5, tmax=2.5)  # list!
    X, y, _ = loader.load_dataset()            # (N,4,500) | labels 0/1
    print(f"[trainer-4ch] windows: {X.shape}, labels: {y.shape}", flush=True)

    # ---------- Split ----------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[trainer-4ch] split => train: {X_tr.shape}, val: {X_val.shape}", flush=True)

    # ---------- Feature pipeline ----------
    '''
    pipe = FeaturePipeline(
        fs=250,
        n_csp=2,                          # good for 4ch
        bands=((8,12),(13,30)),
        asym_pairs=((3,0),),              # C3 vs C4
        use_fbcsp=True,                   # enable small filter bank
        fb_bands=((8,12),(12,16),(16,26)),
        shrinkage="lw",
    )
    '''
    pipe = FeaturePipeline(
        fs=250,
        n_csp=2,
        bands=((8,12),(13,30)),
        asym_pairs=((3,0),(2,1)),              # C3↔C4 and Fp1↔Fp2 asymmetry
        use_fbcsp=True,
        fb_bands=((8,10),(10,12),(12,14),(14,18),(18,26),(26,30)),  # ← more granularity
        shrinkage="lw",
    )
    # before fit:
    X_tr = time_jitter(X_tr, max_shift=25)
    Xtr = pipe.fit_transform(X_tr, y_tr)
    Xva = pipe.transform(X_val)
    print(f"[trainer-4ch] features => train: {Xtr.shape}, val: {Xva.shape}", flush=True)

    # ---------- Model / Optim / Loss / Scheduler ----------
    model = EEGFeatureMLP(
        input_dim=Xtr.shape[1],
        hidden_dims=(64, 64),
        n_classes=2,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,
    steps_per_epoch=max(1, int(np.ceil(len(Xtr)/64))),
    epochs=80,  # train a bit longer, still with early stop
    pct_start=0.25
    )

    # ---------- Train ----------
    best_acc = 0.0
    best_state = None
    patience, patience_ctr = 12, 0
    max_epochs = 60
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, optimizer, loss_fn, Xtr, y_tr,
                                          batch_size=64, device=device, scheduler=scheduler)
        va_loss, va_acc = eval_epoch(model, loss_fn, Xva, y_val,
                                     batch_size=256, device=device)
        scheduler.step()

        print(f"[{epoch:03d}] tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}", flush=True)

        improved = va_acc > best_acc + 1e-4
        if improved:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("[trainer-4ch] early stop", flush=True)
                break

    print(f"[trainer-4ch] done in {time.time()-t0:.1f}s | best val_acc={best_acc:.4f}", flush=True)

    # ---------- Save best ----------
    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs("data/weights", exist_ok=True)
    weights_path = "data/weights/feature_mlp_4ch.pth"
    pipe_path = "data/weights/feature_pipeline_4ch.joblib"
    torch.save(model.state_dict(), weights_path)
    pipe.save(pipe_path)

    manifest = {
        "sampling_rate_hz": 250,
        "bandpass_hz": [8, 30],
        "window_seconds": 2.0,
        "window_samples": 500,
        "n_channels_expected": 4,
        "channel_order_train": loader.BNCI_TRAIN_CHANNELS,
        "channel_order_runtime": ["C4", "Fp2", "Fp1", "C3"],
        "features": {
            "csp_components": 2,
            "bands": {"mu": [8, 12], "beta": [13, 30]},
            "feature_dim": int(Xtr.shape[1]),
            "scaler": "StandardScaler"
        },
        "model": {
            "arch": "EEGFeatureMLP",
            "hidden_dims": [64, 64],
            "dropout": 0.2,
            "n_classes": 2,
            "class_map": {"0": "left", "1": "right"},
            "weights_path": weights_path
        },
        "pipeline_path": pipe_path,
        "notes": "4-ch MI proxies C4,FC2,FC1,C3 (runtime C4,Fp2,Fp1,C3)"
    }
    with open("data/weights/manifest_4ch.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[trainer-4ch] saved:\n  {weights_path}\n  {pipe_path}\n  data/weights/manifest_4ch.json",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("\n[trainer-4ch] FATAL ERROR:\n" + traceback.format_exc(), flush=True)
        raise
