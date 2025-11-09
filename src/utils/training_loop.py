import os
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Models
from src.models.mlp import EEGFeatureMLP


def _to_tensor(x: np.ndarray, y: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    X_t = torch.from_numpy(x.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    return X_t.to(device), y_t.to(device)


def _make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    device: torch.device,
):
    X_train_t, y_train_t = _to_tensor(X_train, y_train, device)
    X_val_t, y_val_t = _to_tensor(X_val, y_val, device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _class_weights(y: np.ndarray, device: torch.device) -> Optional[torch.Tensor]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return None
    total = y.shape[0]
    weights = total / (len(classes) * counts.astype(np.float32))
    # build per-class tensor aligned with [0..n_classes-1]
    max_c = int(classes.max())
    w_vec = np.ones(max_c + 1, dtype=np.float32)
    for c, w in zip(classes, weights):
        w_vec[int(c)] = w
    return torch.tensor(w_vec, dtype=torch.float32, device=device)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_logits, all_y = [], []
    loss_fn = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0

    for xb, yb in loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += float(loss.item())
        n_batches += 1
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    y_pred = all_logits.argmax(axis=1)

    metrics = {
        "loss": total_loss / max(1, n_batches),
        "acc": accuracy_score(all_y, y_pred),
        "f1": f1_score(all_y, y_pred, average="macro"),
    }
    return metrics


def fit_feature_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    save_dir: str = "data/weights",
    run_name: str = "feature_mlp",
    hidden_dim: int = 64,
    batch_size: int = 64,
    epochs: int = 35,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_size: float = 0.2,
    seed: int = 42,
    use_scaler: bool = False,
    scaler: Optional[StandardScaler] = None,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, str]:
    """
    Train an MLP on precomputed feature vectors.

    Notes on scaling:
    - Your current loader already returns standardized features. Keep `use_scaler=False`.
    - If you change the loader to return raw features, set `use_scaler=True` (the loop will fit+save a scaler).
    - If you already have a fitted scaler, pass it via `scaler=` and keep `use_scaler=True`.
    """
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=seed, stratify=y
    )

    # Optional (preferred in production): fit scaler on train and transform both
    scaler_path = None
    if use_scaler:
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        scaler_path = os.path.join(save_dir, f"{run_name}_scaler.pkl")
        try:
            import joblib
            joblib.dump(scaler, scaler_path)
        except Exception as e:
            print(f"[WARN] Could not save scaler: {e}")
            scaler_path = None

    device = torch.device(device_str)

    # Model
    input_dim = X.shape[1]
    n_classes = int(np.max(y) + 1)
    model = EEGFeatureMLP(input_dim=input_dim, hidden_dim=hidden_dim, n_classes=n_classes).to(device)

    # Opt/crit/sched
    weights = _class_weights(y_train, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    train_loader, val_loader = _make_loaders(X_train, y_train, X_val, y_val, batch_size, device)

    best_acc = -1.0
    best_path = os.path.join(save_dir, f"{run_name}.pth")
    meta_path = os.path.join(save_dir, f"{run_name}_meta.json")

    patience = 7
    bad_epochs = 0
    history = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(1, n_batches)
        val_metrics = _evaluate(model, val_loader, device)
        scheduler.step(val_metrics["acc"])

        history.append(
            {"epoch": epoch, "train_loss": train_loss, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        )

        # Early stopping on val acc
        improved = val_metrics["acc"] > best_acc
        if improved:
            best_acc = val_metrics["acc"]
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1

        print(
            f"[{epoch:03d}] train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} | "
            f"best_acc={best_acc:.4f}"
        )

        if bad_epochs >= patience:
            print(f"Early stopping (no improvement for {patience} epochs).")
            break

    t1 = time.time()
    print(f"Training finished in {(t1 - t0):.1f}s. Best val_acc={best_acc:.4f}. Saved to {best_path}")

    # Save meta
    meta = {
        "input_dim": input_dim,
        "n_classes": n_classes,
        "class_names": ["left", "right"] if n_classes == 2 else list(range(n_classes)),
        "hidden_dim": hidden_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "val_size": val_size,
        "best_val_acc": best_acc,
        "paths": {"weights": best_path, "scaler": scaler_path},
        "use_scaler": use_scaler,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {"weights": best_path, "meta": meta_path, "scaler": scaler_path, "history": history}
