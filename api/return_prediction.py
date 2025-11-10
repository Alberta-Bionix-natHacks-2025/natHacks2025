from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Optional
import threading
import numpy as np
import torch

from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None  # required only if fs != EXPECTED_FS

# ------------------------------------------------------------------------------------
# Model expects: 4ch, 200 Hz, 2.0 s => 400 samples, band 8–30 Hz inside pipeline
# ------------------------------------------------------------------------------------
EXPECTED_FS = 200
WIN_SEC = 2.0
WIN_SAMPLES = int(EXPECTED_FS * WIN_SEC)
DEFAULT_RUNTIME_ORDER = ["C4", "Fp2", "Fp1", "C3"]  # order you send from the app

_MANIFEST_PATH = Path("data/weights/manifest_4ch.json")

def _infer_hidden_dims(sd: dict):
    h = []
    if "net.0.weight" in sd:
        h.append(int(sd["net.0.weight"].shape[0]))
    if "net.4.weight" in sd:
        h.append(int(sd["net.4.weight"].shape[0]))
    return h or [64, 64]

# ---------- load artifacts once ----------
with open(_MANIFEST_PATH, "r") as f:
    _MANIFEST = json.load(f)

_PIPE: FeaturePipeline = FeaturePipeline.load(_MANIFEST["pipeline_path"])
_sd = torch.load(_MANIFEST["model"]["weights_path"], map_location="cpu")
_hidden = _MANIFEST["model"].get("hidden_dims") or _infer_hidden_dims(_sd)
_INPUT_DIM = int(_MANIFEST["features"]["feature_dim"])

_MODEL = EEGFeatureMLP(
    input_dim=_INPUT_DIM,
    hidden_dims=tuple(_hidden),
    n_classes=int(_MANIFEST["model"]["n_classes"]),
    dropout=float(_MANIFEST["model"].get("dropout", 0.2)),
)
_MODEL.load_state_dict(_sd)
_MODEL.eval()

_CLASS_MAP = {int(k): v for k, v in _MANIFEST["model"]["class_map"].items()}
_TRAIN_ORDER = [c.upper() for c in _MANIFEST.get("channel_order_train", ["C4", "FC2", "FC1", "C3"])]
_RUNTIME_DEFAULT = [c.upper() for c in _MANIFEST.get("channel_order_runtime", DEFAULT_RUNTIME_ORDER)]

_lock = threading.Lock()  # inference safety


def _zscore_per_channel(win: np.ndarray) -> np.ndarray:
    # win: (C, T) float32
    w = win.astype(np.float32, copy=True)
    for c in range(w.shape[0]):
        mu = float(w[c].mean())
        sd = float(w[c].std())
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        w[c] = (w[c] - mu) / sd
    return w

# ---------- helpers ----------
def _make_index_map(runtime_names: List[str], train_names: List[str]) -> List[int]:
    """Map runtime channel names to the training order, with FC↔FP proxy logic."""
    r2i = {nm.upper(): i for i, nm in enumerate(runtime_names)}
    out = []
    for t in train_names:
        key = t.upper()
        if key == "FC1" and "FP1" in r2i:
            out.append(r2i["FP1"])
        elif key == "FC2" and "FP2" in r2i:
            out.append(r2i["FP2"])
        else:
            if key not in r2i:
                raise ValueError(f"Required channel '{t}' missing in runtime names {runtime_names}")
            out.append(r2i[key])
    return out


class _RingBuffer:
    """Per-channel rolling buffer of the last WIN_SAMPLES at EXPECTED_FS."""
    def __init__(self, n_channels: int = 4, capacity: int = WIN_SAMPLES):
        self.n_ch = n_channels
        self.cap = capacity
        self.buf = np.zeros((n_channels, capacity), dtype=np.float32)
        self.filled = 0  # how many valid samples we’ve accumulated so far

    def append(self, block: np.ndarray):
        """
        block: (n_ch, Nnew) at EXPECTED_FS, runtime order
        """
        if block.ndim != 2 or block.shape[0] != self.n_ch:
            raise ValueError(f"block must be (n_ch, Nnew); got {block.shape}")
        N = block.shape[1]
        if N >= self.cap:
            # keep only the last cap samples
            self.buf = block[:, -self.cap:].astype(np.float32, copy=False)
            self.filled = self.cap
            return
        # roll left by N and insert at the end
        self.buf = np.roll(self.buf, -N, axis=1)
        self.buf[:, -N:] = block
        self.filled = min(self.cap, self.filled + N)

    def ready(self) -> bool:
        return self.filled >= self.cap

    def window(self) -> np.ndarray:
        """
        Returns (n_ch, cap) window in runtime order.
        """
        return self.buf.copy()


# Single global buffer (sufficient for one stream)
_BUF = _RingBuffer(n_channels=4, capacity=WIN_SAMPLES)
_RUNTIME_NAMES = [c.upper() for c in DEFAULT_RUNTIME_ORDER]  # set by first call, if provided


def _ensure_fs_and_stack(block_rows: np.ndarray, fs_in: int) -> np.ndarray:
    """
    block_rows: (N, 4) in runtime order → resample to EXPECTED_FS if needed → (4, Nnew)
    """
    x = np.asarray(block_rows, dtype=np.float32).T  # (4, N)
    fs_in = int(fs_in)
    if fs_in == EXPECTED_FS:
        return x
    if resample_poly is None:
        raise RuntimeError("scipy is required to resample when fs != 200 Hz")
    from math import gcd
    g = gcd(fs_in, EXPECTED_FS)
    up, down = EXPECTED_FS // g, fs_in // g
    return resample_poly(x, up=up, down=down, axis=1).astype(np.float32, copy=False)


# ---------- public API ----------
def predict_list(
    data: np.ndarray,
    fs: int,
    channels: List[str] | None = None,
    neutral_margin: float = 0.15,
    min_samples: int = 50,
) -> list:
    """
    Returns [pred_code, label, p_left, p_right]
      pred_code: 0=left, 1=right, 2=neutral

    Accumulates a 2-second window in a ring buffer. Until full, returns neutral.
    """
    global _RUNTIME_NAMES

    if channels is None:
        channels = DEFAULT_RUNTIME_ORDER
    _RUNTIME_NAMES = [c.upper() for c in channels]

    # require some fresh data at least
    if data.shape[0] < min_samples:
        return [2, "neutral", 0.5, 0.5]

    # (N, 4) runtime → (4, Nnew) @200 Hz
    block_4xN = _ensure_fs_and_stack(data, fs)

    # basic sanity: if block is flat or NaN, don’t update
    if not np.isfinite(block_4xN).all() or np.max(block_4xN.std(axis=1)) < 1e-6:
        return [2, "neutral", 0.5, 0.5]

    # update ring
    _BUF.append(block_4xN)

    # Not enough history yet? neutral.
    if not _BUF.ready():
        return [2, "neutral", 0.5, 0.5]

    # Build window in TRAIN order
    win_runtime = _BUF.window()                                   # (4, 400)
    idxmap = _make_index_map(_RUNTIME_NAMES, _TRAIN_ORDER)
    win_train = win_runtime[np.array(idxmap), :]       
    # (4, 400) reordered

    win_train = _zscore_per_channel(win_train)
    
    # Features → logits → probs
    feat = _PIPE.transform(win_train[None, ...])                 # (1, F)
    with _lock, torch.no_grad():
        logits = _MODEL(torch.from_numpy(feat).float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    p_left, p_right = float(probs[0]), float(probs[1])

    # 3-state decision
    if abs(p_left - p_right) < neutral_margin:
        return [2, "neutral", p_left, p_right]
    pred_idx = int(np.argmax(probs))
    return [pred_idx, _CLASS_MAP.get(pred_idx, "unknown"), p_left, p_right]




def predict_from_brainflow_block(
    board_data: np.ndarray,
    eeg_channel_indices: List[int],
    fs: int,
    channel_names: List[str] | None = None,
    **kwargs,
) -> list:
    """
    Convenience for BrainFlow:
      board_data: (num_rows, Nnew) with NEW samples only (get_board_data()).
      eeg_channel_indices: e.g. BoardShim.get_eeg_channels(board_id)[:4]
      channel_names: names for those four channels in runtime order
                     (default ["C4","Fp2","Fp1","C3"])
    """
    block = np.asarray(board_data, dtype=np.float32)
    if block.ndim != 2 or block.shape[1] == 0:
        return [2, "neutral", 0.5, 0.5]

    # slice 4 channels in the exact order you’ll name in channel_names
    four = block[eeg_channel_indices, :].T  # (Nnew, 4)

    # pass to the ring-buffered predictor
    return predict_list(
        data=four,
        fs=fs,
        channels=channel_names or DEFAULT_RUNTIME_ORDER,
        **kwargs,
    )
