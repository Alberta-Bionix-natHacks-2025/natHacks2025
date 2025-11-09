# api/flask_rt.py
import io, json, sys
from pathlib import Path
from typing import List, Tuple, Deque, Optional
from collections import deque
import threading
import numpy as np

from flask import Blueprint, request, jsonify
from flask_cors import CORS

# ---- repo imports (assumes this file lives under repo root /api) ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP

# optional resampling if fs != 200
try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None

bp = Blueprint("eeg", __name__)
CORS(bp)

# ------------------- constants (Option B: 200Hz) -------------------
EXPECTED_FS = 200
WIN_SEC = 2.0
WIN_SAMPLES = int(EXPECTED_FS * WIN_SEC)   # 400
MANIFEST_PATH = Path("data/weights/manifest_4ch.json")

# ------------------- load artifacts once -------------------
def _infer_hidden_dims(sd: dict):
    h = []
    if "net.0.weight" in sd:
        h.append(int(sd["net.0.weight"].shape[0]))
    if "net.4.weight" in sd:
        h.append(int(sd["net.4.weight"].shape[0]))
    return h or [64, 64]

def load_artifacts(manifest_path: Path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    pipeline_path = Path(manifest.get("pipeline_path", "data/weights/feature_pipeline_4ch.joblib"))
    weights_path  = Path(manifest["model"].get("weights_path", "data/weights/feature_mlp_4ch.pth"))

    pipe: FeaturePipeline = FeaturePipeline.load(str(pipeline_path))

    import torch
    sd = torch.load(str(weights_path), map_location="cpu")
    hidden_dims = manifest["model"].get("hidden_dims") or _infer_hidden_dims(sd)
    input_dim = int(manifest["features"]["feature_dim"])
    model = EEGFeatureMLP(
        input_dim=input_dim,
        hidden_dims=tuple(hidden_dims),
        n_classes=int(manifest["model"]["n_classes"]),
        dropout=float(manifest["model"].get("dropout", 0.2)),
    )
    model.load_state_dict(sd)
    model.eval()

    runtime_order = manifest.get("channel_order_runtime", ["C4","Fp2","Fp1","C3"])
    train_order   = manifest.get("channel_order_train",   ["C4","FC2","FC1","C3"])
    class_map     = manifest["model"]["class_map"]
    return manifest, pipe, model, runtime_order, train_order, class_map

MANIFEST, PIPE, MODEL, RUNTIME_ORDER, TRAIN_ORDER, CLASS_MAP = load_artifacts(MANIFEST_PATH)
_model_lock = threading.Lock()  # safety if Flask runs threaded

# ------------------- helpers -------------------
def make_index_map(runtime_names: List[str], train_names: List[str]) -> List[int]:
    name2idx = {nm.upper(): i for i, nm in enumerate(runtime_names)}
    mapped = []
    for t in train_names:
        key = t.upper()
        if key == "FC2" and "FP2" in name2idx:
            mapped.append(name2idx["FP2"])
        elif key == "FC1" and "FP1" in name2idx:
            mapped.append(name2idx["FP1"])
        else:
            if key not in name2idx:
                raise ValueError(f"Missing channel required for training order: {t}")
            mapped.append(name2idx[key])
    return mapped

RUNTIME_TO_TRAIN = make_index_map(RUNTIME_ORDER, TRAIN_ORDER)

def reorder_to_train(window_runtime: np.ndarray) -> np.ndarray:
    # (C_runtime, T) -> (C_train, T)
    return window_runtime[np.array(RUNTIME_TO_TRAIN), :]

def ensure_window_runtime(x_rows: np.ndarray, fs: int, channels: List[str]) -> np.ndarray:
    """
    x_rows: (N,4) in provided runtime channel order
    returns: (4, WIN_SAMPLES) in TRAIN order
    """
    x = np.asarray(x_rows, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("data must be (N,4)")

    # resample if needed
    xT = x.T  # (4, N)
    if fs != EXPECTED_FS:
        if resample_poly is None:
            raise RuntimeError("scipy is required for resampling when fs != 200.")
        from math import gcd
        g = gcd(fs, EXPECTED_FS)
        up, down = EXPECTED_FS // g, fs // g
        xT = resample_poly(xT, up=up, down=down, axis=1)

    # trim/pad to window
    if xT.shape[1] >= WIN_SAMPLES:
        xT = xT[:, -WIN_SAMPLES:]
    else:
        pad = WIN_SAMPLES - xT.shape[1]
        xT = np.pad(xT, ((0,0),(pad,0)))

    # custom mapping if channels differ from manifest default
    if channels != RUNTIME_ORDER:
        idxmap = make_index_map(channels, TRAIN_ORDER)
        xT = xT[np.array(idxmap), :]
        return xT
    else:
        return reorder_to_train(xT)

def try_parse_csv(bytestr: bytes) -> np.ndarray:
    s = bytestr.decode("utf-8", errors="ignore")
    lines = s.splitlines()
    has_header = bool(lines and any(c.isalpha() for c in lines[0]))
    import csv
    r = csv.reader(io.StringIO(s))
    rows = []
    for i, row in enumerate(r):
        if i == 0 and has_header:
            continue
        if not row:
            continue
        vals = []
        for v in row[:4]:
            try:
                vals.append(float(v))
            except:
                vals.append(np.nan)
        if len(vals) == 4 and not any(np.isnan(vals)):
            rows.append(vals)
    if not rows:
        raise ValueError("No numeric rows found in CSV")
    return np.asarray(rows, dtype=np.float32)

def run_model_on_window(window_train_order: np.ndarray) -> Tuple[int, List[float]]:
    feat = PIPE.transform(window_train_order[None, ...])  # (1, feat_dim)
    import torch
    with _model_lock, torch.no_grad():
        logits = MODEL(torch.from_numpy(feat).float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return pred, probs.tolist()

# ------------------- routes -------------------
@bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "expected_fs": EXPECTED_FS,
        "win_samples": WIN_SAMPLES,
        "runtime_order": RUNTIME_ORDER,
        "train_order": TRAIN_ORDER
    })

@bp.route("/predict-array", methods=["POST"])
def predict_array():
    """
    JSON:
    {
      "fs": 200,
      "channels": ["C4","Fp2","Fp1","C3"],
      "data": [[... 4 floats ...], ...]   # >= 400 rows recommended
    }
    """
    try:
        payload = request.get_json(force=True)
        fs = int(payload.get("fs", EXPECTED_FS))
        channels = payload.get("channels", RUNTIME_ORDER)
        data = np.asarray(payload["data"], dtype=np.float32)
        window = ensure_window_runtime(data, fs, channels)   # (4, 400)
        pred, probs = run_model_on_window(window)
        return jsonify({"pred": pred, "label": CLASS_MAP[str(pred)], "probs": probs, "class_map": CLASS_MAP})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route("/predict-csv", methods=["POST"])
def predict_csv():
    """
    multipart/form-data with:
      file=@window.csv
      fs=200
      ch0=C4&ch1=Fp2&ch2=Fp1&ch3=C3   (optional)
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "missing file"}), 400
        fs = int(request.form.get("fs", EXPECTED_FS))
        chs = [
            request.form.get("ch0", RUNTIME_ORDER[0]),
            request.form.get("ch1", RUNTIME_ORDER[1]),
            request.form.get("ch2", RUNTIME_ORDER[2]),
            request.form.get("ch3", RUNTIME_ORDER[3]),
        ]
        raw = request.files["file"].read()
        arr = try_parse_csv(raw)  # (N,4)
        window = ensure_window_runtime(arr, fs, chs)
        pred, probs = run_model_on_window(window)
        return jsonify({"pred": pred, "label": CLASS_MAP[str(pred)], "probs": probs, "class_map": CLASS_MAP})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ------------- utility you can call directly from Flask GUI code -------------
def predict_from_numpy(data: np.ndarray, fs: int = 200, channels: List[str] = None) -> dict:
    """
    For internal (same-process) use:
      data: (N,4) in runtime order channels
    """
    if channels is None:
        channels = RUNTIME_ORDER
    window = ensure_window_runtime(data, fs, channels)
    pred, probs = run_model_on_window(window)
    return {"pred": pred, "label": CLASS_MAP[str(pred)], "probs": probs, "class_map": CLASS_MAP}
