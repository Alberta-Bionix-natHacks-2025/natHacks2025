# src/realtime/csv_adapter.py
import json, time, collections, numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from .inference import RealTimeClassifier
from src.pipeline.feature_pipeline import FeaturePipeline

class SlidingBuffer:
    def __init__(self, n_ch: int, win_samples: int):
        self.n_ch = n_ch
        self.win = win_samples
        self.buf = np.zeros((n_ch, 0), dtype=np.float32)

    def push(self, frame: np.ndarray):
        """frame: (n_ch,) or (n_ch, n_samples)"""
        if frame.ndim == 1:
            frame = frame.reshape(self.n_ch, 1)
        self.buf = np.concatenate([self.buf, frame], axis=1)
        if self.buf.shape[1] > self.win:
            self.buf = self.buf[:, -self.win:]

    def ready(self) -> bool:
        return self.buf.shape[1] == self.win

    def window(self) -> np.ndarray:
        assert self.ready()
        return self.buf.copy()

def load_runtime_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def build_classifier_from_config(cfg_path: str, device: str = "cpu") -> RealTimeClassifier:
    cfg = load_runtime_config(cfg_path)

    fp = FeaturePipeline.load(cfg["pipeline_path"])
    rtc = RealTimeClassifier(
        pipeline_path=cfg["pipeline_path"],
        weights_path=cfg["model"]["weights_path"],
        input_dim=fp.feature_dim,
        n_classes=cfg["model"]["n_classes"],
        device=device,
    )
    rtc.class_map = {int(k): v for k, v in cfg["model"]["class_map"].items()}
    rtc.expected_order = cfg["channel_order"]  # ["C4","Fp2","Fp1","C3"]
    rtc.fs = cfg["sampling_rate_hz"]
    rtc.win_samples = cfg["window_samples"]
    return rtc

def predict_from_csv_rows(
    rtc: RealTimeClassifier,
    rows: List[Dict[str, float]],
    column_map: Dict[str, str],  # maps csv column -> channel name (e.g., "EXG Channel 0" -> "C4")
) -> List[Dict]:
    """
    rows: each row is a dict like {"EXG Channel 0": v0, "EXG Channel 1": v1, ...}
    column_map: how CSV columns map to e.g. C4, Fp2, Fp1, C3
    """
    order = rtc.expected_order
    win = rtc.win_samples
    buf = SlidingBuffer(n_ch=len(order), win_samples=win)

    outputs = []
    for r in rows:
        # 1) map to expected channel order
        frame = np.array([r[k] for k in column_map.keys()], dtype=np.float32)
        # reorder to expected_order using column_map values
        name_by_col = {col: ch for col, ch in column_map.items()}
        frame_by_name = {name_by_col[col]: r[col] for col in column_map.keys()}
        frame_ord = np.array([frame_by_name[ch] for ch in order], dtype=np.float32)

        buf.push(frame_ord)

        if buf.ready():
            x = buf.window()  # (n_ch, win_samples)
            probs = rtc.predict_proba(x)
            pred  = rtc.predict(x)
            outputs.append({
                "pred": int(pred[0]),
                "label": rtc.class_map.get(int(pred[0]), str(int(pred[0]))),
                "probs": probs[0].tolist(),
                "t": len(outputs)
            })
    return outputs
