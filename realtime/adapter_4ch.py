# realtime/adapter_4ch.py
import argparse, time, json
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from src.pipeline.feature_pipeline import FeaturePipeline
from src.models.mlp import EEGFeatureMLP

MANIFEST = "data/weights/manifest_4ch.json"

def bandpass_sos(lf, hf, fs, order=4):
    return butter(order, [lf, hf], btype="bandpass", fs=fs, output="sos")

def load_artifacts(manifest_path=MANIFEST):
    with open(manifest_path, "r") as f:
        m = json.load(f)
    fp = FeaturePipeline.load(m["pipeline_path"])
    sd = torch.load(m["model"]["weights_path"], map_location="cpu")

    hid = []
    if "net.0.weight" in sd: hid.append(sd["net.0.weight"].shape[0])
    if "net.4.weight" in sd: hid.append(sd["net.4.weight"].shape[0])
    if not hid: hid = [64, 64]

    model = EEGFeatureMLP(
        input_dim=int(m["features"]["feature_dim"]),
        hidden_dims=tuple(hid),
        n_classes=int(m["model"]["n_classes"]),
        dropout=float(m["model"]["dropout"])
    )
    model.load_state_dict(sd)
    model.eval()
    return m, fp, model

def parse_openbci_line(line):
    # OpenBCI raw lines: sample, ch1, ch2, ch3, ch4, ...
    # Return list[float] or None if not a data line
    line = line.strip()
    if not line or line[0] in ("%", "#"):
        return None
    parts = line.split(",")
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        return None
    return vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openbci", help="Path to OpenBCI .txt/.csv to replay")
    ap.add_argument("--cols", default="2,3,4,5",
                    help="1-based columns for [C4,Fp2,Fp1,C3] in the file (default 2,3,4,5)")
    ap.add_argument("--rt", action="store_true", help="Sleep at 1/fs to simulate realtime")
    ap.add_argument("--fs", type=int, default=250, help="Sampling rate if the file header lacks it")
    ap.add_argument("--predict_every", type=float, default=2.0, help="Seconds per prediction window")
    args = ap.parse_args()

    manifest, fp, model = load_artifacts(MANIFEST)
    fs = manifest.get("sampling_rate_hz", args.fs)
    bp = manifest.get("bandpass_hz", [8,30])
    win_sec = manifest.get("window_seconds", args.predict_every)
    n_ch = manifest.get("n_channels_expected", 4)
    order_runtime = manifest.get("channel_order_runtime", ["C4","Fp2","Fp1","C3"])

    if n_ch != 4:
        raise RuntimeError("This adapter is configured for 4 channels.")

    cols = [int(x.strip())-1 for x in args.cols.split(",")]  # to 0-based
    if len(cols) != 4:
        raise ValueError("--cols must specify exactly 4 columns")

    sos = bandpass_sos(bp[0], bp[1], fs, order=4)
    win_samples = int(win_sec * fs)
    ring = np.zeros((4, win_samples), dtype=np.float32)
    filled = 0

    # --- read file & replay ---
    f = open(args.openbci, "r", encoding="utf-8") if args.openbci else None
    get_line = (lambda: f.readline()) if f else (lambda: input())

    print(f"[adapter] fs={fs}Hz, band={bp}, window={win_sec}s/{win_samples} samples, order={order_runtime}")
    print(f"[adapter] reading {args.openbci or 'STDIN'}; using columns (1-based): {args.cols}")
    dt = 1.0 / fs

    while True:
        line = get_line()
        if not line:
            break
        vals = parse_openbci_line(line)
        if vals is None or len(vals) <= max(cols):
            continue

        sample = np.array([vals[c] for c in cols], dtype=np.float32)  # [C4,Fp2,Fp1,C3]
        # append into ring buffer
        ring[:, :-1] = ring[:, 1:]
        ring[:, -1] = sample
        filled = min(filled + 1, win_samples)

        if args.rt:
            time.sleep(dt)

        if filled == win_samples:
            # band-pass to match training
            x = sosfiltfilt(sos, ring, axis=1)
            # pipeline expects (N,C,T)
            feat = fp.transform(x[None, ...])
            with torch.no_grad():
                logits = model(torch.from_numpy(feat).float())
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = int(prob.argmax())

            print(json.dumps({
                "pred_class": pred,
                "pred_label": manifest["model"]["class_map"].get(str(pred), str(pred)),
                "probs": [float(prob[0]), float(prob[1])],
            }), flush=True)

    if f: f.close()

if __name__ == "__main__":
    main()
