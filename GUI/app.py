# GUI/app.py — diagnostic build to fix "stuck LEFT"

import eventlet
eventlet.monkey_patch()

from pathlib import Path
import sys, time, numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.exit_codes import BrainFlowError

# repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.return_prediction import predict_from_brainflow_block

# ---------- CONFIG ----------
SERIAL_PORT = "COM3"
BOARD_ID = BoardIds.GANGLION_BOARD.value
STREAMING = True

# Try one mapping, and if you see p≈1.00 forever, try another:
# our training order was ['C4','FC2','FC1','C3'] and runtime proxies are Fp2↔FC2, Fp1↔FC1.
RUNTIME_CHANNEL_NAMES = ["C4","Fp2","Fp1","C3"]      # baseline
# ALT_1_SWAP_C = ["C3","Fp2","Fp1","C4"]             # swap C3<->C4
# ALT_2_SWAP_FP = ["C4","Fp1","Fp2","C3"]            # swap Fp1<->Fp2
# ALT_3_BOTH = ["C3","Fp1","Fp2","C4"]               # swap both

# optional band-pass to match training
try:
    from scipy.signal import butter, sosfiltfilt
    def bandpass_8_30(block, fs, rows):
        sos = butter(4, [8/(fs/2), 30/(fs/2)], btype="band", output="sos")
        for r in rows:
            block[r, :] = sosfiltfilt(sos, block[r, :])
        return block
except Exception:
    def bandpass_8_30(block, fs, rows):  # no scipy → skip
        return block

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

@app.route("/")
def index():
    return render_template("index.html")

def _sim_loop():
    print("[SIM] starting simulated stream")
    fs = 200
    t = 0.0
    while STREAMING:
        t_arr = t + np.arange(2*fs)/fs
        vis = [
            np.sin(2*np.pi*10*t_arr),
            np.sin(2*np.pi*12*t_arr),
            np.sin(2*np.pi*15*t_arr),
            np.sin(2*np.pi*8*t_arr),
        ]
        socketio.emit("eeg_update", {"values":[v[-fs:].tolist() for v in vis]})
        socketio.emit("direction_update", {"direction":"NEUTRAL","probs":[0.5,0.5]})
        t += 1.0
        socketio.sleep(0.1)

def stream_eeg_data():
    params = BrainFlowInputParams()
    params.serial_port = SERIAL_PORT

    BoardShim.enable_dev_board_logger()
    board = BoardShim(BOARD_ID, params)

    try:
        board.prepare_session()
        board.start_stream()
        print(f"[EEG] Stream started on {SERIAL_PORT}")
        eeg_idx = BoardShim.get_eeg_channels(BOARD_ID)[:4]   # first 4 EEG rows
        fs = BoardShim.get_sampling_rate(BOARD_ID)
        print(f"[EEG] fs={fs} Hz | idx={eeg_idx} | mapping={RUNTIME_CHANNEL_NAMES}")

        time.sleep(1.0)  # warmup ring

        k = 0
        while STREAMING:
            # ---- UI: last 1 second
            last1 = board.get_current_board_data(fs)
            if last1.shape[1] > 0:
                socketio.emit("eeg_update", {
                    "values":[last1[i,:].astype(float).tolist() for i in eeg_idx]
                })

            # ---- MODEL: last 2 seconds, filter, then predict
            last2 = board.get_current_board_data(2*fs)
            if last2.shape[1] == 0:
                socketio.sleep(0.05)
                continue

            # diagnostics: per-channel stats BEFORE filtering
            raws = last2[eeg_idx, :]
            rms = np.sqrt(np.mean(raws**2, axis=1))
            std = raws.std(axis=1)
            if k % 10 == 0:
                print(f"[diag] map={RUNTIME_CHANNEL_NAMES} | rms={np.round(rms,1)} | std={np.round(std,1)}")

            last2_filt = bandpass_8_30(last2.copy(), fs, eeg_idx)

            pred_code, label, p_left, p_right = predict_from_brainflow_block(
                board_data=last2_filt,
                eeg_channel_indices=eeg_idx,
                fs=fs,
                channel_names=RUNTIME_CHANNEL_NAMES,
                neutral_margin=0.25,   # slightly larger neutral gate in live data
            )
            socketio.emit("direction_update", {
                "direction": label.upper(),
                "probs": [float(p_left), float(p_right)]
            })

            if k % 10 == 0:
                print(f"[pred] {label:7s}  pL={p_left:.2f}  pR={p_right:.2f}")
            k += 1

            socketio.sleep(0.1)

    except BrainFlowError as e:
        print("[EEG] BrainFlowError:", repr(e))
        _sim_loop()
    except Exception:
        import traceback
        print("[EEG] Streamer error:\n"+traceback.format_exc())
        _sim_loop()
    finally:
        try: board.stop_stream()
        except Exception: pass
        try: board.release_session()
        except Exception: pass
        print("[EEG] Session ended.")

@socketio.on("connect")
def on_connect():
    print("Client connected")

if __name__ == "__main__":
    socketio.start_background_task(target=stream_eeg_data)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
