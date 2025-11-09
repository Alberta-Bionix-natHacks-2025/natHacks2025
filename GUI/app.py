import time
import numpy as np
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

streaming = True

@app.route('/')
def index():
    return render_template('index.html')

def stream_eeg_data():
    """Continuously emit EEG values to the web client."""
    serial_port = '/dev/cu.usbmodem11'   # ← change this for your system
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board_id = BoardIds.GANGLION_BOARD.value

    board = BoardShim(board_id, params)

    try:
        board.prepare_session()
        board.start_stream()
        print("EEG stream started...")

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        time.sleep(3)

        temp = board.get_current_board_data(sampling_rate)
        eegtest = temp[eeg_channels[0], :].tolist()

        while streaming:
            data = board.get_current_board_data(sampling_rate)

            if data.shape[1] > 0:
                eeg_values = [data[ch, :].tolist() for ch in eeg_channels]
                print(eegtest)
                socketio.emit('eeg_update', {'values': eeg_values})
            socketio.sleep(0.2)

    except Exception as e:
        print("Error:", e)
    finally:
        board.stop_stream()
        board.release_session()
        print("EEG session ended.")

@socketio.on('connect')
def on_connect():
    print("Client connected")

if __name__ == '__main__':
    # thread = threading.Thread(target=stream_eeg_data, daemon=True)
    # thread.start()
    socketio.start_background_task(target=stream_eeg_data)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
