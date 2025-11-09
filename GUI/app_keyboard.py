import eventlet
eventlet.monkey_patch()

import time
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

streaming = True
direction = "NEUTRAL"  # Global state

@app.route('/')
def index():
    return render_template('index_keyboard.html')


def simulate_eeg_data(num_channels=4, sampling_rate=200):
    """
    Generate synthetic EEG-like sine wave data with noise.
    """
    t = np.linspace(0, 1, sampling_rate)
    freqs = [10, 12, 8, 15]  # Different frequency per channel
    eeg_data = []

    for f in freqs:
        wave = 50 * np.sin(2 * np.pi * f * t) + np.random.normal(0, 10, len(t))
        eeg_data.append(wave.tolist())

    return eeg_data


def stream_eeg_data():
    """
    Continuously emit simulated EEG data and direction updates.
    """
    global direction
    print("Starting simulated EEG stream...")

    while streaming:
        eeg_values = simulate_eeg_data()
        socketio.emit('eeg_update', {'values': eeg_values})
        socketio.emit('direction_update', {'direction': direction})
        socketio.sleep(0.2)

    print("EEG simulation ended.")


@socketio.on('connect')
def on_connect():
    print("Client connected")
    socketio.emit('direction_update', {'direction': direction})


@socketio.on('direction_update')
def handle_direction(data):
    """
    Update direction from client button clicks.
    """
    global direction
    direction = data.get('direction', 'NEUTRAL')
    print("Direction updated to:", direction)


if __name__ == '__main__':
    socketio.start_background_task(stream_eeg_data)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
