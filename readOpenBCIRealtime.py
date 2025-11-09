import time
import numpy as np
import threading
import queue
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

from csp import CSP
from spectral_feature_extractor import SpectralFeatureExtractor

window_size = 40
data_queue = queue.Queue(maxsize=10)


def realtime_ganglion_stream():
    # Ensure you replace 'COMX' with the actual serial port of your OpenBCI dongle
    # On Windows it's usually COM#, on Linux it's /dev/ttyUSB# or similar
    # On macOS it's usually /dev/tty.usbmodem# or similar
    
    serial_port = '/dev/cu.usbmodem11' # CHANGE THIS
    
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board_id = BoardIds.GANGLION_BOARD.value
    
    board = BoardShim(board_id, params)

    return board

def producer(board):
    # Load a fitted CSP object

    spectralFeatureExtractor = SpectralFeatureExtractor(sampling_rate=200)
    
    while True:
        time.sleep(1)

        data = board.get_current_board_data(window_size)
        
        # Get EEG Channels
        eeg_channels = BoardShim.get_eeg_channels(board.get_board_id())
        sampling_rate = board.get_sampling_rate(BoardIds.GANGLION_BOARD.value)
        

        # Filter out noise & bandpass.
        for ch in eeg_channels:
            DataFilter.perform_bandpass(data[ch], sampling_rate, 1.0, 50.0, 2, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.remove_environmental_noise(data[ch], sampling_rate, 0)
        
        # Convert to Epoch Shape (For OpenBCI)
        data = np.array([data[1:5]])

        # Feature Extraction
        # Need to load an already fitted CSP
        features = spectralFeatureExtractor.extract_features(data)
        asymmetry = spectralFeatureExtractor.extract_asymmetry(data, channel_pairs=[(0,1 ), (2, 3)])

        # Create feature Matrix

        # Pass to model
        # Need to load model from pickle


        try:
            data_queue.put_nowait(data)
        except queue.Full:
            # Skip oldest if processing is behind
            data_queue.get_nowait()
            data_queue.put_nowait(data)
            
            
        time.sleep(0.5)  # window interval

def consumer():
    while True:
        status = data_queue.get()
        start = time.time()

        # Show on GUI?
        
        # print(status)

        # print(f"Processing took {time.time() - start:.3f} s")


if __name__ == "__main__":
    board = realtime_ganglion_stream()
    board.prepare_session()
    board.start_stream()

    # Run threads separately
    threading.Thread(target=producer, daemon=True, args=(board,)).start()
    threading.Thread(target=consumer, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        board.stop_stream()
        board.release_session()
