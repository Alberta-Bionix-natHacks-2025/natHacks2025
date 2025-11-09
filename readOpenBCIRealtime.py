import time
import numpy as np
import threading
import queue
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

window_size = 40
data_queue = queue.Queue(maxsize=10)


def realtime_ganglion_stream():
    # Ensure you replace 'COMX' with the actual serial port of your OpenBCI dongle
    # On Windows it's usually COM#, on Linux it's /dev/ttyUSB# or similar
    # On macOS it's usually /dev/tty.usbmodem# or similar
    
    serial_port = 'COM8' # CHANGE THIS
    
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board_id = BoardIds.GANGLION_BOARD.value
    
    board = BoardShim(board_id, params)

    return board

def producer(board):
    while True:
        data = board.get_current_board_data(window_size)

        #filtering example: bandpass 1-50 Hz
        DataFilter.perform_bandpass(data, board.get_sampling_rate(), 1.0, 50.0, 2, FilterTypes.BUTTERWORTH.value, 0)


        try:
            data_queue.put_nowait(data)
        except queue.Full:
            # Skip oldest if processing is behind
            data_queue.get_nowait()
            data_queue.put_nowait(data)
        time.sleep(0.5)  # window interval

def consumer():
    while True:
        data = data_queue.get()
        start = time.time()

        # Show on GUI?

        print(f"Processing took {time.time() - start:.3f} s")


if __name__ == "__main__":
    board = realtime_ganglion_stream()
    board.prepare_session()
    board.start_stream(45000)

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