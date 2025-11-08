import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes


def run_ganglion_stream():
    # Ensure you replace 'COMX' with the actual serial port of your OpenBCI dongle
    # On Windows it's usually COM#, on Linux it's /dev/ttyUSB# or similar
    # On macOS it's usually /dev/tty.usbmodem# or similar
    
    serial_port = '/dev/cu.usbmodem11' # CHANGE THIS
    
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board_id = BoardIds.GANGLION_BOARD.value
    
    board = BoardShim(board_id, params)
    
    try:
        # 1. Prepare and start the session
        board.prepare_session()
        board.start_stream()
        print("Streaming started. Waiting for 10 seconds to collect data...")
        
        # 2. Wait to collect some data
        time.sleep(10)
        
        # 3. Get all data from the board's internal buffer
        # This gets all data collected since the stream started
        data = board.get_board_data()
        print(f"Collected {data.shape[1]} samples.")
        
        # 4. Extract EEG channels and apply basic signal processing
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        eeg_data = data[eeg_channels, :]
        
        # Example: Apply a bandpass filter (optional, but useful)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        # for ch in eeg_channels:
        #     DataFilter.perform_bandpass(data[ch], sampling_rate, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0)

        print("EEG data shape:", eeg_data.shape)

        # 5. You can now use `eeg_data` for analysis or save it
        DataFilter.write_file(data, 'ganglion_data.csv', 'w')
        print("Data saved to ganglion_data.csv")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 6. Stop stream and release the session
        if board.is_prepared():
            print("Stopping stream and releasing session.")
            board.stop_stream()
            board.release_session()


if __name__ == "__main__":
    run_ganglion_stream()
