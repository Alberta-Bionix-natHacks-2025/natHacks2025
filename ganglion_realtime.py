import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter


def run_ganglion_realtime_plot():
    """
    Streams real-time EEG data from the OpenBCI Ganglion board
    and plots it continuously using matplotlib.
    """

    # Replace 'COM5' with the actual port where your Ganglion dongle is connected:
    # - Windows: 'COM5', 'COM3', etc.
    # - macOS: '/dev/tty.usbmodemXXXX'
    # - Linux: '/dev/ttyUSB0' or '/dev/ttyACM0'
    serial_port = 'COM8'

    # 2. CONFIGURE BRAINFLOW CONNECTION
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board_id = BoardIds.GANGLION_BOARD.value

    # Create board instance
    board = BoardShim(board_id, params)

    try:
        # 3. PREPARE AND START STREAMING
        board.prepare_session()
        board.start_stream()
        print("Streaming started. Close plot or press Ctrl+C to stop.")

        # 4. GET BOARD INFO
        eeg_channels = BoardShim.get_eeg_channels(board_id)   # EEG channel indices
        sampling_rate = BoardShim.get_sampling_rate(board_id) # Typically 200 Hz for Ganglion

        # 5. SETUP MATPLOTLIB FIGURE
        fig, ax = plt.subplots()
        ax.set_title("Real-Time EEG (Channel 1)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Microvolts (μV)")
        ax.set_ylim(-60, 60)   # Adjust to match expected signal range
        ax.set_xlim(0, sampling_rate * 5)  # Show 5 seconds of data at a time

        # Create an empty line that we'll update with live data
        line, = ax.plot([], [], lw=1, color='b')

        # Define a fixed-size circular buffer for the rolling EEG window
        window_size = sampling_rate * 5   # 5 seconds of data
        buffer = np.zeros(window_size)

        # 6. DEFINE ANIMATION UPDATE FUNCTION
        def update(frame):
            """
            Called repeatedly by FuncAnimation (~5 times/sec).
            Grabs new EEG samples and updates the plot.
            """
            nonlocal buffer

            # Fetch ~1 second of the newest data from the board’s ring buffer
            data = board.get_current_board_data(sampling_rate)

            if data.shape[1] > 0:
                # Extract the first EEG channel’s data
                eeg_data = data[eeg_channels[0], :]

                # Append to buffer and keep only the last N samples
                buffer = np.append(buffer, eeg_data)[-window_size:]

                # Update the plotted data
                line.set_data(np.arange(len(buffer)), buffer)

            return line,

        # 7. RUN LIVE PLOT LOOP
        # This updates the plot every 200 ms (~5 FPS)
        ani = FuncAnimation(fig, update, interval=200, blit=True)

        plt.tight_layout()
        plt.show()
        
        
        # GET DATA AND SAVE IT
        
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

    except KeyboardInterrupt:
        # Gracefully handle Ctrl+C
        print("Interrupted by user.")
    except Exception as e:
        # Print any other error
        print(f"Error: {e}")
    finally:
        # 8. CLEANUP: STOP STREAM AND RELEASE SESSION
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
            print("Stream stopped and session released.")


# 9. MAIN ENTRY POINT
if __name__ == "__main__":
    run_ganglion_realtime_plot()
