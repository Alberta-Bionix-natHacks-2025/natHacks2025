import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


def run_ganglion_realtime_plot():
    """
    Streams real-time EEG data from the OpenBCI Ganglion board
    and shows:
        - Top: real-time EEG waveforms (all 4 channels)
        - Bottom: real-time bandpower bars (delta, theta, alpha, beta)
    """

    # ⚙️ Adjust this for your actual device port:
    serial_port = 'COM5'

    # --- BrainFlow setup ---
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board_id = BoardIds.GANGLION_BOARD.value
    board = BoardShim(board_id, params)

    try:
        board.prepare_session()
        board.start_stream()
        print("Streaming started. Close plot or press Ctrl+C to stop.")

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        n_channels = len(eeg_channels)
        window_size = sampling_rate * 5  # 5 seconds of data

        # --- Setup figure with 2 rows: EEG + Bandpower ---
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        plt.subplots_adjust(hspace=0.5)

        # 🧩 Subplot 1: Real-time EEG (4 channels stacked)
        eeg_ax = axes[0]
        eeg_ax.set_title("Real-Time EEG (All Channels)")
        eeg_ax.set_xlabel("Samples")
        eeg_ax.set_ylabel("µV")
        eeg_ax.set_xlim(0, window_size)
        eeg_ax.set_ylim(-200, 200)
        lines = []
        buffers = [np.zeros(window_size) for _ in range(n_channels)]
        colors = ['b', 'r', 'g', 'm']

        for i in range(n_channels):
            line, = eeg_ax.plot([], [], lw=1, color=colors[i], label=f"Ch {eeg_channels[i]}")
            lines.append(line)
        eeg_ax.legend(loc="upper right")

        # ⚡ Subplot 2: Bandpower bar plot
        bp_ax = axes[1]
        bp_ax.set_title("Real-Time Bandpower (Averaged Across Channels)")
        bp_bands = ['Delta (1–4 Hz)', 'Theta (4–8 Hz)', 'Alpha (8–13 Hz)', 'Beta (13–30 Hz)']
        bar_positions = np.arange(len(bp_bands))
        bar_values = np.zeros(len(bp_bands))
        bars = bp_ax.bar(bar_positions, bar_values, color='skyblue')
        bp_ax.set_xticks(bar_positions)
        bp_ax.set_xticklabels(bp_bands)
        bp_ax.set_ylim(0, 1)  # normalized
        bp_ax.set_ylabel("Normalized Power")

        # --- Define update function for animation ---
        def update(frame):
            nonlocal buffers

            # Get latest ~1 sec of data
            data = board.get_current_board_data(sampling_rate)
            if data.shape[1] == 0:
                return lines + list(bars)

            # Update EEG plots
            for ch_idx, ch in enumerate(eeg_channels):
                eeg_data = data[ch, :]
                buffers[ch_idx] = np.append(buffers[ch_idx], eeg_data)[-window_size:]
                lines[ch_idx].set_data(np.arange(len(buffers[ch_idx])), buffers[ch_idx])

            # Compute bandpower from all EEG channels combined
            # Use 5 seconds of most recent buffer for spectral analysis
            all_bp = []
            for ch_idx, ch in enumerate(eeg_channels):
                ch_data = buffers[ch_idx]
                delta = DataFilter.get_band_power(ch_data, sampling_rate, 1.0, 4.0)
                theta = DataFilter.get_band_power(ch_data, sampling_rate, 4.0, 8.0)
                alpha = DataFilter.get_band_power(ch_data, sampling_rate, 8.0, 13.0)
                beta = DataFilter.get_band_power(ch_data, sampling_rate, 13.0, 30.0)
                all_bp.append([delta, theta, alpha, beta])

            # Average across channels
            mean_bp = np.mean(all_bp, axis=0)
            norm_bp = mean_bp / np.sum(mean_bp)  # normalize 0–1
            for rect, h in zip(bars, norm_bp):
                rect.set_height(h)

            return lines + list(bars)

        # --- Run the animation ---
        ani = FuncAnimation(fig, update, interval=300, blit=True)
        plt.show()

        # After closing plot, save collected data
        data = board.get_board_data()
        eeg_data = data[eeg_channels, :]
        DataFilter.write_file(data, 'ganglion_data.csv', 'w')
        print(f"Data saved to ganglion_data.csv ({eeg_data.shape[1]} samples).")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
            print("Stream stopped and session released.")


if __name__ == "__main__":
    run_ganglion_realtime_plot()
