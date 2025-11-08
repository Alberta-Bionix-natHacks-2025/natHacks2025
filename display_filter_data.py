import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

# Try to import WindowFunctions if available (newer BrainFlow)
try:
    from brainflow.data_filter import WindowFunctions
    WINDOW_AVAILABLE = True
except ImportError:
    WINDOW_AVAILABLE = False


def run_ganglion_realtime_plot():
    """
    Streams real-time EEG data from the OpenBCI Ganglion board
    and shows:
        - Top: individual EEG waveforms (1 per subplot)
        - Bottom: real-time bandpower bars (delta, theta, alpha, beta)
    """

    serial_port = 'COM8'  # ⚙️ Update this if needed

    # --- BrainFlow setup ---
    params = BrainFlowInputParams()
    params.serial_port = serial_port
    board_id = BoardIds.GANGLION_BOARD.value
    board = BoardShim(board_id, params)

    try:
        board.prepare_session()
        board.start_stream()
        print("✅ Streaming started. Close plot or press Ctrl+C to stop.")

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        n_channels = len(eeg_channels)
        window_size = sampling_rate * 5  # 5 seconds of data

        # --- Setup figure: one plot per EEG + one for bandpower ---
        fig, axes = plt.subplots(n_channels + 1, 1, figsize=(10, 10))
        plt.subplots_adjust(hspace=0.6)

        buffers = [np.zeros(window_size) for _ in range(n_channels)]
        colors = ['b', 'r', 'g', 'm']
        lines = []

        # EEG plots (top 4 subplots)
        for i, ch in enumerate(eeg_channels):
            
            
            ax = axes[i]
            ax.set_title(f"EEG Channel {ch}")
            ax.set_xlim(0, 600)
        
            if (i == 3 or i == 1):  # Channels 1 and 3 have larger amplitude
                ax.set_ylim(-500, 500)
            else:
                ax.set_ylim(-50, 50)
            
            ax.set_ylabel("µV")
            ax.set_xlabel("")
            line, = ax.plot([], [], lw=1, color=colors[i % len(colors)])
            lines.append(line)

        # Bandpower subplot (bottom)
        bp_ax = axes[-1]
        bp_ax.set_title("Average Bandpower Across Channels")
        bp_bands = ['Delta (1–4 Hz)', 'Theta (4–8 Hz)', 'Alpha (8–13 Hz)', 'Beta (13–30 Hz)']
        bar_positions = np.arange(len(bp_bands))
        bar_values = np.zeros(len(bp_bands))
        bars = bp_ax.bar(bar_positions, bar_values, color='skyblue')
        bp_ax.set_xticks(bar_positions)
        bp_ax.set_xticklabels(bp_bands)
        bp_ax.set_ylim(0, 1.2)
        bp_ax.set_ylabel("Normalized Power")

        # --- Update function for animation ---
        def update(frame):
            nonlocal buffers

            # Get ~1 sec of data
            data = board.get_current_board_data(sampling_rate)
            if data.shape[1] == 0:
                return lines + list(bars)

            # Update EEG traces
            for ch_idx, ch in enumerate(eeg_channels):
                eeg_data = data[ch, :]
                buffers[ch_idx] = np.append(buffers[ch_idx], eeg_data)[-window_size:]
                lines[ch_idx].set_data(np.arange(len(buffers[ch_idx])), buffers[ch_idx])

            # --- Bandpower computation ---
            all_bp = []
            for ch_idx, ch in enumerate(eeg_channels):
                ch_data = buffers[ch_idx]

                # Choose window mode depending on BrainFlow version
                window_mode = (
                    WindowFunctions.HANNING.value if WINDOW_AVAILABLE else 2
                )

                psd = DataFilter.get_psd_welch(
                    ch_data,
                    nfft=256,
                    overlap=128,
                    sampling_rate=sampling_rate,
                    window=window_mode
                )

                delta = DataFilter.get_band_power(psd, 1.0, 4.0)
                theta = DataFilter.get_band_power(psd, 4.0, 8.0)
                alpha = DataFilter.get_band_power(psd, 8.0, 13.0)
                beta = DataFilter.get_band_power(psd, 13.0, 30.0)

                all_bp.append([delta, theta, alpha, beta])

            mean_bp = np.mean(all_bp, axis=0)
            norm_bp = mean_bp / (np.sum(mean_bp) + 1e-8)

            for rect, h in zip(bars, norm_bp):
                rect.set_height(h)

            return lines + list(bars)

        # --- Start the animation ---
        ani = FuncAnimation(fig, update, interval=300, blit=True, cache_frame_data=False)
        plt.show()

        # Save data on close
        data = board.get_board_data()
        eeg_data = data[eeg_channels, :]
        DataFilter.write_file(data, 'ganglion_data.csv', 'w')
        print(f"💾 Data saved to ganglion_data.csv ({eeg_data.shape[1]} samples).")

    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if board.is_prepared():
            board.stop_stream()
            board.release_session()
            print("✅ Stream stopped and session released.")


if __name__ == "__main__":
    run_ganglion_realtime_plot()
