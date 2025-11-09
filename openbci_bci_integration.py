"""
OPENBCI GANGLION + TRAINED BCI MODEL INTEGRATION
Real-time motor imagery classification using OpenBCI Ganglion
"""

import time
import numpy as np
import threading
import queue
import joblib
from scipy import signal
from collections import deque

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes


class OpenBCIMotorImageryBCI:
    """
    Complete BCI system integrating OpenBCI Ganglion with trained model
    """
    
    def __init__(self, model_path, csp_path, serial_port='/dev/cu.usbmodem11'):
        """
        Initialize the BCI system
        
        Args:
            model_path: Path to trained LDA classifier (.pkl)
            csp_path: Path to trained CSP model (.pkl)
            serial_port: Serial port for OpenBCI Ganglion
        """
        # Load trained models
        print("Loading trained models...")
        self.classifier = joblib.load(model_path)
        self.csp = joblib.load(csp_path)
        print("✓ Models loaded successfully")
        
        # OpenBCI setup
        self.serial_port = serial_port
        self.board = None
        self.board_id = BoardIds.GANGLION_BOARD.value
        
        # BCI parameters
        self.sampling_rate = 200  # Ganglion samples at 200 Hz
        self.window_duration = 3.0  # 3 second classification window
        self.window_samples = int(self.sampling_rate * self.window_duration)
        
        # Channel mapping
        # OpenBCI Ganglion has 4 channels
        # Map them to approximate motor cortex locations
        # Adjust based on your electrode placement!
        self.channel_map = {
            'C3': 0,   # Channel 1 -> Left motor cortex
            'C4': 1,   # Channel 2 -> Right motor cortex
            'Cz': 2,   # Channel 3 -> Central
        }
        self.n_channels = 3
        
        # Filtering parameters (match training: 8-30 Hz)
        self.lowcut = 8.0
        self.highcut = 30.0
        
        # Data buffer for sliding window
        self.data_buffer = deque(maxlen=self.window_samples)
        
        # Results
        self.current_prediction = None
        self.current_confidence = None
        self.is_running = False
        
        # Thread-safe queue
        self.data_queue = queue.Queue(maxsize=10)
        
        print(f"✓ BCI initialized")
        print(f"  Sampling rate: {self.sampling_rate} Hz")
        print(f"  Window size: {self.window_duration}s ({self.window_samples} samples)")
        print(f"  Channels: {list(self.channel_map.keys())}")
    
    def setup_board(self):
        """Initialize OpenBCI board"""
        print(f"\nConnecting to OpenBCI Ganglion on {self.serial_port}...")
        
        params = BrainFlowInputParams()
        params.serial_port = self.serial_port
        
        self.board = BoardShim(self.board_id, params)
        
        try:
            self.board.prepare_session()
            self.board.start_stream()
            print("✓ OpenBCI streaming started")
            return True
        except Exception as e:
            print(f"✗ Error connecting to OpenBCI: {e}")
            return False
    
    def preprocess_epoch(self, data):
        """
        Preprocess data to match training pipeline
        
        Args:
            data: Array of shape (n_channels, n_samples)
        
        Returns:
            Preprocessed epoch of shape (1, n_channels, n_samples)
        """
        # Apply bandpass filter (8-30 Hz) to match training
        filtered_data = np.zeros_like(data)
        
        nyquist = self.sampling_rate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        for i in range(data.shape[0]):
            filtered_data[i] = signal.filtfilt(b, a, data[i])
        
        # Reshape to epoch format: (1, n_channels, n_samples)
        epoch = filtered_data.reshape(1, self.n_channels, -1)
        
        return epoch
    
    def classify_window(self, data):
        """
        Classify current window of data
        
        Args:
            data: Array of shape (n_channels, n_samples)
        
        Returns:
            prediction: 0 (left) or 1 (right)
            confidence: Probability of predicted class
        """
        # Preprocess
        epoch = self.preprocess_epoch(data)
        
        # Extract CSP features
        features = self.csp.transform(epoch)
        
        # Classify
        prediction = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        confidence = probabilities[prediction]
        
        return int(prediction), float(confidence)
    
    def producer(self):
        """
        Producer thread: Continuously fetch data from OpenBCI
        """
        print("\n🔴 Producer started - fetching data from OpenBCI...")
        
        # Get board info
        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        
        while self.is_running:
            try:
                # Fetch recent data (40 samples = 0.2 seconds at 200 Hz)
                data = self.board.get_current_board_data(40)
                
                if data.shape[1] == 0:
                    time.sleep(0.05)
                    continue
                
                # Extract EEG channels
                eeg_data = data[eeg_channels, :]
                
                # Apply BrainFlow's built-in filtering for noise reduction
                for ch_idx in range(eeg_data.shape[0]):
                    # Remove 60 Hz power line noise
                    DataFilter.remove_environmental_noise(
                        eeg_data[ch_idx], 
                        self.sampling_rate, 
                        0  # 0 = 60 Hz, 1 = 50 Hz
                    )
                
                # Select channels matching your training (C3, C4, Cz)
                # NOTE: Adjust indices based on your electrode placement!
                selected_channels = eeg_data[0:3, :]  # First 3 channels
                
                # Add samples to buffer
                for i in range(selected_channels.shape[1]):
                    sample = selected_channels[:, i]
                    self.data_buffer.append(sample)
                
                # Put data in queue for consumer (for visualization, etc.)
                try:
                    self.data_queue.put_nowait(selected_channels)
                except queue.Full:
                    # Skip if queue is full
                    pass
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                print(f"Error in producer: {e}")
                time.sleep(0.1)
    
    def consumer(self):
        """
        Consumer thread: Classify data and display results
        """
        print("🤖 Consumer started - classifying motor imagery...\n")
        
        classification_interval = 0.5  # Classify every 0.5 seconds
        last_classification = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Classify periodically
                if current_time - last_classification >= classification_interval:
                    # Check if we have enough data
                    if len(self.data_buffer) >= self.window_samples:
                        # Get window from buffer
                        window_data = np.array(list(self.data_buffer))
                        # Transpose to (n_channels, n_samples)
                        window_data = window_data.T
                        
                        # Classify
                        prediction, confidence = self.classify_window(window_data)
                        
                        # Store results
                        self.current_prediction = prediction
                        self.current_confidence = confidence
                        
                        # Display
                        class_name = "LEFT HAND" if prediction == 0 else "RIGHT HAND"
                        confidence_bar = "█" * int(confidence * 20)
                        
                        print(f"[{time.strftime('%H:%M:%S')}] "
                              f"Prediction: {class_name:>11} | "
                              f"Confidence: {confidence:.1%} | "
                              f"{confidence_bar}")
                        
                        # High confidence alert
                        if confidence > 0.75:
                            print(f"  → HIGH CONFIDENCE {class_name}! ⚡")
                        
                        last_classification = current_time
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in consumer: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start the BCI system"""
        # Setup OpenBCI board
        if not self.setup_board():
            return False
        
        # Wait for initial data collection
        print("\nCollecting initial data (3 seconds)...")
        time.sleep(3)
        
        # Start threads
        self.is_running = True
        
        self.producer_thread = threading.Thread(target=self.producer, daemon=True)
        self.consumer_thread = threading.Thread(target=self.consumer, daemon=True)
        
        self.producer_thread.start()
        self.consumer_thread.start()
        
        print("\n" + "="*60)
        print("MOTOR IMAGERY BCI - RUNNING")
        print("="*60)
        print("Imagine moving your LEFT or RIGHT hand")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        return True
    
    def stop(self):
        """Stop the BCI system"""
        print("\n\nStopping BCI system...")
        self.is_running = False
        
        # Wait for threads to finish
        if hasattr(self, 'producer_thread'):
            self.producer_thread.join(timeout=2)
        if hasattr(self, 'consumer_thread'):
            self.consumer_thread.join(timeout=2)
        
        # Stop OpenBCI
        if self.board and self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()
            print("✓ OpenBCI disconnected")
        
        print("✓ BCI system stopped")
    
    def get_latest_prediction(self):
        """Get most recent prediction"""
        return self.current_prediction, self.current_confidence


# =================================================================
# MAIN EXECUTION
# =================================================================

def main():
    """Main function to run the BCI system"""
    
    print("="*60)
    print("OPENBCI MOTOR IMAGERY BCI")
    print("="*60)
    
    # Initialize BCI with your trained models
    bci = OpenBCIMotorImageryBCI(
        model_path='data/models/lda_classifier.pkl',
        csp_path='data/models/csp_model.pkl',
        serial_port='/dev/cu.usbmodem11'  # CHANGE THIS to your port
    )
    
    # Start the system
    if bci.start():
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            bci.stop()
    else:
        print("Failed to start BCI system")


if __name__ == "__main__":
    main()