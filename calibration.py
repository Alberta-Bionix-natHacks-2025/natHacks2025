"""
CALIBRATE MODEL FOR NEW USER
Collect 2-3 minutes of data with OpenBCI and retrain
"""

import numpy as np
import joblib
from openbci_bci_integration import OpenBCIMotorImageryBCI

def collect_calibration_data(duration=120):  # 2 minutes
    """Collect labeled data for calibration"""
    
    print("CALIBRATION MODE")
    print("="*60)
    print("We'll collect data while you imagine movements")
    print(f"Total time: {duration} seconds")
    print()
    
    # Initialize board (without classification)
    bci = OpenBCIMotorImageryBCI(
        model_path='data/models/lda_classifier.pkl',
        csp_path='data/models/csp_model.pkl'
    )
    bci.setup_board()
    
    collected_data = []
    collected_labels = []
    
    # Alternate between left and right
    trial_duration = 10  # 10 seconds per trial
    n_trials = duration // trial_duration
    
    for trial in range(n_trials):
        label = trial % 2  # Alternate: 0 (left), 1 (right)
        class_name = "LEFT HAND" if label == 0 else "RIGHT HAND"
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"Imagine moving: {class_name}")
        print("Starting in 3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("GO! Imagine the movement NOW!")
        
        # Collect data for this trial
        trial_data = []
        start_time = time.time()
        
        while time.time() - start_time < trial_duration:
            data = bci.board.get_current_board_data(40)
            if data.shape[1] > 0:
                eeg_data = data[BoardShim.get_eeg_channels(bci.board_id), :]
                trial_data.append(eeg_data[0:3, :])  # First 3 channels
            time.sleep(0.1)
        
        # Concatenate and store
        trial_data = np.concatenate(trial_data, axis=1)
        collected_data.append(trial_data)
        collected_labels.append(label)
        
        print("Trial complete! Rest for 3 seconds...")
        time.sleep(3)
    
    bci.stop()
    
    # Save calibration data
    np.save('calibration_data.npy', collected_data)
    np.save('calibration_labels.npy', collected_labels)
    
    print("\n✓ Calibration data collected!")
    print(f"  Trials: {len(collected_data)}")
    print(f"  Saved to: calibration_data.npy")
    
    return collected_data, collected_labels

# Run calibration
if __name__ == "__main__":
    collect_calibration_data(duration=120)