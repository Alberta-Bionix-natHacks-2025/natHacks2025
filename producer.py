# REPLACE their producer function with this:

def producer(board, bci_system):
    """
    Modified producer that integrates with trained BCI model
    """
    # Load models
    classifier = joblib.load('data/models/lda_classifier.pkl')
    csp = joblib.load('data/models/csp_model.pkl')
    
    while True:
        time.sleep(0.5)
        
        # Get data
        data = board.get_current_board_data(window_size)
        
        if data.shape[1] == 0:
            continue
        
        # Get EEG Channels
        eeg_channels = BoardShim.get_eeg_channels(board.get_board_id())
        sampling_rate = board.get_sampling_rate(BoardIds.GANGLION_BOARD.value)
        
        # Filter (match training: 8-30 Hz)
        for ch in eeg_channels[:3]:  # First 3 channels
            DataFilter.perform_bandpass(
                data[ch], sampling_rate, 8.0, 30.0, 4, 
                FilterTypes.BUTTERWORTH.value, 0
            )
            DataFilter.remove_environmental_noise(data[ch], sampling_rate, 0)
        
        # Convert to epoch format: (1, n_channels, n_samples)
        epoch = data[1:4].reshape(1, 3, -1)  # First 3 channels
        
        # Extract CSP features
        features = csp.transform(epoch)
        
        # Classify
        prediction = classifier.predict(features)[0]
        confidence = classifier.predict_proba(features)[0][prediction]
        
        # Display
        class_name = "LEFT HAND" if prediction == 0 else "RIGHT HAND"
        print(f"Prediction: {class_name} | Confidence: {confidence:.1%}")
        
        # Put in queue for GUI
        try:
            data_queue.put_nowait({
                'prediction': prediction,
                'confidence': confidence,
                'class_name': class_name
            })
        except queue.Full:
            data_queue.get_nowait()
            data_queue.put_nowait({...})