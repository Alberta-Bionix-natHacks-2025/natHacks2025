'''How much electrical activity exists in specific frequency ranges'''
import numpy as np
from scipy import signal
from scipy.integrate import simpson

class SpectralFeatureExtractor:
    """Extract power spectral density features from EEG"""
    
    def __init__(self, sampling_rate=160, mu_band=(8, 12), beta_band=(13, 30)):
        """
        Parameters:
        - sampling_rate: Hz (160 for PhysioNet)
        - mu_band: Frequency range for mu rhythm
        - beta_band: Frequency range for beta rhythm
        """
        self.fs = sampling_rate
        self.mu_band = mu_band
        self.beta_band = beta_band
    
    def compute_psd(self, data):
        """
        Compute Power Spectral Density using Welch's method
        
        Input: data (n_channels, n_timepoints) or (n_timepoints,)
        Output: freqs, psd
        """
        freqs, psd = signal.welch(
            data,
            fs=self.fs,
            nperseg=min(256, data.shape[-1]),  # Window size
            axis=-1
        )
        return freqs, psd
    
    def extract_band_power(self, data, freq_band):
        """
        Extract total power in a frequency band
        
        Input: data (n_channels, n_timepoints)
        Output: band_power (n_channels,)
        """
        freqs, psd = self.compute_psd(data)
        
        # Find frequency indices for this band
        idx_band = np.logical_and(freqs >= freq_band[0], freqs <= freq_band[1])
        
        # Integrate power using Simpson's rule
        band_power = simpson(psd[..., idx_band], freqs[idx_band], axis=-1)
        
        return band_power
    
    def extract_features(self, epochs_data):
        """
        Extract spectral features from all epochs
        
        Input: epochs_data (n_epochs, n_channels, n_timepoints)
        Output: features (n_epochs, n_channels * 2)
                        [mu_ch1, mu_ch2, ..., beta_ch1, beta_ch2, ...]
        """
        n_epochs, n_channels, n_times = epochs_data.shape
        features = []
        
        for epoch in epochs_data:
            epoch_features = []
            
            # Extract mu and beta power for each channel
            mu_power = self.extract_band_power(epoch, self.mu_band)
            beta_power = self.extract_band_power(epoch, self.beta_band)
            
            # Flatten: [mu_ch1, mu_ch2, mu_ch3, beta_ch1, beta_ch2, beta_ch3]
            epoch_features = np.concatenate([mu_power, beta_power])
            features.append(epoch_features)
        
        return np.array(features)
    
    def extract_asymmetry(self, epochs_data, channel_pairs=[(0, 1)]):
        """
        Extract asymmetry features between channel pairs
        
        Input: 
        - epochs_data (n_epochs, n_channels, n_timepoints)
        - channel_pairs: [(ch1_idx, ch2_idx), ...] e.g., [(0, 1)] for C3-C4
        
        Output: asymmetry features (n_epochs, len(channel_pairs) * 2)
        """
        n_epochs = epochs_data.shape[0]
        features = []
        
        for epoch in epochs_data:
            epoch_features = []
            
            for ch1_idx, ch2_idx in channel_pairs:
                # Mu band asymmetry
                mu_ch1 = self.extract_band_power(epoch[ch1_idx:ch1_idx+1], self.mu_band)[0]
                mu_ch2 = self.extract_band_power(epoch[ch2_idx:ch2_idx+1], self.mu_band)[0]
                mu_asymmetry = (mu_ch1 - mu_ch2) / (mu_ch1 + mu_ch2 + 1e-10)
                
                # Beta band asymmetry
                beta_ch1 = self.extract_band_power(epoch[ch1_idx:ch1_idx+1], self.beta_band)[0]
                beta_ch2 = self.extract_band_power(epoch[ch2_idx:ch2_idx+1], self.beta_band)[0]
                beta_asymmetry = (beta_ch1 - beta_ch2) / (beta_ch1 + beta_ch2 + 1e-10)
                
                epoch_features.extend([mu_asymmetry, beta_asymmetry])
            
            features.append(epoch_features)
        
        return np.array(features)


# Quick test
if __name__ == '__main__':
    # Test with dummy data
    dummy_data = np.random.randn(10, 3, 480)  # 10 epochs, 3 channels, 480 timepoints
    
    extractor = SpectralFeatureExtractor(sampling_rate=160)
    
    # Test spectral features
    features = extractor.extract_features(dummy_data)
    print(f"Spectral features shape: {features.shape}")  # Should be (10, 6)
    
    # Test asymmetry
    asymmetry = extractor.extract_asymmetry(dummy_data, channel_pairs=[(0, 1)])
    print(f"Asymmetry features shape: {asymmetry.shape}")  # Should be (10, 2)