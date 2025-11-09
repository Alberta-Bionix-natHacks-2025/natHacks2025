"""Simple spectral (PSD) and asymmetry features for EEG."""
import numpy as np
from scipy import signal
from scipy.integrate import simpson


class SpectralFeatureExtractor:
    """
    Extracts band powers (mu, beta by default) and asymmetry features.

    Parameters
    ----------
    sampling_rate : float
        Sampling frequency in Hz.
    mu_band : tuple(float, float)
        Mu-band limits.
    beta_band : tuple(float, float)
        Beta-band limits.
    welch_nperseg : int or None
        Segment length for Welch. If None, uses min(256, n_times).
    """

    def __init__(self, sampling_rate=250, mu_band=(8, 12), beta_band=(13, 30),
                 welch_nperseg=None):
        self.fs = float(sampling_rate)
        self.mu_band = mu_band
        self.beta_band = beta_band
        self.welch_nperseg = welch_nperseg

    # -------------------------- PSD / band power ------------------------------

    def compute_psd(self, data):
        """
        Welch PSD.

        Parameters
        ----------
        data : ndarray
            (n_channels, n_times) or (n_times,)

        Returns
        -------
        freqs : ndarray
        psd   : ndarray aligned with `data` shape
        """
        data = np.asarray(data)
        nperseg = self.welch_nperseg or min(256, data.shape[-1])
        freqs, psd = signal.welch(data, fs=self.fs, nperseg=nperseg, axis=-1)
        return freqs, psd

    def extract_band_power(self, data, freq_band):
        """
        Integrate PSD within a frequency band.

        Parameters
        ----------
        data : ndarray
            (n_channels, n_times) or (n_times,)
        freq_band : (low, high)

        Returns
        -------
        band_power : ndarray
            (n_channels,)
        """
        freqs, psd = self.compute_psd(data)
        lo, hi = freq_band
        idx = (freqs >= lo) & (freqs <= hi)
        # integrate along frequency axis
        bp = simpson(psd[..., idx], freqs[idx], axis=-1)
        return bp

    # --------------------------- feature stacks -------------------------------

    def extract_features(self, epochs_data):
        """
        Per-epoch band powers (mu+beta) for each channel.

        Parameters
        ----------
        epochs_data : ndarray
            (n_epochs, n_channels, n_times)

        Returns
        -------
        feats : ndarray
            (n_epochs, n_channels*2) as [mu_ch1..mu_chC, beta_ch1..beta_chC]
        """
        X = np.asarray(epochs_data)
        n_ep, n_ch, _ = X.shape
        feats = np.empty((n_ep, n_ch * 2), dtype=np.float64)
        for i, epoch in enumerate(X):
            mu = self.extract_band_power(epoch, self.mu_band)
            be = self.extract_band_power(epoch, self.beta_band)
            feats[i] = np.concatenate([mu, be])
        return feats

    def extract_asymmetry(self, epochs_data, channel_pairs=((0, 1),)):
        """
        Asymmetry (A-B)/(A+B) for mu and beta, per channel pair.

        Parameters
        ----------
        epochs_data : ndarray
            (n_epochs, n_channels, n_times)
        channel_pairs : iterable of tuple(int, int)
            Channel index pairs (A,B). Example: [(C3_idx, C4_idx)]

        Returns
        -------
        feats : ndarray
            (n_epochs, 2 * len(channel_pairs))
            order per pair: [mu_asym, beta_asym]
        """
        X = np.asarray(epochs_data)
        n_ep = X.shape[0]
        n_pairs = len(channel_pairs)
        feats = np.empty((n_ep, n_pairs * 2), dtype=np.float64)

        tiny = 1e-10
        for i, epoch in enumerate(X):
            vals = []
            for a, b in channel_pairs:
                mu_a = self.extract_band_power(epoch[a:a+1], self.mu_band)[0]
                mu_b = self.extract_band_power(epoch[b:b+1], self.mu_band)[0]
                mu_asym = (mu_a - mu_b) / (mu_a + mu_b + tiny)

                be_a = self.extract_band_power(epoch[a:a+1], self.beta_band)[0]
                be_b = self.extract_band_power(epoch[b:b+1], self.beta_band)[0]
                be_asym = (be_a - be_b) / (be_a + be_b + tiny)

                vals.extend([mu_asym, be_asym])
            feats[i] = vals
        return feats


# Quick self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((10, 4, 500))
    spec = SpectralFeatureExtractor(sampling_rate=250)
    f = spec.extract_features(dummy)
    a = spec.extract_asymmetry(dummy, channel_pairs=[(2, 3)])
    print("spectral:", f.shape, "asym:", a.shape)
