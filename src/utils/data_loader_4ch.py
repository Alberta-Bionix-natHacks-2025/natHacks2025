# src/utils/data_loader_4ch.py
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import StandardScaler

from src.feature_extraction.csp import CSP
from src.feature_extraction.spectral_features import SpectralFeatureExtractor


class EEGFeatureLoader4Ch:
    """
    4-ch loader using BNCI2014-001 with channel proxies for your hardware.

    Hardware order at runtime: [C4, Fp2, Fp1, C3]
    BNCI proxies for training: C4->C4, Fp2->FC2, Fp1->FC1, C3->C3
    """
    BNCI_TRAIN_CHANNELS = ['C4', 'FC2', 'FC1', 'C3']   # train order == pipeline order
    HW_CHANNELS = ['C4', 'Fp2', 'Fp1', 'C3']          # what your device streams

    def __init__(self,
                 tmin=0.5, tmax=2.5, fmin=8, fmax=30,
                 n_csp_components=2,
                 asym_pairs=(('C3', 'C4'), ('Fp1', 'Fp2')),
                 subjects=(1,)):
        # normalize subjects to list
        if isinstance(subjects, int):
            subjects = [subjects]
        self.subjects = list(subjects)

        self.tmin, self.tmax = float(tmin), float(tmax)
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.n_csp_components = int(n_csp_components)
        self.window_samples = 500  # enforce 500 samples (2s @ 250 Hz)

        self.spec = SpectralFeatureExtractor(
            sampling_rate=250, mu_band=(8, 12), beta_band=(13, 30)
        )

        # map asymmetry pairs to the BNCI proxies
        proxy_map = {'Fp1': 'FC1', 'Fp2': 'FC2', 'C3': 'C3', 'C4': 'C4'}
        self.asym_pairs_bnci = tuple((proxy_map[a], proxy_map[b]) for (a, b) in asym_pairs)

    def load_dataset(self):
        ds = BNCI2014_001()
        # Ask explicitly for Left/Right only to avoid feet/tongue in labels
        p = MotorImagery(
            n_classes=2,
            events=['left_hand', 'right_hand'],
            channels=self.BNCI_TRAIN_CHANNELS,
            tmin=self.tmin, tmax=self.tmax,
            fmin=self.fmin, fmax=self.fmax
        )

        X, y, meta = p.get_data(dataset=ds, subjects=self.subjects)
        # X: (N, 4, ~500), y: may be strings or ints depending on MOABB version

        # --- Robust label mapping to {left:0, right:1} ---
        y = np.asarray(y)
        if y.dtype.kind in {'U', 'S', 'O'}:
            lbl_map = {'left_hand': 0, 'right_hand': 1}
            mask = np.isin(y, list(lbl_map.keys()))
            X = X[mask]
            y = np.vectorize(lbl_map.get)(y[mask]).astype(int)
        else:
            uniq = np.unique(y)
            if set(uniq) <= {0, 1}:
                y = y.astype(int)
            else:
                # fallback: map sorted uniques to 0/1
                uniq_sorted = np.sort(uniq)
                mapping = {uniq_sorted[0]: 0, uniq_sorted[-1]: 1}
                y = np.vectorize(mapping.get)(y).astype(int)

        # --- Force exactly 500 samples ---
        S = X.shape[-1]
        if S > self.window_samples:
            X = X[..., :self.window_samples]
        elif S < self.window_samples:
            pad = np.zeros((X.shape[0], X.shape[1], self.window_samples - S), dtype=X.dtype)
            X = np.concatenate([X, pad], axis=-1)

        print(f"[4ch] Loaded {X.shape[0]} trials | chans={self.BNCI_TRAIN_CHANNELS} | samples={X.shape[-1]}")
        # Sanity
        # print("Label counts:", {int(v): int((y==v).sum()) for v in np.unique(y)})
        return X.astype(np.float32), y.astype(int), meta

    def extract_features(self, X, y):
        # 1) CSP (2 comps for 4-ch)
        csp = CSP(n_components=self.n_csp_components)
        csp_feat = csp.fit_transform(X, y)                      # (N, 2)

        # 2) PSD (mu+beta per channel) -> (N, 8)
        psd_feat = self.spec.extract_features(X)                # (N, 8)

        # 3) Asymmetry on BNCI proxy indices
        name2idx = {nm: i for i, nm in enumerate(self.BNCI_TRAIN_CHANNELS)}
        pairs_idx = [(name2idx[a], name2idx[b]) for (a, b) in self.asym_pairs_bnci]
        asym_feat = self.spec.extract_asymmetry(X, channel_pairs=pairs_idx)  # (N, 4)

        feats = np.concatenate([csp_feat, psd_feat, asym_feat], axis=1)      # (N, 14)

        scaler = StandardScaler()
        feats = scaler.fit_transform(feats)
        return feats.astype(np.float32), y, scaler

    def load_features(self):
        X, y, _ = self.load_dataset()
        return self.extract_features(X, y)
