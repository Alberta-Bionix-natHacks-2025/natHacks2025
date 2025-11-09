# src/utils/data_loader_4ch.py
import numpy as np
from math import gcd
from scipy.signal import resample_poly
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from src.feature_extraction.csp import CSP
from src.feature_extraction.spectral_features import SpectralFeatureExtractor


class EEGFeatureLoader4Ch:
    """
    4-ch training on BNCI2014-001 using channel proxies for your hardware.
    Hardware order at runtime: [C4, Fp2, Fp1, C3]
    BNCI proxies used here:    [C4, FC2, FC1, C3]
    """

    BNCI_TRAIN_CHANNELS = ['C4', 'FC2', 'FC1', 'C3']
    HW_CHANNELS = ['C4', 'Fp2', 'Fp1', 'C3']

    def __init__(self,
                 tmin=0.0, tmax=2.0, fmin=8, fmax=30,
                 n_csp_components=2,
                 asym_pairs=(('C3','C4'), ('Fp1','Fp2')),
                 subjects=(1,),
                 target_fs=200):               # <<< NEW: train at 200 Hz
        if isinstance(subjects, int):
            subjects = [subjects]
        self.subjects = list(subjects)

        self.tmin, self.tmax = float(tmin), float(tmax)
        self.fmin, self.fmax = float(fmin), float(fmax)
        self.n_csp_components = int(n_csp_components)
        self.target_fs = int(target_fs)       # 200

        self.spec = SpectralFeatureExtractor(sampling_rate=self.target_fs,
                                             mu_band=(8,12), beta_band=(13,30))

        proxy_map = {'Fp1':'FC1', 'Fp2':'FC2', 'C3':'C3', 'C4':'C4'}
        self.asym_pairs_bnci = tuple((proxy_map[a], proxy_map[b]) for (a,b) in asym_pairs)

    def _resample_to_target(self, X_250):
        """X_250 shape (N, C, 500) at 250 Hz → (N, C, 400) at 200 Hz."""
        fs_in = 250
        g = gcd(self.target_fs, fs_in)       # 200 & 250 -> g=50 → up=4, down=5
        up, down = self.target_fs // g, fs_in // g
        X = resample_poly(X_250, up=up, down=down, axis=2).astype(np.float32)
        n_target = int((self.tmax - self.tmin) * self.target_fs)  # 400
        if X.shape[2] > n_target:
            X = X[:, :, :n_target]
        elif X.shape[2] < n_target:
            pad = n_target - X.shape[2]
            X = np.pad(X, ((0,0),(0,0),(0,pad)))
        return X

    def load_dataset(self):
        ds = BNCI2014_001()
        p = MotorImagery(
            n_classes=2,
            channels=self.BNCI_TRAIN_CHANNELS,
            tmin=self.tmin, tmax=self.tmax,
            fmin=self.fmin, fmax=self.fmax,
            # Some MOABB versions support this; if not, our filter below still works.
            events=["left_hand", "right_hand"],
        )
        # Raw @250 Hz (N,4,500)
        X_250, y, meta = p.get_data(dataset=ds, subjects=self.subjects)

        # Enforce left/right only → map to 0/1
        X_250, y = self._filter_left_right(X_250, y, ds)

        # Resample to 200 Hz (N,4,400)
        X = self._resample_to_target(X_250)

        # Report
        uniq, cnt = np.unique(y, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
        print(f"[4ch/200Hz] Loaded {X.shape[0]} L/R trials | chans={list(self.BNCI_TRAIN_CHANNELS)} | samples={X.shape[2]}")
        print(f"[4ch/200Hz] Class distribution: {dist}  (0=left, 1=right)")

        return X.astype(np.float32), y.astype(int), meta

    def extract_features(self, X, y):
        csp = CSP(n_components=self.n_csp_components, shrinkage='lw')
        csp_feat = csp.fit_transform(X, y)                               # (N, n_csp)

        psd_feat = self.spec.extract_features(X)                         # (N, 2*C)

        name2idx = {nm:i for i, nm in enumerate(self.BNCI_TRAIN_CHANNELS)}
        pairs_idx = [(name2idx[a], name2idx[b]) for (a,b) in self.asym_pairs_bnci]
        asym_feat = self.spec.extract_asymmetry(X, channel_pairs=pairs_idx)  # (N, 2*#pairs)

        feats = np.concatenate([csp_feat, psd_feat, asym_feat], axis=1).astype(np.float32)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feats = scaler.fit_transform(feats).astype(np.float32)
        return feats, y, scaler

    def load_features(self):
        X, y, _ = self.load_dataset()
        return self.extract_features(X, y)
    
    def _filter_left_right(self, X, y, ds):
        """
        Keep only LEFT/RIGHT MI trials and map to 0/1.
        Works whether y are strings ('left_hand',...) or numeric codes.
        """
        y = np.asarray(y)

        # Case A: y are strings
        if y.dtype.kind in ("O", "U", "S"):
            s = np.char.lower(y.astype(str))
            is_left  = np.array([("left"  in t) and ("hand" in t) for t in s])
            is_right = np.array([("right" in t) and ("hand" in t) for t in s])
            keep = is_left | is_right
            X = X[keep]
            y = np.where(is_right[keep], 1, 0).astype(int)
            return X, y

        # Case B: y are numeric, use dataset's event_id mapping if available
        eid = getattr(ds, "event_id", {})
        if isinstance(eid, dict) and "left_hand" in eid and "right_hand" in eid:
            left_id, right_id = eid["left_hand"], eid["right_hand"]
            keep = np.isin(y, [left_id, right_id])
            X = X[keep]
            y = (y[keep] == right_id).astype(int)
            return X, y

        # Fallback: keep two most frequent labels
        uniq, cnt = np.unique(y, return_counts=True)
        top2 = uniq[np.argsort(-cnt)[:2]]
        keep = np.isin(y, top2)
        X = X[keep]
        yk = y[keep]
        a, b = sorted(np.unique(yk))[:2]
        y = (yk == b).astype(int)
        return X, y
