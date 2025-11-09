import numpy as np
from sklearn.preprocessing import StandardScaler

# Clean, labeled MI via MOABB
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

# Your feature extractors
from src.feature_extraction.csp import CSP
from src.feature_extraction.spectral_features import SpectralFeatureExtractor


def _ensure_fixed_len_trials(X: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ensure every trial has exactly target_len samples (crop/pad at end).
    X: (N, C, T)
    """
    N, C, T = X.shape
    if T == target_len:
        return X
    out = np.zeros((N, C, target_len), dtype=X.dtype)
    if T > target_len:
        out[:] = X[:, :, :target_len]
    else:
        out[:, :, :T] = X
    return out


def _filter_and_map_left_right(X, y, meta):
    """
    Robustly keep only Left/Right trials and map labels to {0:left, 1:right},
    regardless of whether y is strings or ints.
    Returns: X_lr, y_lr
    """
    y_np = np.asarray(y)

    # Case A: string/object labels like ['left_hand','right_hand','feet','tongue']
    if y_np.dtype.kind in {"U", "S", "O"}:
        y_str = y_np.astype(str)
        keep = np.isin(y_str, ["left_hand", "right_hand"])
        X = X[keep]
        y_str = y_str[keep]
        mapping = {"left_hand": 0, "right_hand": 1}
        y = np.array([mapping[s] for s in y_str], dtype=int)
        return X, y

    # Case B: numeric labels (varies with MOABB version)
    uniq = np.unique(y_np).tolist()

    # Already binary {0,1}
    if set(uniq).issubset({0, 1}):
        return X, y_np.astype(int)

    # Classic PhysioBank codes {1:left, 2:right, 3:feet, 4:tongue}
    if set(uniq).issubset({1, 2, 3, 4}):
        keep = np.isin(y_np, [1, 2])
        X = X[keep]
        y_np = y_np[keep]
        y = np.where(y_np == 1, 0, 1).astype(int)  # 1->0 (left), 2->1 (right)
        return X, y

    # Fallback: see if meta holds string labels we can use
    if hasattr(meta, "columns"):
        for col in ["labels", "event", "events", "class", "classes", "target_name"]:
            if col in meta.columns:
                names = meta[col].astype(str).to_numpy()
                keep = np.isin(names, ["left_hand", "right_hand"])
                if keep.any():
                    X = X[keep]
                    names = names[keep]
                    mapping = {"left_hand": 0, "right_hand": 1}
                    y = np.array([mapping[s] for s in names], dtype=int)
                    return X, y

    # If we reach here, we couldn't confidently remap—show what we saw.
    raise RuntimeError(
        f"Could not map labels to left/right. Unique labels found: {uniq}. "
        "Try upgrading moabb, or inspect `meta` to find the correct label column."
    )


class EEGFeatureLoader:
    """
    Loads BNCI 2014-001 Left/Right MI with MOABB and extracts:
      • CSP (n_components)
      • PSD (mu + beta per channel)
      • Asymmetry (mu & beta for pairs)
    Returns:
        X_features: (N, D) standardized
        y:          (N,)   in {0,1}
    """

    def __init__(
        self,
        dataset_name: str = "BNCI2014_001",
        sampling_rate: int = 250,
        n_csp_components: int = 4,
        channel_pairs=None,  # e.g., [(C3_idx, C4_idx)] when you map indices later
    ):
        self.dataset_name = dataset_name
        self.sampling_rate = sampling_rate
        self.n_csp_components = n_csp_components
        self.channel_pairs = channel_pairs if channel_pairs is not None else [(0, 1)]
        self.spectral_extractor = SpectralFeatureExtractor(sampling_rate=sampling_rate)

    def load_dataset(self, subjects=None, tmin=0.0, tmax=2.0, fmin=8.0, fmax=30.0):
        """
        Use MOABB MotorImagery to get Left/Right trials.
        Returns X: (N,C,T) and y: (N,) raw (possibly strings or ints).
        """
        if subjects is None:
            subjects = [1]

        print("Loading BNCI 2014-001 via MOABB (Left/Right, 2s, 8–30 Hz)...")
        dataset = BNCI2014_001()

        # Use 2-class MI robustly across MOABB versions
        paradigm = MotorImagery(
            n_classes=2,          # older moabb prefers this over events=[...]
            fmin=fmin, fmax=fmax,
            tmin=tmin, tmax=tmax,
            resample=self.sampling_rate,
        )

        X, y, meta = paradigm.get_data(dataset=dataset, subjects=subjects)

        # Enforce exact expected length (tmax-tmin)*fs in case of off-by-ones
        target_len = int(self.sampling_rate * (tmax - tmin))
        X = _ensure_fixed_len_trials(X, target_len)

        # Filter to left/right and map to {0,1}
        X, y = _filter_and_map_left_right(X, y, meta)

        print(f"Trials after LR filter: {X.shape[0]}, Channels: {X.shape[1]}, Samples: {X.shape[2]}")
        uniq, cnt = np.unique(y, return_counts=True)
        print("Class distribution:", {int(k): int(v) for k, v in zip(uniq, cnt)})
        return X, y

    def extract_features(self, X: np.ndarray, y: np.ndarray):
        """
        Compute CSP + PSD + Asymmetry feature vectors and standardize.
        """
        print("Extracting features...")

        # 1) CSP
        print("  ➤ Applying CSP...")
        csp = CSP(n_components=self.n_csp_components)
        csp_features = csp.fit_transform(X, y)  # (N, n_csp_components)

        # 2) PSD (mu + beta per channel)
        print("  ➤ Extracting PSD features...")
        psd_features = self.spectral_extractor.extract_features(X)  # (N, 2*C)

        # 3) Asymmetry
        print("  ➤ Computing asymmetry...")
        asym_features = self.spectral_extractor.extract_asymmetry(
            X, channel_pairs=self.channel_pairs
        )  # (N, 2*len(pairs))

        # Concatenate
        X_features = np.concatenate([csp_features, psd_features, asym_features], axis=1)
        print("Feature extraction complete.")
        print(f"Final feature vector shape: {X_features.shape}")

        # Standardize
        scaler = StandardScaler()
        X_features = scaler.fit_transform(X_features)

        return X_features, y

    def load_features(self, subjects=None, tmin=0.0, tmax=2.0, fmin=8.0, fmax=30.0):
        """
        Full pipeline: load → features → return arrays.
        """
        X, y = self.load_dataset(subjects=subjects, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        return self.extract_features(X, y)


if __name__ == "__main__":
    loader = EEGFeatureLoader(
        sampling_rate=250,
        n_csp_components=4,
        channel_pairs=[(0, 1)],  # swap to actual (C3,C4) indices later if you map them
    )
    X, y = loader.load_features()
    print("X shape:", X.shape)
    print("y shape:", y.shape)
