import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import joblib  # for saving/loading
except Exception as e:
    joblib = None

from src.feature_extraction.csp import CSP
from src.feature_extraction.spectral_features import SpectralFeatureExtractor


class FeaturePipeline:
    """
    Train+serve feature pipeline:
      - CSP(n_components)
      - PSD (mu + beta)
      - Asymmetry (mu & beta for provided channel_pairs)
      - StandardScaler

    Usage:
        fp = FeaturePipeline(fs=250, n_csp=4, channel_pairs=[(0,1)])
        X_train_feat = fp.fit_transform(X_train_raw, y_train)
        X_val_feat   = fp.transform(X_val_raw)
        fp.save("data/weights/feature_pipeline.joblib")
        # later:
        fp = FeaturePipeline.load("data/weights/feature_pipeline.joblib")
        X_rt_feat = fp.transform(X_window[np.newaxis, ...])  # (1, C, T)
    """

    def __init__(
        self,
        fs: int = 250,
        n_csp: int = 4,
        mu_band=(8, 12),
        beta_band=(13, 30),
        channel_pairs=None,  # e.g., [(C3_idx, C4_idx)]
    ):
        self.fs = fs
        self.n_csp = n_csp
        self.mu_band = mu_band
        self.beta_band = beta_band
        self.channel_pairs = channel_pairs if channel_pairs is not None else [(0, 1)]

        self.csp = CSP(n_components=n_csp)
        self.spec = SpectralFeatureExtractor(
            sampling_rate=fs, mu_band=mu_band, beta_band=beta_band
        )
        self.scaler = StandardScaler()

        self._fitted = False
        self._feature_dim = None

    def _concat_features(self, X_raw, use_csp_transform_only=False):
        """
        Build feature matrix from raw windows X_raw: (N, C, T)
        If use_csp_transform_only=True, assumes CSP is already fitted and uses transform().
        """
        if use_csp_transform_only:
            csp_feat = self.csp.transform(X_raw)
        else:
            # during fit we call fit_transform
            csp_feat = self.csp.fit_transform(X_raw, self._fit_labels)

        psd_feat = self.spec.extract_features(X_raw)                       # (N, 2*C)
        asym_feat = self.spec.extract_asymmetry(X_raw, self.channel_pairs) # (N, 2*len(pairs))

        X_feat = np.concatenate([csp_feat, psd_feat, asym_feat], axis=1)
        return X_feat

    def fit(self, X_raw: np.ndarray, y: np.ndarray):
        """
        Fit CSP (needs labels) + scaler on concatenated feature vectors.
        Returns fitted self; call transform() to get features.
        """
        self._fit_labels = y  # store just for csp.fit_transform
        X_feat = self._concat_features(X_raw, use_csp_transform_only=False)

        self.scaler.fit(X_feat)
        self._feature_dim = X_feat.shape[1]
        self._fitted = True
        # clean temp
        del self._fit_labels
        return self

    def fit_transform(self, X_raw: np.ndarray, y: np.ndarray):
        self.fit(X_raw, y)
        return self.transform(X_raw)

    def transform(self, X_raw: np.ndarray):
        """
        Transform raw windows with fitted CSP + PSD + Asym + Scaler.
        """
        if not self._fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transform()")
        X_feat = self._concat_features(X_raw, use_csp_transform_only=True)
        X_scaled = self.scaler.transform(X_feat)
        return X_scaled

    @property
    def feature_dim(self):
        return self._feature_dim

    def save(self, path: str):
        if joblib is None:
            raise RuntimeError("joblib is not available to save the pipeline.")
        obj = {
            "fs": self.fs,
            "n_csp": self.n_csp,
            "mu_band": self.mu_band,
            "beta_band": self.beta_band,
            "channel_pairs": self.channel_pairs,
            "csp": self.csp,
            "spec": self.spec,
            "scaler": self.scaler,
            "_fitted": self._fitted,
            "_feature_dim": self._feature_dim,
        }
        joblib.dump(obj, path)

    @classmethod
    def load(cls, path: str):
        if joblib is None:
            raise RuntimeError("joblib is not available to load the pipeline.")
        obj = joblib.load(path)
        fp = cls(
            fs=obj["fs"],
            n_csp=obj["n_csp"],
            mu_band=obj["mu_band"],
            beta_band=obj["beta_band"],
            channel_pairs=obj["channel_pairs"],
        )
        fp.csp = obj["csp"]
        fp.spec = obj["spec"]
        fp.scaler = obj["scaler"]
        fp._fitted = obj["_fitted"]
        fp._feature_dim = obj["_feature_dim"]
        return fp
