# src/training/feature_pipeline.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

try:
    import joblib
except Exception:
    joblib = None

from src.feature_extraction.csp import CSP
from src.feature_extraction.spectral_features import SpectralFeatureExtractor


def _bp_filter(X, fs, f_lo, f_hi, order=4):
    """Zero-phase IIR bandpass per channel, axis=-1. X: (N,C,T)."""
    b, a = butter(order, [f_lo/(fs/2), f_hi/(fs/2)], btype="bandpass")
    Xf = filtfilt(b, a, X, axis=-1)
    return Xf.astype(np.float32)


class FeaturePipeline:
    """
    Features from raw windows X (N, C, T):
      - CSP (log-var) with shrinkage
      - optional FBCSP over fb_bands (CSP per band, concat)
      - PSD features (mu + beta per channel)
      - Asymmetry features (mu & beta for 'asym_pairs')
      - StandardScaler

    4-ch order assumed: [C4, FC2, FC1, C3]
      -> default asym_pairs=((3,0),) == (C3, C4)

    Backward-compatible kwargs accepted: sampling_rate, csp_components,
    mu_band/beta_band, channel_pairs/channel_names.
    """

    def __init__(
        self,
        fs=None, sampling_rate=None,
        n_csp=None, csp_components=None,
        bands=None, mu_band=None, beta_band=None,
        asym_pairs=None, channel_pairs=None,
        channel_names=None,
        use_fbcsp=False, fb_bands=((8,12), (12,16), (16,26)),
        shrinkage="lw",
        **kwargs,
    ):
        # fs
        if fs is None and sampling_rate is None: fs = 250
        self.fs = fs if fs is not None else sampling_rate

        # CSP comps
        if n_csp is None: n_csp = csp_components if csp_components is not None else 2
        self.n_csp = int(n_csp)

        # bands
        if bands is not None:
            self.bands = tuple(bands)
            self.mu_band = tuple(self.bands[0]) if len(self.bands) > 0 else (8,12)
            self.beta_band = tuple(self.bands[1]) if len(self.bands) > 1 else (13,30)
        else:
            self.mu_band = tuple(mu_band) if mu_band is not None else (8,12)
            self.beta_band = tuple(beta_band) if beta_band is not None else (13,30)
            self.bands = (self.mu_band, self.beta_band)

        # asymmetry
        if asym_pairs is not None:
            self.asym_pairs = tuple(tuple(p) for p in asym_pairs)
        elif channel_pairs is not None:
            self.asym_pairs = tuple(tuple(p) for p in channel_pairs)
        else:
            # focus on motor pair only (C3 vs C4) for 4-ch order [C4, FC2, FC1, C3]
            self.asym_pairs = ((3, 0),)

        self.channel_names = tuple(channel_names) if channel_names is not None else None
        self.use_fbcsp = bool(use_fbcsp)
        self.fb_bands = tuple(tuple(b) for b in fb_bands)
        self.shrinkage = shrinkage

        # blocks
        self.csp = CSP(n_components=self.n_csp, shrinkage=self.shrinkage)
        self.spec = SpectralFeatureExtractor(
            sampling_rate=self.fs, mu_band=self.mu_band, beta_band=self.beta_band
        )
        self.scaler = StandardScaler()

        self._fitted = False
        self._feature_dim = None

    # ---------- internals ----------
    def _csp_block(self, X, y=None):
        return self.csp.fit_transform(X, y) if y is not None else self.csp.transform(X)

    def _fbcsp_block(self, X, y=None):
        """CSP per sub-band, concat horizontally."""
        feats = []
        for (lo, hi) in self.fb_bands:
            Xb = _bp_filter(X, self.fs, lo, hi)
            if y is not None:
                feats.append(self.csp.__class__(n_components=self.n_csp, shrinkage=self.shrinkage).fit_transform(Xb, y))
            else:
                # Use main CSP if already fitted on fullband; for strictness, you can refit per band earlier.
                feats.append(self.csp.transform(Xb))
        return np.concatenate(feats, axis=1) if feats else np.zeros((X.shape[0], 0), dtype=np.float32)

    def _build_feats(self, X, y=None):
        # Base CSP on full band
        csp_feat = self._csp_block(X, y=y)

        # Optional FBCSP (adds small extra CSP features)
        fbcsp_feat = self._fbcsp_block(X, y=y) if self.use_fbcsp else np.zeros((X.shape[0], 0), dtype=np.float32)

        # PSD mu+beta per channel
        psd_feat = self.spec.extract_features(X)  # (N, 2*C)

        # Asymmetry (mu & beta per pair)
        asym_feat = self.spec.extract_asymmetry(X, channel_pairs=self.asym_pairs)

        feats = np.concatenate([csp_feat, fbcsp_feat, psd_feat, asym_feat], axis=1)
        return feats.astype(np.float32)

    # ---------- API ----------
    def fit_transform(self, X, y):
        feats = self._build_feats(X, y=y)
        feats = self.scaler.fit_transform(feats)
        self._fitted = True
        self._feature_dim = feats.shape[1]
        return feats

    def fit(self, X, y):
        _ = self.fit_transform(X, y)
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transform().")
        feats = self._build_feats(X, y=None)
        feats = self.scaler.transform(feats)
        return feats

    @property
    def feature_dim(self): return self._feature_dim

    # ---------- persistence ----------
    def save(self, path: str):
        if joblib is None: raise RuntimeError("joblib is not available.")
        obj = dict(
            fs=self.fs, n_csp=self.n_csp, bands=self.bands,
            mu_band=self.mu_band, beta_band=self.beta_band,
            asym_pairs=self.asym_pairs, channel_names=self.channel_names,
            use_fbcsp=self.use_fbcsp, fb_bands=self.fb_bands, shrinkage=self.shrinkage,
            csp=self.csp, spec=self.spec, scaler=self.scaler,
            _fitted=self._fitted, _feature_dim=self._feature_dim,
        )
        joblib.dump(obj, path)

    @classmethod
    def load(cls, path: str):
        if joblib is None: raise RuntimeError("joblib is not available.")
        obj = joblib.load(path)
        fp = cls(
            fs=obj.get("fs", 250),
            n_csp=obj.get("n_csp", 2),
            bands=obj.get("bands", ((8,12),(13,30))),
            asym_pairs=obj.get("asym_pairs", ((3,0),)),
            channel_names=obj.get("channel_names", None),
            use_fbcsp=obj.get("use_fbcsp", False),
            fb_bands=obj.get("fb_bands", ((8,12),(12,16),(16,26))),
            shrinkage=obj.get("shrinkage", "lw"),
        )
        fp.csp = obj["csp"]; fp.spec = obj["spec"]; fp.scaler = obj["scaler"]
        fp._fitted = obj.get("_fitted", True); fp._feature_dim = obj.get("_feature_dim", None)
        return fp
