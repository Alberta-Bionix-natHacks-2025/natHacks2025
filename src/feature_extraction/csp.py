"""Common Spatial Patterns (CSP) with optional shrinkage regularization."""
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import LedoitWolf


class CSP(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns for BCI.

    Parameters
    ----------
    n_components : int, default=4
        Number of spatial filters to keep. Should be even; we take n/2
        filters from each end of the generalized eigen-spectrum.
    shrinkage : {'lw', None}, default=None
        If 'lw', use Ledoit–Wolf shrinkage for per-epoch covariance
        estimation before averaging across epochs (helps with few channels
        or few trials).

    Notes
    -----
    Input X is expected as (n_epochs, n_channels, n_timepoints).
    Labels y are binary (two classes).
    """

    def __init__(self, n_components=4, shrinkage=None, eps=1e-12):
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        if n_components % 2 != 0:
            # enforce even to keep symmetric ends of spectrum
            n_components -= 1
        self.n_components = max(2, n_components)
        self.shrinkage = shrinkage
        self.eps = eps

        self.filters_ = None  # W, shape (n_components, n_channels)
        self.patterns_ = None

    # ------------------------------- public API -------------------------------

    def fit(self, X, y):
        """
        Fit CSP spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
        y : ndarray, shape (n_epochs,), two unique classes

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires 2 classes, got {len(classes)}")

        X0 = X[y == classes[0]]
        X1 = X[y == classes[1]]

        C0 = self._avg_cov(X0)
        C1 = self._avg_cov(X1)

        # Solve C0 w = lambda (C0 + C1) w
        # Add tiny ridge to ensure PD
        Csum = C0 + C1 + self.eps * np.eye(C0.shape[0])

        evals, evecs = linalg.eigh(C0, Csum)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]

        n_half = self.n_components // 2
        sel = np.concatenate([np.arange(n_half), np.arange(-n_half, 0)])
        W = evecs[:, sel].T  # (n_comp, n_chan)

        self.filters_ = W
        # Spatial patterns for viz (columns)
        self.patterns_ = linalg.pinv(W).T
        return self

    def transform(self, X):
        """
        Apply CSP filters and return log-variance features.

        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)

        Returns
        -------
        feats : ndarray, shape (n_epochs, n_components)
        """
        if self.filters_ is None:
            raise ValueError("CSP must be fitted before transform.")

        # (n_comp, n_chan) @ (n_ep, n_chan, n_time) -> (n_comp, n_ep, n_time)
        Xf = np.tensordot(self.filters_, X, axes=(1, 1))
        Xf = np.transpose(Xf, (1, 0, 2))  # (n_ep, n_comp, n_time)

        feats = np.log(np.var(Xf, axis=2) + self.eps)
        return feats

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    # ------------------------------ helpers ----------------------------------

    def _avg_cov(self, X):
        """
        Average per-epoch covariances, normalized by trace.
        Optional Ledoit–Wolf shrinkage when self.shrinkage == 'lw'.
        """
        covs = []
        use_lw = (self.shrinkage in ("lw", "ledoit"))
        for epoch in X:
            # epoch: (n_chan, n_time)
            if use_lw:
                # LedoitWolf expects (n_samples, n_features) = (T, C)
                lw = LedoitWolf().fit(epoch.T)
                C = lw.covariance_
            else:
                C = epoch @ epoch.T
            tr = np.trace(C)
            if tr <= 0:
                tr = 1.0
            covs.append(C / tr)
        return np.mean(covs, axis=0)


# Quick self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_ep, n_ch, n_t = 60, 4, 500
    X = rng.standard_normal((n_ep, n_ch, n_t))
    y = np.array([0] * (n_ep // 2) + [1] * (n_ep // 2))

    # inject a tiny class-dependent spatial variance
    X[y == 0, 0] *= 1.5
    X[y == 1, 1] *= 1.5

    csp = CSP(n_components=4, shrinkage="lw")
    F = csp.fit_transform(X, y)
    print("CSP features:", F.shape)
