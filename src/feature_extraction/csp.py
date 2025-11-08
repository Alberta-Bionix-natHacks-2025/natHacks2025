'''Finds the best "combination" of electrodes to maximize class separation'''
import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin

class CSP(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns for BCI
    
    CSP finds spatial filters that maximize variance for one class
    and minimize for another class.
    """
    
    def __init__(self, n_components=4):
        """
        Parameters:
        - n_components: Number of CSP filters to use (take n/2 from each end)
        """
        self.n_components = n_components
        self.filters_ = None
        self.patterns_ = None
        
    def fit(self, X, y):
        """
        Fit CSP filters
        
        Input:
        - X: (n_epochs, n_channels, n_timepoints)
        - y: (n_epochs,) - binary labels (0 or 1)
        
        Output: self
        """
        # Check for binary classification
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires 2 classes, got {len(classes)}")
        
        # Split data by class
        X_class0 = X[y == classes[0]]
        X_class1 = X[y == classes[1]]
        
        # Compute average covariance matrices
        cov_class0 = self._compute_covariance(X_class0)
        cov_class1 = self._compute_covariance(X_class1)
        
        # Solve generalized eigenvalue problem
        # cov_class0 @ w = lambda @ (cov_class0 + cov_class1) @ w
        eigenvalues, eigenvectors = linalg.eigh(
            cov_class0, 
            cov_class0 + cov_class1
        )
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components: n_components/2 from each end
        n_comp_half = self.n_components // 2
        selected_idx = np.concatenate([
            np.arange(n_comp_half),  # First n_comp_half (high variance for class 0)
            np.arange(len(eigenvalues) - n_comp_half, len(eigenvalues))  # Last n_comp_half
        ])
        
        # Spatial filters (rows are filters)
        self.filters_ = eigenvectors[:, selected_idx].T
        
        # Spatial patterns (for visualization)
        self.patterns_ = linalg.pinv(self.filters_).T
        
        return self
    
    def transform(self, X):
        """
        Apply CSP filters and extract log-variance features
        
        Input: X (n_epochs, n_channels, n_timepoints)
        Output: features (n_epochs, n_components)
        """
        if self.filters_ is None:
            raise ValueError("CSP must be fitted before transform")
        
        # Apply spatial filters: (n_components, n_channels) @ (n_channels, n_timepoints)
        X_filtered = np.tensordot(self.filters_, X, axes=(1, 1))
        X_filtered = np.transpose(X_filtered, (1, 0, 2))  # (n_epochs, n_components, n_timepoints)
        
        # Compute log variance as features
        features = np.log(np.var(X_filtered, axis=2))
        
        return features
    
    def fit_transform(self, X, y):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def _compute_covariance(self, X):
        """
        Compute normalized covariance matrix averaged over trials
        
        Input: X (n_epochs, n_channels, n_timepoints)
        Output: cov (n_channels, n_channels)
        """
        n_epochs, n_channels, n_timepoints = X.shape
        cov_sum = np.zeros((n_channels, n_channels))
        
        for epoch in X:
            # Covariance of this epoch
            cov = np.dot(epoch, epoch.T)
            # Normalize by trace
            cov = cov / np.trace(cov)
            cov_sum += cov
        
        # Average over epochs
        return cov_sum / n_epochs


class MultiClassCSP:
    """
    CSP for multi-class problems using One-vs-Rest approach
    """
    
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.csp_filters = {}
        
    def fit(self, X, y):
        """
        Fit one CSP per class (one-vs-rest)
        
        Input:
        - X: (n_epochs, n_channels, n_timepoints)
        - y: (n_epochs,) - class labels
        """
        classes = np.unique(y)
        
        for cls in classes:
            print(f"Training CSP for class {cls} vs rest...")
            
            # Create binary labels (this class vs all others)
            y_binary = (y == cls).astype(int)
            
            # Fit CSP
            csp = CSP(n_components=self.n_components)
            csp.fit(X, y_binary)
            
            self.csp_filters[cls] = csp
        
        return self
    
    def transform(self, X):
        """
        Transform using all CSP filters and concatenate
        
        Input: X (n_epochs, n_channels, n_timepoints)
        Output: features (n_epochs, n_classes * n_components)
        """
        features_list = []
        
        for cls in sorted(self.csp_filters.keys()):
            csp = self.csp_filters[cls]
            features = csp.transform(X)
            features_list.append(features)
        
        # Concatenate all CSP features
        all_features = np.concatenate(features_list, axis=1)
        
        return all_features
    
    def fit_transform(self, X, y):
        """Fit and transform"""
        return self.fit(X, y).transform(X)


# Quick test
if __name__ == '__main__':
    # Test with dummy data
    np.random.seed(42)
    
    # Simulate 2 classes with different spatial patterns
    n_epochs = 50
    n_channels = 3
    n_times = 480
    
    X = np.random.randn(n_epochs, n_channels, n_times)
    y = np.array([0] * 25 + [1] * 25)
    
    # Add some class-specific patterns
    X[y == 0, 0, :] *= 2  # Class 0 has higher power in channel 0
    X[y == 1, 1, :] *= 2  # Class 1 has higher power in channel 1
    
    # Test CSP
    csp = CSP(n_components=4)
    features = csp.fit_transform(X, y)
    
    print(f"CSP features shape: {features.shape}")  # Should be (50, 4)
    print(f"Mean features class 0: {features[y==0].mean(axis=0)}")
    print(f"Mean features class 1: {features[y==1].mean(axis=0)}")
    
    # Test multi-class CSP
    y_multiclass = np.array([0]*16 + [1]*17 + [2]*17)
    X_multiclass = np.random.randn(50, 3, 480)
    
    mcsp = MultiClassCSP(n_components=4)
    features_mc = mcsp.fit_transform(X_multiclass, y_multiclass)
    
    print(f"\nMulti-class CSP features shape: {features_mc.shape}")  # Should be (50, 8)