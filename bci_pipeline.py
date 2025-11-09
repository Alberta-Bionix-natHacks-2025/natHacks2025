import mne
import numpy as np
import os
import joblib # Needed by MotorImageryClassifier
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC # For SVM model option
from sklearn.ensemble import RandomForestClassifier # For RandomForest model option
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold # For Classifier functionality
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# --- MNE PATH FIX ---
# Set the MNE data path explicitly to avoid FileNotFoundError
current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'mne_datasets')
os.makedirs(data_path, exist_ok=True)
mne.set_config('MNE_DATA', data_path, set_env=True)
# The data will be downloaded to: /Users/dinma/Desktop/NatHacks25/natHacks2025/mne_datasets
# --------------------


# =================================================================
# --- CLASS DEFINITION SECTION ---
# =================================================================

class CSP(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns for BCI
    CSP finds spatial filters that maximize variance for one class
    and minimize for another class.
    """
    
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None
        self.patterns_ = None
        
    def fit(self, X, y):
        """Fit CSP filters"""
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires 2 classes, got {len(classes)}")
        
        X_class0 = X[y == classes[0]]
        X_class1 = X[y == classes[1]]
        
        cov_class0 = self._compute_covariance(X_class0)
        cov_class1 = self._compute_covariance(X_class1)
        
        # Solve generalized eigenvalue problem: cov_class0 @ w = lambda @ (cov_class0 + cov_class1) @ w
        eigenvalues, eigenvectors = linalg.eigh(
            cov_class0, 
            cov_class0 + cov_class1
        )
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        n_comp_half = self.n_components // 2
        selected_idx = np.concatenate([
            np.arange(n_comp_half),  # First n_comp_half (Class 0)
            np.arange(len(eigenvalues) - n_comp_half, len(eigenvalues))  # Last n_comp_half (Class 1)
        ])
        
        self.filters_ = eigenvectors[:, selected_idx].T
        self.patterns_ = linalg.pinv(self.filters_).T
        
        return self
    
    def transform(self, X):
        """Apply CSP filters and extract log-variance features"""
        if self.filters_ is None:
            raise ValueError("CSP must be fitted before transform")
        
        # Apply spatial filters
        X_filtered = np.tensordot(self.filters_, X, axes=(1, 1))
        X_filtered = np.transpose(X_filtered, (1, 0, 2))  # (n_epochs, n_components, n_timepoints)
        
        # Compute log variance as features
        features = np.log(np.var(X_filtered, axis=2))
        
        return features
    
    def fit_transform(self, X, y):
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)
    
    def _compute_covariance(self, X):
        """Compute normalized covariance matrix averaged over trials"""
        n_epochs, n_channels, n_timepoints = X.shape
        cov_sum = np.zeros((n_channels, n_channels))
        
        for epoch in X:
            cov = np.dot(epoch, epoch.T)
            cov = cov / np.trace(cov)
            cov_sum += cov
        
        return cov_sum / n_epochs


class EpochCreator:
    """Segment continuous EEG into epochs/trials"""
    
    def __init__(self, tmin=-0.2, tmax=3.0, baseline=(-0.2, 0.0)):
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
    
    def create_epochs(self, raw, events, event_dict):
        """Create epochs from continuous data"""
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_dict,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            preload=True,
            reject=None,
            picks='eeg'
        )
        
        return epochs
    
    def extract_motor_imagery_window(self, epochs, start=3.0, duration=2.0):
        """Extract specific time window for motor imagery (0.5s to 2.5s)"""
        epochs_cropped = epochs.copy().crop(tmin=start, tmax=start+duration)
        return epochs_cropped
    

class MotorImageryClassifier:
    """Train and evaluate motor imagery classifiers"""
    
    def __init__(self, model_type='LDA', **model_params):
        self.model_type = model_type
        
        if model_type == 'LDA':
            self.model = LinearDiscriminantAnalysis(**model_params)
        elif model_type == 'SVM':
            default_params = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
            default_params.update(model_params)
            self.model = SVC(probability=True, **default_params)
        elif model_type == 'RandomForest':
            default_params = {'n_estimators': 100, 'max_depth': 10}
            default_params.update(model_params)
            self.model = RandomForestClassifier(**default_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """Train the classifier"""
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        
        train_accuracy = self.model.score(X_train, y_train)
        print(f"Training accuracy: {train_accuracy:.3f}")
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set"""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.3f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def cross_validate(self, X, y, n_folds=5):
        """Perform k-fold cross-validation"""
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"\nCross-validation scores ({n_folds} folds):")
        print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
        print(f"Individual fold scores: {scores}")
        
        return scores
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return self

# =================================================================
# --- MAIN EXECUTION PIPELINE ---
# =================================================================

# 1. Load Data and Apply Filters
print("Step 1: Loading and Preprocessing Raw Data...")
# Subject 1, Runs 3 (Left/Right Hand) and 7 (Left/Right Hand)
raw_fnames = mne.datasets.eegbci.load_data(1, [3, 7])
raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
original_raw = raw.copy()

# Preprocessing
original_raw.filter(l_freq=None, h_freq=50.0, verbose=False) # Low-pass at 50Hz
original_raw.filter(l_freq=0.1, h_freq=None, verbose=False) # High-pass at 0.1Hz
original_raw.notch_filter(freqs=[50, 60], verbose=False)    # Notch filter
original_raw.pick("eeg")                                    # Pick EEG channels

# 2. Segment Data using EpochCreator
print("Step 2: Segmenting into Epochs...")
events, event_id_all = mne.events_from_annotations(original_raw)
# T1: Left Hand Motor Imagery, T2: Right Hand Motor Imagery
event_id_binary = {'T1': 1, 'T2': 2} 

# Create epochs: tmin=-0.2s to tmax=0.5s relative to the event (cue)
epoch_creator = EpochCreator(tmin=-0.2, tmax=3.0, baseline=(-0.2, 0.0))
epochs = epoch_creator.create_epochs(original_raw, events, event_id_binary)

# Extract motor imagery time window (2.0 seconds starting 0.5 seconds AFTER the cue)
epochs_mi = epoch_creator.extract_motor_imagery_window(epochs, start=0.5, duration=2.0)

# Extract data (X) and labels (y)
X = epochs_mi.get_data()
y_mne_labels = epochs_mi.events[:, 2]

# Map MNE labels (1 and 2) to binary labels (0 and 1)
y = y_mne_labels - 1 

print(f"Final epoch data shape (X): {X.shape}")
print(f"Final label vector shape (y): {y.shape}")

# -----------------------------------------------------------------
# STEP 3: Split Data for Training and Testing
# -----------------------------------------------------------------
print("\nStep 3: Splitting data (80% Train / 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set size: {X_train.shape[0]} epochs")
print(f"Test set size: {X_test.shape[0]} epochs")

# -----------------------------------------------------------------
# STEP 4: Feature Extraction using CSP
# -----------------------------------------------------------------
print("\nStep 4: Fitting CSP and Transforming Features...")
csp = CSP(n_components=4) 

# Fit CSP ONLY on the training data to prevent data leakage
X_train_features = csp.fit_transform(X_train, y_train)

# Transform both training and testing data
X_test_features = csp.transform(X_test)

print(f"CSP Training Features shape: {X_train_features.shape}")
print(f"CSP Testing Features shape: {X_test_features.shape}")

# -----------------------------------------------------------------
# STEP 5: Train and Evaluate the Classifier
# -----------------------------------------------------------------
print("\nStep 5: Training and Evaluating Classifier (LDA)...")

# Initialize Classifier (Using LDA as specified)
classifier = MotorImageryClassifier(model_type='LDA')

# Train the model using the transformed CSP features
classifier.train(X_train_features, y_train)

# Evaluate the model on the held-out test set
results = classifier.evaluate(X_test_features, y_test)