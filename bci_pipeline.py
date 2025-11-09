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
    
    def __init__(self, tmin=-1.0, tmax=4.0, baseline=(-1.0, 0.0)):
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

print("="*60)
print("MOTOR IMAGERY BCI - TRAINING PIPELINE")
print("="*60)

# STEP 1: Load Data from Multiple Subjects
print("\nStep 1: Loading data from multiple subjects...")

# Subject 1, Runs 3 (Left/Right Hand) and 7 (Left/Right Hand) and 11 
subject_ids = [1, 2, 3]  # Use 3 subjects
runs = [3, 7, 11]  # Motor imagery runs (left/right hand)

all_raws = []
for subject in subject_ids:
    print(f"  Loading subject {subject}...")
    for run in runs:
        try:
            raw_fnames = mne.datasets.eegbci.load_data(subject, [run])
            raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
            all_raws.append(raw)
        except Exception as e:
            print(f"    Warning: Could not load subject {subject}, run {run}: {e}")

# Concatenate all runs
raw = mne.concatenate_raws(all_raws, verbose=False)
print(f"✓ Loaded {len(all_raws)} runs from {len(subject_ids)} subjects")
print(f"  Total duration: {raw.times[-1]:.1f} seconds")


# Step 2: Preprocessing
print("\nStep 2: Preprocessing...")

print("  Filtering to mu (8-12 Hz) and beta (13-30 Hz) bands...")
raw.filter(l_freq=8.0, h_freq=30.0, fir_design='firwin', verbose=False)

# Notch filter for power line noise
raw.notch_filter(freqs=[60], verbose=False)

# Standardize channel names and pick motor cortex channels
mne.datasets.eegbci.standardize(raw)
motor_channels = ['C3', 'C4', 'Cz']  # Minimum for motor imagery
available_channels = [ch for ch in motor_channels if ch in raw.ch_names]
raw.pick_channels(available_channels)

print(f"✓ Using {len(available_channels)} motor cortex channels: {available_channels}")


# 3. Segment Data using EpochCreator
print("Step 3: Segmenting into Epochs...")
events, event_id_all = mne.events_from_annotations(raw, verbose=False)

# T1: Left Hand Motor Imagery, T2: Right Hand Motor Imagery
event_id_binary = {'T1': 1, 'T2': 2} 

# Create epochs: tmin=-0.2s to tmax=0.5s relative to the event (cue)
epoch_creator = EpochCreator(tmin=-1.0, tmax=4.0, baseline=(-1.0, 0.0))
epochs = epoch_creator.create_epochs(raw, events, event_id_binary)

# Extract motor imagery time window (2.0 seconds starting 0.5 seconds AFTER the cue)
epochs_mi = epoch_creator.extract_motor_imagery_window(epochs, start=0.5, duration=3.0)

# Extract data (X) and labels (y)
X = epochs_mi.get_data()
y = epochs_mi.events[:, 2] - 1

# Map MNE labels (1 and 2) to binary labels (0 and 1)
y = epochs_mi.events[:, 2] - 1

print(f"✓ Created {len(X)} epochs")
print(f"  Shape: {X.shape}")
print(f"  Class distribution: {np.bincount(y)}")

# -----------------------------------------------------------------
# STEP 4: Split Data for Training and Testing
# -----------------------------------------------------------------
print("\nStep 4: Splitting data (80% Train / 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {X_train.shape[0]} epochs")
print(f"  Test: {X_test.shape[0]} epochs")

# -----------------------------------------------------------------
# STEP 5: Feature Extraction using CSP
# -----------------------------------------------------------------
print("\nStep 5: Fitting CSP and Transforming Features...")
csp = CSP(n_components=6) 

# Fit CSP ONLY on the training data to prevent data leakage
X_train_features = csp.fit_transform(X_train, y_train)

# Transform both training and testing data
X_test_features = csp.transform(X_test)

print(f"CSP Training Features shape: {X_train_features.shape}")
print(f"CSP Testing Features shape: {X_test_features.shape}")

# STEP 6: Train Classifier
print("\nStep 6: Training LDA classifier...")

classifier = MotorImageryClassifier(
    model_type='LDA',
    solver='lsqr',      # Better for small datasets
    shrinkage='auto'    # Regularization
)

classifier.train(X_train_features, y_train)

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_scores = classifier.cross_validate(X_train_features, y_train, n_folds=5)

# STEP 7: Evaluate
print("\n" + "="*60)
print("EVALUATION ON TEST SET")
print("="*60)

results = classifier.evaluate(X_test_features, y_test)

# STEP 8: Save Model
print("\nStep 8: Saving model...")
os.makedirs('data/models', exist_ok=True)
classifier.save_model('data/models/lda_classifier.pkl')
joblib.dump(csp, 'data/models/csp_model.pkl')

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Final Test Accuracy: {results['accuracy']:.1%}")
print(f"CV Mean: {cv_scores.mean():.1%} (±{cv_scores.std()*2:.1%})")

if results['accuracy'] >= 0.70:
    print("✓✓✓ SUCCESS! Exceeds 70% requirement")
else:
    print(f"⚠️ Below 70% target")