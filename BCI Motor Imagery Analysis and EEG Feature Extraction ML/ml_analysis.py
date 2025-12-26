"""
Purpose:
    - Comprehensive Machine Learning Pipeline for Motor Imagery BCI.
    - Features:
        1. Dual-Band Architecture (Mu + Beta) for Feature Extraction.
        2. Feature Fusion: Combines CSP (Spatial) + Temporal Statistics.
        3. Explicit Train/Test Split Support (renamed back to run_full_comparison for GUI compatibility).
        4. Deep Analysis Tools: Learning Curves, Loss Curves, Confusion Matrices.
    
    - Design Philosophy:
        - Maximize accuracy by using spectral-spatial features.
        - Ensure GUI compatibility while enforcing Explicit Data Splitting.
        - Full code expansion (No Brevity).

Dependencies:
    - numpy
    - matplotlib
    - sklearn (Scikit-Learn)
    - scipy (Signal Processing)
    - csp_scratch (Custom Module)
"""

# 1. IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# Signal Processing
from scipy.signal import butter, filtfilt

# Scikit-Learn: Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Scikit-Learn: Model Selection and Metrics
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

# Custom Project Modules
import csp_scratch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 2. CONTEXT & DESCRIPTION HELPER
def get_ml_description():
    """
    Returns a descriptive string explaining the Dual-Band ML strategy.
    """
    description = (
        "--- DUAL-BAND MACHINE LEARNING CLASSIFICATION ---\n\n"
        "1. ADVANCED STRATEGY (FBCSP):\n"
        "   - The pipeline processes two distinct frequency bands:\n"
        "     * Mu Rhythm (8-13 Hz): Motor preparation/idling.\n"
        "     * Beta Rhythm (13-30 Hz): Active motor processing.\n\n"
        "2. FEATURE EXTRACTION PROCESS:\n"
        "   - **Step 1:** Signal is filtered into Mu and Beta bands.\n"
        "   - **Step 2:** CSP Spatial Filters are learned separately for each band.\n"
        "   - **Step 3:** Temporal Statistics (Mean, Var, Skew, Kurt) extracted for each band.\n"
        "   - **Step 4:** All features are concatenated into a single vector.\n\n"
        "3. CLASSIFICATION:\n"
        "   - 8 Models compete to classify the fused feature vector.\n"
        "   - Training uses 100% of labeled training data.\n"
        "   - Testing uses 100% of separate test data (Cross-Session).\n"
    )
    return description

# 3. MACHINE LEARNING PIPELINE CLASS
class ML_Pipeline:
    """
    Manages the end-to-end Machine Learning process with Dual-Band processing.
    """
    
    def __init__(self):
        """
        Initialize the Machine Learning Pipeline.
        Sets up containers for data, models, and separate feature extractors.
        """
        # Feature Extractors for MU BAND (8-13 Hz)
        self.csp_mu = None
        self.temp_mu = None
        
        # Feature Extractors for BETA BAND (13-30 Hz)
        self.csp_beta = None
        self.temp_beta = None
        
        # Data Containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Scaling Object
        self.scaler = None
        
        # Results Containers
        self.trained_models = {}   
        self.model_metrics = {}    
        self.best_model_name = None

    # Helper: Bandpass Filter
    def _apply_bandpass(self, data, fs, lowcut, highcut, order=4):
        """
        Applies a Butterworth Bandpass Filter.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        # Apply filter along time axis (last axis)
        y = filtfilt(b, a, data, axis=-1)
        return y

    # Data Preparation (Raw Segmentation Only)
    def prepare_data(self, raw_data, events, fs, tmin=0.5, tmax=3.5, mode='train'):
        """
        Segments raw EEG data into epochs.
        Note: Unlike previous versions, this does NOT apply filtering yet.
        Filtering happens later during Feature Extraction to support multiple bands.
        
        Args:
            raw_data: (n_channels, n_samples)
            events: Event matrix
            fs: Sampling rate
            tmin, tmax: Window relative to cue
            mode: 'train' or 'inference'
        """
        EV_LEFT = 769
        EV_RIGHT = 770
        EV_UNKNOWN = 783
        
        target_events = [EV_LEFT, EV_RIGHT] if mode == 'train' else [EV_UNKNOWN]
        
        trials = []
        labels = []
        
        offset_start = int(tmin * fs)
        offset_end = int(tmax * fs)
        n_samples_raw = raw_data.shape[1]
        
        for ev in events:
            idx = ev[0]
            eid = ev[2]
            
            if eid not in target_events:
                continue
            
            start = idx + offset_start
            end = idx + offset_end
            
            if start < 0 or end > n_samples_raw:
                continue
                
            # Extract Raw Epoch (No Filter yet)
            epoch_data = raw_data[:, start:end]
            trials.append(epoch_data)
            
            # Labels
            if mode == 'train':
                labels.append(0 if eid == EV_LEFT else 1)
            else:
                labels.append(-1)
            
        return np.array(trials), np.array(labels)
        
    # Advanced Feature Extraction (Dual Band)
    def extract_dual_band_features(self, epochs_raw, fs=250.0, labels=None, is_training=True):
        """
        The core of the advanced pipeline.
        1. Filters data into Mu and Beta bands.
        2. Applies CSP and Temporal Extraction for EACH band separately.
        3. Concatenates everything into one feature vector.
        
        Args:
            epochs_raw: Unfiltered epochs (n_trials, n_channels, n_samples)
            fs: Sampling rate
            labels: Required for training CSP
            is_training: If True, fits models. If False, applies them.
        """
        
        # --- 1. Filter Banks ---
        # Band 1: Mu (8-13 Hz)
        epochs_mu = self._apply_bandpass(epochs_raw, fs, 8.0, 13.0)
        
        # Band 2: Beta (13-30 Hz)
        epochs_beta = self._apply_bandpass(epochs_raw, fs, 13.0, 30.0)
        
        # --- 2. Initialize Models (Training Mode) ---
        if is_training:
            self.csp_mu = csp_scratch.CSP_Scratch(n_components=2)
            self.csp_beta = csp_scratch.CSP_Scratch(n_components=2)
            self.temp_mu = csp_scratch.TemporalFeatureExtractor()
            self.temp_beta = csp_scratch.TemporalFeatureExtractor()
            
            # Fit CSPs
            if labels is None: raise ValueError("Labels needed for training.")
            self.csp_mu.fit(epochs_mu, labels)
            self.csp_beta.fit(epochs_beta, labels)
            
        # Check integrity
        if self.csp_mu is None or self.csp_beta is None:
            raise RuntimeError("Models not trained.")
            
        # --- 3. Transform (Feature Extraction) ---
        
        # Band 1 Features
        feat_spatial_mu = self.csp_mu.transform(epochs_mu)
        feat_temp_mu = self.temp_mu.transform(epochs_mu)
        
        # Band 2 Features
        feat_spatial_beta = self.csp_beta.transform(epochs_beta)
        feat_temp_beta = self.temp_beta.transform(epochs_beta)
        
        # --- 4. Fusion ---
        # Stack all features horizontally
        # Structure: [CSP_Mu, Temp_Mu, CSP_Beta, Temp_Beta]
        features_combined = np.hstack([
            feat_spatial_mu, 
            feat_temp_mu, 
            feat_spatial_beta, 
            feat_temp_beta
        ])
        
        return features_combined

    # Main Training Runner (RENAMED for Compatibility)
    def run_full_comparison(self, X_train_raw, y_train_raw, X_test_raw=None, y_test_raw=None, fs=250.0):
        """
        Trains models using Dual-Band features.
        Supports EXPLICIT split if X_test_raw is provided.
        If X_test_raw is None (Legacy Mode), it splits X_train_raw internally.
        """
        # 1. Handle Explicit vs Internal Split
        if X_test_raw is None:
            # Fallback (Legacy)
            X_train, X_test, y_train, y_test = train_test_split(X_train_raw, y_train_raw, test_size=0.25, stratify=y_train_raw)
            print("[ML] Using Internal 75-25 Split (Legacy Mode)")
            X_train_proc = X_train
            X_test_proc = X_test
        else:
            # Explicit Mode
            print(f"[ML] Using Explicit Split: Train={len(y_train_raw)}, Test={len(y_test_raw)}")
            X_train_proc = X_train_raw
            y_train = y_train_raw
            X_test_proc = X_test_raw
            y_test = y_test_raw
        
        # 2. Feature Extraction
        print(f"[ML] Extracting features for Training...")
        X_train_feat = self.extract_dual_band_features(X_train_proc, fs, y_train, is_training=True)
        
        print(f"[ML] Extracting features for Testing...")
        X_test_feat = self.extract_dual_band_features(X_test_proc, fs, y_test, is_training=False)
        
        # 3. Standardization
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train_feat)
        self.X_test = self.scaler.transform(X_test_feat)
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"[ML INFO] Final Feature Matrix: {self.X_train.shape}")
        
        # 4. Models
        models = {
            'SVM (Linear)': SVC(kernel='linear', C=1.0),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Naive Bayes': GaussianNB(),
            'Kernel SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale'),
            'RF (Tuned)': RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=42),
            'MLP (Neural Network)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, 
                                                  activation='relu', solver='adam', random_state=42)
        }
        
        self.results = {}
        self.model_metrics = {}
        self.trained_models = {}

        # 5. Training Loop
        for name, clf in models.items():
            try:
                clf.fit(self.X_train, self.y_train)
                pred = clf.predict(self.X_test)
                
                # Metrics
                acc = accuracy_score(self.y_test, pred)
                prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, pred, average='macro', zero_division=0)
                
                self.model_metrics[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
                self.results[name] = acc
                self.trained_models[name] = clf
                
            except Exception as e:
                print(f"[ML ERROR] {name}: {e}")
                self.results[name] = 0.0

        if self.results:
            self.best_model_name = max(self.results, key=self.results.get)
            best_acc = self.results[self.best_model_name]
            print(f"\n[ML RESULT] Best Model: {self.best_model_name} ({best_acc*100:.2f}%)")
        
        return self.model_metrics

    # ---------------------------------------------------------
    # Inference Runner
    # ---------------------------------------------------------
    def predict_new_data(self, epochs_raw, model_name, fs=250.0):
        """
        Predicts classes for new data using the Dual-Band pipeline.
        """
        if model_name not in self.trained_models:
            print(f"[ERROR] Model {model_name} not trained.")
            return []
            
        if self.scaler is None:
            print("[ERROR] Scaler not fitted.")
            return []
            
        # 1. Extract Dual-Band Features (using trained filters)
        features = self.extract_dual_band_features(epochs_raw, fs, is_training=False)
        
        # 2. Scale
        features_scaled = self.scaler.transform(features)
        
        # 3. Predict
        model = self.trained_models[model_name]
        preds = model.predict(features_scaled)
        
        return ["Left" if p == 0 else "Right" for p in preds]

    # ---------------------------------------------------------
    # Visualization Generators
    # ---------------------------------------------------------
    def get_prediction(self, model_name):
        if model_name not in self.trained_models: return None, None
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
        return self.y_test, y_pred

    def generate_learning_curve(self, model_name):
        if model_name not in self.trained_models: return None
        model = self.trained_models[model_name]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d0d0d'); ax.set_facecolor('#0d0d0d')
        
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, self.X_train, self.y_train, 
                cv=5, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5)
            )
            train_mean = np.mean(train_scores, axis=1) * 100
            test_mean = np.mean(test_scores, axis=1) * 100
            
            ax.plot(train_sizes, train_mean, 'o-', color="cyan", label="Training")
            ax.plot(train_sizes, test_mean, 'o-', color="magenta", label="CV Score")
            
            ax.set_title(f"LEARNING CURVE: {model_name}", color='white')
            ax.set_xlabel("Samples", color='white'); ax.set_ylabel("Acc %", color='white')
            ax.grid(True, color='#333', linestyle='--'); ax.tick_params(colors='white')
            legend = ax.legend(loc="lower right")
            plt.setp(legend.get_texts(), color='white')
            legend.get_frame().set_facecolor('#1a1a1a')
        except Exception:
            ax.text(0.5, 0.5, "Insufficient Data", color='white', ha='center')
        
        fig.tight_layout()
        return fig

    def generate_loss_curve(self, model_name):
        if "MLP" not in model_name or model_name not in self.trained_models: return None
        model = self.trained_models[model_name]
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d0d0d'); ax.set_facecolor('#0d0d0d')
        ax.plot(model.loss_curve_, color='#00ff41', linewidth=2)
        ax.set_title(f"LOSS CURVE: {model_name}", color='white')
        ax.set_xlabel("Epochs", color='white'); ax.set_ylabel("Loss", color='white')
        ax.grid(True, color='#333', linestyle='--'); ax.tick_params(colors='white')
        fig.tight_layout()
        return fig

    def generate_confusion_matrix(self, model_name):
        if model_name not in self.trained_models: return None
        y_pred = self.trained_models[model_name].predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#0d0d0d'); ax.set_facecolor('#0d0d0d')
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
        cbar = fig.colorbar(im, ax=ax); cbar.ax.yaxis.set_tick_params(color='white')
        
        ax.set_xticks(np.arange(2)); ax.set_yticks(np.arange(2))
        ax.set_xticklabels(['Left', 'Right'], color='white'); ax.set_yticklabels(['Left', 'Right'], color='white')
        ax.set_title(f"CONFUSION MATRIX: {model_name}", color='white', fontweight='bold')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig

    def get_detailed_predictions(self, model_name):
        if model_name not in self.trained_models: return []
        y_pred = self.trained_models[model_name].predict(self.X_test)
        details = []
        for i in range(len(self.y_test)):
            true_lbl = "Left" if self.y_test[i] == 0 else "Right"
            pred_lbl = "Left" if y_pred[i] == 0 else "Right"
            status = "CORRECT" if self.y_test[i] == y_pred[i] else "WRONG"
            details.append((i+1, true_lbl, pred_lbl, status))
        return details

# Standalone Test (Dual Band Check)
if __name__ == "__main__":
    print(">> RUNNING ML_ANALYSIS V7.0 TEST (Dual Band FBCSP + Compatibility)...")
    
    # 1. Simulate Data
    X_train = np.random.randn(100, 3, 500)
    y_train = np.array([0]*50 + [1]*50)
    # Inject Signal Differences
    X_train[0:50, 0, :] *= 5.0 
    
    X_test = np.random.randn(60, 3, 500)
    y_test = np.array([0]*30 + [1]*30)
    X_test[0:30, 0, :] *= 5.0
    
    pipeline = ML_Pipeline()
    # Note: Using run_full_comparison (renamed for GUI compatibility)
    metrics = pipeline.run_full_comparison(X_train, y_train, X_test, y_test, fs=250.0)
    
    print("\n>> METRICS:")
    for k, v in metrics.items():
        print(f"{k}: ACC={v['Accuracy']*100:.1f}%")
        
    print(">> TEST COMPLETE.")