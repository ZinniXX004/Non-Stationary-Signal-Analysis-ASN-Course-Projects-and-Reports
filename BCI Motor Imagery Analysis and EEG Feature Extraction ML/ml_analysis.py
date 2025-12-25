"""
ml_analysis.py (VERSION 5.0 - EXPLICIT SPLIT SUPPORT)

Purpose:
    - Comprehensive Machine Learning Pipeline for Motor Imagery BCI.
    - Features:
        1. Feature Fusion: Combines Spatial (CSP) + Temporal Features.
        2. Explicit Training/Testing: Supports separate datasets for Training (e.g., Session 1+2)
           and Testing (e.g., Session 3) to ensure 100% test data usage in evaluation.
        3. Inference Mode: Predicts on unlabeled Evaluation data (Session 4+5).
        4. Deep Analysis: Learning Curves, Loss Curves, Confusion Matrices.
    
    - Design Philosophy:
        - Maximize readability.
        - Strict separation of Training and Testing logic.

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

# 2. CONTEXT & DESCRIPTION HELPER
def get_ml_description():
    """
    Returns a descriptive string explaining the Machine Learning strategy.
    """
    description = (
        "--- MACHINE LEARNING (EXPLICIT TRAIN/TEST SPLIT) ---\n\n"
        "1. TRAINING PHASE:\n"
        "   - Uses 100% of data from loaded Training Files (e.g., B01T + B02T).\n"
        "   - Extracts CSP Spatial + Temporal features.\n"
        "   - Fits the Scaler (Z-Score) and CSP Filters.\n\n"
        "2. TESTING PHASE:\n"
        "   - Uses 100% of data from the designated Test File (e.g., B03T).\n"
        "   - Evaluates the model on unseen data (Cross-Session Validation).\n"
        "   - Confusion Matrix reflects ALL trials in the Test File.\n\n"
        "3. INFERENCE PHASE:\n"
        "   - Predicts classes for unlabeled Evaluation Data (E-Files).\n"
    )
    return description

# 3. MACHINE LEARNING PIPELINE CLASS
class ML_Pipeline:
    """
    Manages the end-to-end Machine Learning process:
    Data Preparation -> Feature Extraction -> Training -> Evaluation -> Inference.
    """
    
    def __init__(self):
        """
        Initialize the Machine Learning Pipeline.
        """
        # Feature Extractors
        self.csp = None
        self.temporal_extractor = None
        
        # Data Containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Scaling Object
        self.scaler = None
        
        # Results
        self.trained_models = {}
        self.model_metrics = {}
        self.best_model_name = None

    # Internal Helper: Bandpass Filter
    def _internal_bandpass_filter(self, data, fs, lowcut=8.0, highcut=30.0, order=5):
        """
        Applies a specific Bandpass Filter (Butterworth) for ML preprocessing.
        CSP works optimally in the 8-30 Hz range (Mu and Beta rhythms).
        """
        nyquist_freq = 0.5 * fs
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        
        b, a = butter(order, [low, high], btype='band')
        
        # Apply filter along time axis (last axis)
        y = filtfilt(b, a, data, axis=-1)
        return y

    # Data Preparation (Segmentation)
    def prepare_data(self, raw_data, events, fs, tmin=0.5, tmax=3.5, mode='train'):
        """
        Segments raw EEG data into epochs.
        Ensures NO valid trials are dropped unless they are artifacts or out of bounds.
        
        Args:
            raw_data (np.array): Raw EEG signals.
            events (np.array): Event matrix.
            fs (float): Sampling frequency.
            tmin, tmax (float): Time window relative to cue.
            mode (str): 'train' (769/770) or 'inference' (783).
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
            
            # Boundary check
            if start < 0 or end > n_samples_raw:
                continue
            
            # Extract and Filter
            epoch_data = raw_data[:, start:end]
            epoch_filtered = self._internal_bandpass_filter(epoch_data, fs, 8.0, 30.0)
            
            trials.append(epoch_filtered)
            
            # Labels
            if mode == 'train':
                labels.append(0 if eid == EV_LEFT else 1)
            else:
                labels.append(-1)
            
        return np.array(trials), np.array(labels)

    # Feature Extraction (Fusion)
    def extract_combined_features(self, epochs, labels=None, is_training=True):
        """
        Performs Feature Fusion: Spatial (CSP) + Temporal.
        
        Args:
            is_training (bool): 
                - If True, FITS the CSP filters on the provided epochs.
                - If False, APPLIES existing filters (Transform only).
        """
        # 1. Spatial Features (CSP)
        if is_training:
            self.csp = csp_scratch.CSP_Scratch(n_components=2)
            if labels is None:
                raise ValueError("Labels required for CSP training.")
            self.csp.fit(epochs, labels)
            
        if self.csp is None:
            raise RuntimeError("CSP not trained! Cannot extract features.")
            
        csp_features = self.csp.transform(epochs)
        
        # 2. Temporal Features
        if is_training:
            self.temporal_extractor = csp_scratch.TemporalFeatureExtractor()
            
        temp_features = self.temporal_extractor.transform(epochs)
        
        # 3. Concatenate
        combined_features = np.hstack([csp_features, temp_features])
        return combined_features

    # EXPLICIT TRAIN/TEST COMPARISON RUNNER
    def run_explicit_comparison(self, X_train_raw, y_train_raw, X_test_raw, y_test_raw):
        """
        Trains on X_train_raw (100% of Training Files), Tests on X_test_raw (100% of Test File).
        NO internal splitting is performed.
        
        Args:
            X_train_raw: Raw epochs for training (e.g., from B01T + B02T).
            y_train_raw: Labels for training.
            X_test_raw: Raw epochs for testing (e.g., from B03T).
            y_test_raw: Labels for testing.
            
        Returns:
            dict: Model metrics.
        """
        # 1. Feature Extraction (Fit on Train, Transform on Test)
        print(f"[ML] Processing Training Data ({len(y_train_raw)} trials)...")
        X_train_feat = self.extract_combined_features(X_train_raw, y_train_raw, is_training=True)
        
        print(f"[ML] Processing Test Data ({len(y_test_raw)} trials)...")
        X_test_feat = self.extract_combined_features(X_test_raw, is_training=False)
        
        # 2. Standardization (Fit on Train, Transform on Test)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train_feat)
        self.X_test = self.scaler.transform(X_test_feat)
        
        # Store labels globally for other methods (like confusion matrix)
        self.y_train = y_train_raw
        self.y_test = y_test_raw
        
        print(f"[ML INFO] Feature Matrix: Train={self.X_train.shape}, Test={self.X_test.shape}")
        
        # 3. Define Models
        models = {
            'SVM (Linear)': SVC(kernel='linear', C=1.0),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Naive Bayes': GaussianNB(),
            'Kernel SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale'),
            'RF (Tuned)': RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=42),
            'MLP (Neural Network)': MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=5000, 
                                                  activation='relu', solver='adam', random_state=42)
        }
        
        self.results = {}
        self.model_metrics = {}
        self.trained_models = {}

        # 4. Training Loop
        for name, clf in models.items():
            try:
                # print(f"[ML] Training {name}...") # Optional debug print
                clf.fit(self.X_train, self.y_train)
                
                # Predict on Test Set
                pred = clf.predict(self.X_test)
                
                # Metrics
                acc = accuracy_score(self.y_test, pred)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    self.y_test, pred, average='macro', zero_division=0
                )
                
                self.model_metrics[name] = {
                    'Accuracy': acc, 
                    'Precision': prec, 
                    'Recall': rec, 
                    'F1': f1
                }
                self.results[name] = acc
                self.trained_models[name] = clf
                
            except Exception as e:
                print(f"[ML ERROR] {name}: {e}")
                self.results[name] = 0.0

        # 5. Identify Best Model
        if self.results:
            self.best_model_name = max(self.results, key=self.results.get)
            best_acc = self.results[self.best_model_name]
            print(f"\n[ML RESULT] Best Model: {self.best_model_name} ({best_acc*100:.2f}%)")
        
        return self.model_metrics

    # Inference Runner (Unlabeled Data)
    def predict_new_data(self, epochs_raw, model_name):
        """
        Predicts class for new, unlabeled data (e.g., Evaluation set).
        """
        if model_name not in self.trained_models:
            print(f"[ERROR] Model {model_name} not trained yet.")
            return []
            
        if self.scaler is None:
            print("[ERROR] Scaler not found. Training must be done first.")
            return []
            
        # 1. Extract Features (Using existing CSP filters)
        features = self.extract_combined_features(epochs_raw, is_training=False)
        
        # 2. Scale Features
        features_scaled = self.scaler.transform(features)
        
        # 3. Predict
        model = self.trained_models[model_name]
        preds = model.predict(features_scaled)
        
        # 4. Convert to String
        return ["Left" if p == 0 else "Right" for p in preds]

    # Visualization Helpers
    def get_prediction(self, model_name):
        """Returns y_test and y_pred for Confusion Matrix."""
        if model_name not in self.trained_models: return None, None
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
        return self.y_test, y_pred

    def generate_learning_curve(self, model_name):
        """
        Generates Learning Curve.
        Note: Learning curve typically uses Cross-Validation on the TRAINING set
        to show model stability/fitting status, not the Test set performance.
        """
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
            
            ax.set_title(f"LEARNING CURVE: {model_name}", color='white', fontweight='bold')
            ax.set_xlabel("Training Examples", color='white')
            ax.set_ylabel("Accuracy (%)", color='white')
            ax.set_ylim(0, 105)
            ax.tick_params(colors='white')
            ax.grid(True, color='#333', linestyle='--')
            
            legend = ax.legend(loc="lower right")
            plt.setp(legend.get_texts(), color='white')
            legend.get_frame().set_facecolor('#1a1a1a')
            
        except Exception:
            ax.text(0.5, 0.5, "Insufficient Data for Curve", color='white', ha='center')
        
        fig.tight_layout()
        return fig

    def generate_loss_curve(self, model_name):
        """Generates Loss Curve for MLP."""
        if "MLP" not in model_name or model_name not in self.trained_models: return None
        model = self.trained_models[model_name]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d0d0d'); ax.set_facecolor('#0d0d0d')
        
        ax.plot(model.loss_curve_, color='#00ff41', linewidth=2)
        ax.set_title(f"LOSS CURVE: {model_name}", color='white', fontweight='bold')
        ax.set_xlabel("Epochs", color='white')
        ax.set_ylabel("Loss", color='white')
        ax.tick_params(colors='white')
        ax.grid(True, color='#333', linestyle='--')
        
        fig.tight_layout()
        return fig

    def generate_confusion_matrix(self, model_name):
        """Generates Confusion Matrix Heatmap."""
        if model_name not in self.trained_models: return None
        y_pred = self.trained_models[model_name].predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#0d0d0d'); ax.set_facecolor('#0d0d0d')
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='white')
        
        classes = ['Left', 'Right']
        ax.set_xticks(np.arange(2)); ax.set_yticks(np.arange(2))
        ax.set_xticklabels(classes, color='white'); ax.set_yticklabels(classes, color='white')
        
        ax.set_title(f"CONFUSION MATRIX: {model_name}", color='white', fontweight='bold')
        ax.set_ylabel('True Label', color='white')
        ax.set_xlabel('Predicted Label', color='white')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig

    def get_detailed_predictions(self, model_name):
        """Returns predictions for the Test Set (Validation)."""
        if model_name not in self.trained_models: return []
        y_pred = self.trained_models[model_name].predict(self.X_test)
        
        details = []
        for i in range(len(self.y_test)):
            true_lbl = "Left" if self.y_test[i] == 0 else "Right"
            pred_lbl = "Left" if y_pred[i] == 0 else "Right"
            status = "CORRECT" if self.y_test[i] == y_pred[i] else "WRONG"
            details.append((i+1, true_lbl, pred_lbl, status))
        return details

# Standalone Test (Training + Testing + Inference)
if __name__ == "__main__":
    print(">> RUNNING ML_ANALYSIS V5.0 TEST (Explicit Split)...")
    
    # 1. Simulate Train Data (e.g., File 1 & 2)
    # 200 Trials Total
    X_train = np.random.randn(200, 3, 500)
    y_train = np.array([0]*100 + [1]*100)
    X_train[0:100, 0, :] *= 5.0 # Class 0 High Var
    
    # 2. Simulate Test Data (e.g., File 3)
    # 120 Trials Total
    X_test = np.random.randn(120, 3, 500)
    y_test = np.array([0]*60 + [1]*60)
    X_test[0:60, 0, :] *= 5.0 # Same pattern
    
    # 3. Simulate Eval Data (Unlabeled)
    X_eval = np.random.randn(10, 3, 500)
    X_eval[0:5, 0, :] *= 5.0
    
    pipeline = ML_Pipeline()
    
    print(">> Phase 1: Explicit Training & Testing...")
    metrics = pipeline.run_explicit_comparison(X_train, y_train, X_test, y_test)
    
    for k, v in metrics.items():
        print(f"{k}: ACC={v['Accuracy']*100:.1f}%")
        
    print(f">> Phase 2: Inference (using best model)...")
    preds = pipeline.predict_new_data(X_eval, pipeline.best_model_name)
    print(f"Predictions: {preds}")
    
    print(">> TEST COMPLETE.")