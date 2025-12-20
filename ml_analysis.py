"""
ml_analysis.py

Purpose:
    - Comprehensive Machine Learning Pipeline for BCI.
    - Features:
        1. Feature Fusion: Combines CSP (Spatial) + Temporal Statistics.
        2. Model Comparison: Trains 8 distinct classifiers using Scikit-Learn.
        3. Deep Analysis Tools: Learning Curves (%), Loss Curves, Confusion Matrices.
    - Updates V2.0: 
        - Integrated 8-30Hz Bandpass Filter for better CSP Accuracy.
        - Extended Epoch Window (0.5s to 3.5s).
        - Converted Learning Curve Y-axis to Percentage (0-100%).

Dependencies:
    - numpy, matplotlib, sklearn, scipy
    - csp_scratch (Custom Module)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt

# Scikit-Learn Components
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

# Custom Modules
import csp_scratch

# =========================================================
# 1. Context & Description Helper
# =========================================================
def get_ml_description():
    """
    Returns a descriptive string explaining the Machine Learning comparison.
    """
    description = (
        "--- MACHINE LEARNING CLASSIFICATION ---\n\n"
        "1. OBJECTIVE:\n"
        "   To classify the brain state as 'Left Hand' or 'Right Hand' based on \n"
        "   the spatial features extracted by CSP and temporal statistics.\n\n"
        "2. PREPROCESSING FOR ML:\n"
        "   - Data is internally filtered to 8-30 Hz (Mu/Beta bands) to maximize CSP performance.\n"
        "   - Time Window: 0.5s to 3.5s post-cue.\n\n"
        "3. MODELS COMPARED:\n"
        "   - **Linear Models:** Logistic Regression, Linear SVM (Simple, Fast).\n"
        "   - **Non-Linear Models:** Kernel SVM (RBF), Naive Bayes.\n"
        "   - **Tree-Based:** Decision Tree (Interpretable), Random Forest (Robust).\n"
        "   - **Deep Learning:** MLP (Multi-Layer Perceptron / Neural Network).\n"
    )
    return description

class ML_Pipeline:
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
        
        # Results Containers
        self.trained_models = {}   # Stores the actual model objects
        self.model_metrics = {}    # Stores Accuracy, Precision, Recall, F1
        self.best_model_name = None
        self.scaler = None

    def _internal_bandpass_filter(self, data, fs, lowcut=8.0, highcut=30.0, order=5):
        """
        Applies a specific Bandpass Filter (Butterworth) for ML preprocessing.
        CSP works best in the 8-30 Hz range (Mu and Beta rhythms).
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        # filtfilt applies the filter forward and backward (zero-phase distortion)
        # axis=-1 ensures we filter along the time dimension
        y = filtfilt(b, a, data, axis=-1)
        return y

    def prepare_data(self, raw_data, events, fs, tmin=0.5, tmax=3.5):
        """
        Segments raw EEG data into epochs based on Left/Right cues.
        Applies 8-30Hz filtering to boost CSP accuracy.
        
        Args:
            raw_data (np.array): Raw EEG signals (n_channels, n_samples).
            events (np.array): Event matrix from MNE.
            fs (float): Sampling frequency.
            tmin, tmax (float): Time window relative to cue.
            
        Returns:
            epochs (np.array): (n_trials, n_channels, n_samples)
            labels (np.array): (n_trials,)
        """
        EV_LEFT = 769
        EV_RIGHT = 770
        
        trials = []
        labels = []
        
        offset_start = int(tmin * fs)
        offset_end = int(tmax * fs)
        n_samples_raw = raw_data.shape[1]
        
        for ev in events:
            idx = ev[0]
            eid = ev[2]
            
            # Filter strictly for Motor Imagery classes
            if eid not in [EV_LEFT, EV_RIGHT]:
                continue
            
            start = idx + offset_start
            end = idx + offset_end
            
            # Boundary check
            if start < 0 or end > n_samples_raw:
                continue
                
            # 1. Extract Epoch
            epoch_data = raw_data[:, start:end]
            
            # 2. Apply ML-Specific Filtering (8-30 Hz)
            # This is crucial. Even if the global data is 0.5-30Hz, 
            # CSP needs cleaner Mu/Beta bands to find good spatial patterns.
            epoch_data_filtered = self._internal_bandpass_filter(epoch_data, fs, 8.0, 30.0)
            
            trials.append(epoch_data_filtered)
            
            # Map Labels: 769->0 (Left), 770->1 (Right)
            labels.append(0 if eid == EV_LEFT else 1)
            
        return np.array(trials), np.array(labels)

    def extract_combined_features(self, epochs, labels, is_training=True):
        """
        Performs Feature Fusion: CSP Features + Temporal Features.
        
        Args:
            epochs (np.array): EEG Epochs.
            labels (np.array): Class labels (needed for CSP fitting).
            is_training (bool): If True, fits the CSP model.
            
        Returns:
            features (np.array): Combined feature matrix (n_trials, n_features).
        """
        # 1. Spatial Features (CSP)
        if is_training:
            self.csp = csp_scratch.CSP_Scratch(n_components=2)
            self.csp.fit(epochs, labels)
            
        csp_features = self.csp.transform(epochs)
        
        # 2. Temporal Features (Mean, Var, Skew, Kurtosis)
        if is_training:
            self.temporal_extractor = csp_scratch.TemporalFeatureExtractor()
            
        temp_features = self.temporal_extractor.transform(epochs)
        
        # 3. Concatenate (Fuse) Features
        # Shape: (n_trials, n_csp_components + n_temporal_metrics)
        combined_features = np.hstack([csp_features, temp_features])
        
        return combined_features

    def run_full_comparison(self, epochs, labels, test_size=0.25):
        """
        Extracts features and trains 8 Machine Learning models.
        """
        # 1. Split Data (Stratified to maintain class balance)
        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            epochs, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 2. Extract Combined Features (CSP + Temporal) on Training Data
        print("[ML] Extracting features for Training Set...")
        X_train_feat = self.extract_combined_features(X_train_raw, self.y_train, is_training=True)
        
        # 3. Extract Features on Test Data (Using fitted extractors)
        print("[ML] Extracting features for Test Set...")
        X_test_feat = self.extract_combined_features(X_test_raw, self.y_test, is_training=False)
        
        # 4. Standardization (Crucial for SVM and MLP)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(X_train_feat)
        self.X_test = self.scaler.transform(X_test_feat)
        
        print(f"[ML INFO] Final Feature Matrix Shape: {self.X_train.shape}")
        
        # 5. Define The 8 Classifiers
        models = {
            'SVM (Linear)': SVC(kernel='linear', C=1.0),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Naive Bayes': GaussianNB(),
            'Kernel SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale'),
            'RF (Tuned)': RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=42),
            # MLP: Max Iter 3000 for convergence
            'MLP (Neural Network)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, 
                                                  activation='relu', solver='adam', random_state=42)
        }
        
        self.results = {}
        self.model_metrics = {}
        self.trained_models = {}

        # 6. Training Loop
        for name, clf in models.items():
            try:
                print(f"[ML] Training {name}...")
                clf.fit(self.X_train, self.y_train)
                
                # Evaluation
                pred = clf.predict(self.X_test)
                acc = accuracy_score(self.y_test, pred)
                prec, rec, f1, _ = precision_recall_fscore_support(self.y_test, pred, average='macro', zero_division=0)
                
                # Store
                self.model_metrics[name] = {
                    'Accuracy': acc, 
                    'Precision': prec, 
                    'Recall': rec, 
                    'F1': f1
                }
                self.results[name] = acc
                self.trained_models[name] = clf
                
            except Exception as e:
                print(f"[ML ERROR] Failed to train {name}: {e}")
                self.results[name] = 0.0

        # 7. Identify Best Model
        if self.results:
            self.best_model_name = max(self.results, key=self.results.get)
            best_acc = self.results[self.best_model_name]
            print(f"\n[ML RESULT] Best Model: {self.best_model_name} ({best_acc*100:.2f}%)")
        
        return self.model_metrics

    # =========================================================
    # KEY FUNCTION: get_prediction
    # =========================================================
    def get_prediction(self, model_name):
        """
        Returns y_test and y_pred for a specific model name to build Confusion Matrix.
        """
        if model_name not in self.trained_models:
            return None, None
            
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
            
        return self.y_test, y_pred

    # =========================================================
    # VISUALIZATION GENERATORS (PERCENTAGE UPDATED)
    # =========================================================

    def generate_learning_curve(self, model_name):
        """
        Generates a Learning Curve plot.
        Y-AXIS IS NOW PERCENTAGE (0-100%).
        """
        if model_name not in self.trained_models:
            return None
            
        model = self.trained_models[model_name]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        
        try:
            # FIX: Start at 0.2 to avoid 'ValueError: 1 class' on small datasets
            train_sizes, train_scores, test_scores = learning_curve(
                model, self.X_train, self.y_train, 
                cv=5, n_jobs=-1, 
                train_sizes=np.linspace(0.2, 1.0, 5)
            )
            
            # CONVERT TO PERCENTAGE (Multiply by 100)
            train_mean = np.mean(train_scores, axis=1) * 100
            train_std = np.std(train_scores, axis=1) * 100
            test_mean = np.mean(test_scores, axis=1) * 100
            test_std = np.std(test_scores, axis=1) * 100
            
            # Plot Training Accuracy
            ax.plot(train_sizes, train_mean, 'o-', color="cyan", label="Training Score")
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="cyan")
            
            # Plot Validation Accuracy
            ax.plot(train_sizes, test_mean, 'o-', color="magenta", label="Cross-Validation Score")
            ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="magenta")
            
            # Labels and Styling
            ax.set_title(f"LEARNING CURVE (ACCURACY %): {model_name}", color='white', fontweight='bold')
            ax.set_xlabel("Training Examples", color='white')
            ax.set_ylabel("Accuracy Score (%)", color='white')
            ax.set_ylim(0, 105) # Force Y-axis to 0-100 range
            ax.tick_params(colors='white')
            ax.grid(True, color='#333', linestyle='--')
            
            # Legend
            legend = ax.legend(loc="lower right")
            plt.setp(legend.get_texts(), color='white')
            legend.get_frame().set_facecolor('#1a1a1a')
            
        except ValueError as e:
            print(f"[ML WARN] Could not generate learning curve: {e}")
            ax.text(0.5, 0.5, "Insufficient Data for Learning Curve", 
                    color='white', ha='center', va='center')
        
        fig.tight_layout()
        return fig

    def generate_loss_curve(self, model_name):
        """Generates a Loss Curve (MLP only). Keep Loss as Value (Standard Practice)."""
        if "MLP" not in model_name or model_name not in self.trained_models:
            return None
            
        model = self.trained_models[model_name]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        
        ax.plot(model.loss_curve_, color='#00ff41', linewidth=2)
        
        ax.set_title(f"LOSS CURVE: {model_name}", color='white', fontweight='bold')
        ax.set_xlabel("Iterations (Epochs)", color='white')
        ax.set_ylabel("Loss (Cross-Entropy)", color='white')
        ax.tick_params(colors='white')
        ax.grid(True, color='#333', linestyle='--')
        
        fig.tight_layout()
        return fig

    def generate_confusion_matrix(self, model_name):
        """Generates a Confusion Matrix Heatmap."""
        if model_name not in self.trained_models:
            return None
            
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='white')
        
        classes = ['Left Hand', 'Right Hand']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, color='white')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, color='white')
        
        ax.set_title(f"CONFUSION MATRIX: {model_name}", color='white', fontweight='bold')
        ax.set_ylabel('True Label', color='white')
        ax.set_xlabel('Predicted Label', color='white')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig

    def get_detailed_predictions(self, model_name):
        """Returns detailed prediction list for GUI table."""
        if model_name not in self.trained_models:
            return []
            
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
        
        details = []
        for i in range(len(self.y_test)):
            true_lbl = "Left" if self.y_test[i] == 0 else "Right"
            pred_lbl = "Left" if y_pred[i] == 0 else "Right"
            status = "CORRECT" if self.y_test[i] == y_pred[i] else "WRONG"
            details.append((i+1, true_lbl, pred_lbl, status))
            
        return details

# =========================================================
# Standalone Testing Block
# =========================================================
if __name__ == "__main__":
    print(">> RUNNING ML_ANALYSIS STANDALONE TEST...")
    # Generate Synthetic Data
    X = np.random.randn(100, 3, 500)
    y = np.array([0]*50 + [1]*50)
    X[0:50, 0, :] *= 5.0 
    X[50:, 2, :] *= 5.0 
    
    pipeline = ML_Pipeline()
    metrics = pipeline.run_full_comparison(X, y, test_size=0.3)
    
    print("\n>> METRICS SUMMARY:")
    for name, m in metrics.items():
        print(f"{name}: ACC={m['Accuracy']:.2f}")
    
    # Test Learning Curve Generation
    print("\n>> Generating Learning Curve...")
    fig = pipeline.generate_learning_curve(pipeline.best_model_name)
    if fig:
        plt.show()
        
    print(">> TEST COMPLETE.")