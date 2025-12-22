"""
ml_analysis.py

Purpose:
    - Comprehensive Machine Learning Pipeline for Motor Imagery BCI.
    - Features:
        1. Feature Fusion: Combines Spatial (CSP) + Temporal (Mean, Variance, Skewness, Kurtosis).
           * Note: ERD/ERS magnitude is implicitly captured by the Variance/Power temporal features.
        2. Model Comparison: Trains and evaluates 8 distinct classifiers using Scikit-Learn.
        3. Deep Analysis Tools: Generates Learning Curves, Loss Curves, and Confusion Matrices.
    - Dataset Handling: Designed to split labeled Training Data for valid performance estimation.
    - Design Philosophy: Maximum code readability and explicit logical steps.

Dependencies:
    - numpy
    - matplotlib
    - sklearn (Scikit-Learn)
    - scipy (Signal Processing)
    - csp_scratch (Custom Module containing CSP and Temporal Extractors)
"""

# =========================================================
# 1. IMPORTS
# =========================================================
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

# =========================================================
# 2. CONTEXT & DESCRIPTION HELPER
# =========================================================
def get_ml_description():
    """
    Returns a descriptive string explaining the Machine Learning comparison strategy.
    Used by the GUI to display educational information to the user.
    
    Returns:
        str: Description of the pipeline.
    """
    description = (
        "--- MACHINE LEARNING CLASSIFICATION ---\n\n"
        "1. OBJECTIVE:\n"
        "   To classify the brain state as 'Left Hand' or 'Right Hand' based on \n"
        "   a FUSED FEATURE VECTOR combining Spatial and Temporal characteristics.\n\n"
        "2. FEATURE FUSION STRATEGY:\n"
        "   - **Spatial Features (CSP):** Maximizes variance difference between classes.\n"
        "   - **Temporal Features:** Mean, Variance (ERD/ERS proxy), Skewness, Kurtosis\n"
        "     extracted from C3, Cz, and C4 channels.\n\n"
        "3. MODELS COMPARED:\n"
        "   - **Linear Models:** Logistic Regression, Linear SVM (Simple, Fast).\n"
        "   - **Non-Linear Models:** Kernel SVM (RBF), Naive Bayes.\n"
        "   - **Tree-Based:** Decision Tree (Interpretable), Random Forest (Robust).\n"
        "   - **Deep Learning:** MLP (Multi-Layer Perceptron / Neural Network).\n"
    )
    return description

# =========================================================
# 3. MACHINE LEARNING PIPELINE CLASS
# =========================================================
class ML_Pipeline:
    """
    Manages the end-to-end Machine Learning process:
    Data Preparation -> Feature Extraction -> Training -> Evaluation -> Visualization.
    """
    
    def __init__(self):
        """
        Initialize the Machine Learning Pipeline.
        Sets up containers for data, models, and results.
        """
        # Feature Extractors (Initialized during training)
        self.csp = None
        self.temporal_extractor = None
        
        # Data Containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Scaling Object (for Z-Score Normalization)
        self.scaler = None
        
        # Results Containers
        self.trained_models = {}   # Stores the actual trained model objects
        self.model_metrics = {}    # Stores dictionary of results (Accuracy, F1, etc.)
        self.best_model_name = None

    # ---------------------------------------------------------
    # Internal Helper: Bandpass Filter
    # ---------------------------------------------------------
    def _internal_bandpass_filter(self, data, fs, lowcut=8.0, highcut=30.0, order=5):
        """
        Applies a specific Bandpass Filter (Butterworth) for ML preprocessing.
        
        Why 8-30 Hz?
        Common Spatial Pattern (CSP) algorithms work optimally when the signal 
        contains only the specific brain rhythms associated with movement 
        (Mu: 8-13 Hz and Beta: 13-30 Hz). Excluding low frequencies (<8Hz) removes 
        artifacts like EOG (eye movements), and excluding high frequencies (>30Hz) 
        removes EMG (muscle noise).
        
        Args:
            data (np.array): Input EEG data.
            fs (float): Sampling frequency.
            lowcut (float): Lower cutoff frequency.
            highcut (float): Upper cutoff frequency.
            order (int): Order of the filter.
            
        Returns:
            np.array: Filtered data.
        """
        nyquist_freq = 0.5 * fs
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        
        # Design the Butterworth filter
        b, a = butter(order, [low, high], btype='band')
        
        # Apply the filter using filtfilt
        # filtfilt applies the filter forward and backward to ensure zero-phase distortion.
        # axis=-1 ensures we filter along the time dimension.
        y = filtfilt(b, a, data, axis=-1)
        
        return y

    # ---------------------------------------------------------
    # Data Preparation (Segmentation)
    # ---------------------------------------------------------
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
        # Define Standard Class Codes
        EV_LEFT = 769
        EV_RIGHT = 770
        
        trials = []
        labels = []
        
        # Convert time (seconds) to samples
        offset_start = int(tmin * fs)
        offset_end = int(tmax * fs)
        
        n_samples_raw = raw_data.shape[1]
        
        # Iterate through all events found in the file
        for ev in events:
            idx = ev[0] # Sample index of the event
            eid = ev[2] # Event ID code
            
            # Filter strictly for Motor Imagery classes (Left vs Right)
            if eid not in [EV_LEFT, EV_RIGHT]:
                continue
            
            # Calculate start and end indices for slicing
            start = idx + offset_start
            end = idx + offset_end
            
            # Boundary check: Ensure we don't try to slice outside the data array
            if start < 0 or end > n_samples_raw:
                continue
                
            # 1. Extract the Raw Epoch
            epoch_data = raw_data[:, start:end]
            
            # 2. Apply ML-Specific Filtering (8-30 Hz)
            # This is crucial. Even if the global data is 0.5-30Hz, 
            # CSP needs cleaner Mu/Beta bands to find optimal spatial patterns.
            epoch_data_filtered = self._internal_bandpass_filter(epoch_data, fs, 8.0, 30.0)
            
            trials.append(epoch_data_filtered)
            
            # Map Labels to 0 and 1 for Binary Classification
            # 769 (Left) -> 0
            # 770 (Right) -> 1
            labels.append(0 if eid == EV_LEFT else 1)
            
        return np.array(trials), np.array(labels)

    # ---------------------------------------------------------
    # Feature Extraction (Fusion)
    # ---------------------------------------------------------
    def extract_combined_features(self, epochs, labels, is_training=True):
        """
        Performs Feature Fusion: Combines Spatial Features (CSP) and Temporal Features.
        
        Args:
            epochs (np.array): EEG Epochs (n_trials, 3, n_samples).
            labels (np.array): Class labels (needed for CSP fitting during training).
            is_training (bool): If True, the CSP model calculates new filters (Fit).
                                If False, it applies existing filters (Transform only).
            
        Returns:
            features (np.array): Combined feature matrix (n_trials, n_features).
        """
        # 1. Spatial Features (CSP)
        # We use the custom CSP_Scratch module created previously
        if is_training:
            self.csp = csp_scratch.CSP_Scratch(n_components=2)
            self.csp.fit(epochs, labels)
            
        # Transform data into Log-Variance of Spatially Filtered components
        csp_features = self.csp.transform(epochs)
        
        # 2. Temporal Features (Mean, Var, Skew, Kurtosis)
        # We use the custom TemporalFeatureExtractor
        if is_training:
            self.temporal_extractor = csp_scratch.TemporalFeatureExtractor()
            
        temp_features = self.temporal_extractor.transform(epochs)
        
        # 3. Concatenate (Fuse) Features
        # We horizontally stack the spatial and temporal feature matrices.
        # Shape becomes: (n_trials, n_csp_components + n_temporal_metrics)
        combined_features = np.hstack([csp_features, temp_features])
        
        return combined_features

    # ---------------------------------------------------------
    # Main Comparison Runner
    # ---------------------------------------------------------
    def run_full_comparison(self, epochs, labels, test_size=0.25):
        """
        Extracts features and trains 8 Machine Learning models.
        
        Args:
            epochs (np.array): EEG data.
            labels (np.array): Target classes.
            test_size (float): Fraction of data used for testing (Evaluation).
            
        Returns:
            dict: Model metrics dictionary containing accuracy, precision, recall, etc.
        """
        # 1. Split Data (Stratified to maintain class balance)
        # Stratify is important because dataset 2b often has perfectly balanced classes (60 vs 60),
        # we want to ensure the train/test splits reflect this balance.
        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            epochs, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 2. Extract Combined Features (CSP + Temporal) on Training Data
        print("[ML] Extracting features for Training Set...")
        X_train_feat = self.extract_combined_features(X_train_raw, self.y_train, is_training=True)
        
        # 3. Extract Features on Test Data (Using fitted extractors)
        # Important: We do NOT re-fit CSP on test data. We use filters learned from training.
        print("[ML] Extracting features for Test Set...")
        X_test_feat = self.extract_combined_features(X_test_raw, self.y_test, is_training=False)
        
        # 4. Standardization (Crucial for SVM and MLP)
        # Z-Score Normalization ensures all features contribute equally.
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
            # MLP: Increased max_iter to 3000 to ensure convergence on smaller datasets
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
                
                # Fit the model
                clf.fit(self.X_train, self.y_train)
                
                # Predict on Test Set
                pred = clf.predict(self.X_test)
                
                # Calculate Metrics
                acc = accuracy_score(self.y_test, pred)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    self.y_test, pred, average='macro', zero_division=0
                )
                
                # Store Results
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
    # KEY HELPER: GET PREDICTION FOR VISUALIZATION
    # =========================================================
    def get_prediction(self, model_name):
        """
        Returns y_test (True Labels) and y_pred (Predicted Labels) 
        for a specific model name. This is used to build Confusion Matrices.
        
        Args:
            model_name (str): The key name of the model (e.g., 'SVM (Linear)').
            
        Returns:
            tuple: (y_true, y_pred) or (None, None) if model is not found.
        """
        if model_name not in self.trained_models:
            return None, None
            
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
            
        return self.y_test, y_pred

    # =========================================================
    # VISUALIZATION GENERATORS (For GUI Embedding)
    # =========================================================

    def generate_learning_curve(self, model_name):
        """
        Generates a Learning Curve plot.
        The Y-axis is formatted as PERCENTAGE (0-100%).
        
        Note: The training size starts at 20% to avoid errors with 
        small datasets where cross-validation splits might lack classes.
        
        Returns:
            matplotlib.figure.Figure: The figure object ready to be plotted.
        """
        if model_name not in self.trained_models:
            return None
            
        model = self.trained_models[model_name]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Styling for Dark Theme
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        
        try:
            # Generate Learning Curve Data using 5-Fold Cross Validation
            # start=0.2 means we start training with 20% of the data, up to 100%
            train_sizes, train_scores, test_scores = learning_curve(
                model, self.X_train, self.y_train, 
                cv=5, n_jobs=-1, 
                train_sizes=np.linspace(0.2, 1.0, 5)
            )
            
            # Convert scores to Percentage (Multiply by 100)
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
            # Fallback in case dataset is too small for CV
            print(f"[ML WARN] Could not generate learning curve: {e}")
            ax.text(0.5, 0.5, "Insufficient Data for Learning Curve", 
                    color='white', ha='center', va='center')
        
        fig.tight_layout()
        return fig

    def generate_loss_curve(self, model_name):
        """
        Generates a Loss Curve (MSE/Log-Loss per Epoch).
        This is only available for MLP (Neural Network).
        
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        if "MLP" not in model_name or model_name not in self.trained_models:
            return None
            
        model = self.trained_models[model_name]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Styling for Dark Theme
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        
        # Plot Loss History
        ax.plot(model.loss_curve_, color='#00ff41', linewidth=2)
        
        ax.set_title(f"LOSS CURVE: {model_name}", color='white', fontweight='bold')
        ax.set_xlabel("Iterations (Epochs)", color='white')
        ax.set_ylabel("Loss (Cross-Entropy)", color='white')
        ax.tick_params(colors='white')
        ax.grid(True, color='#333', linestyle='--')
        
        fig.tight_layout()
        return fig

    def generate_confusion_matrix(self, model_name):
        """
        Generates a Confusion Matrix Heatmap for the test set predictions.
        
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        if model_name not in self.trained_models:
            return None
            
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Styling for Dark Theme
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        
        # Draw Heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
        
        # Add Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='white')
        
        # Labels
        classes = ['Left Hand', 'Right Hand']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, color='white')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, color='white')
        
        ax.set_title(f"CONFUSION MATRIX: {model_name}", color='white', fontweight='bold')
        ax.set_ylabel('True Label', color='white')
        ax.set_xlabel('Predicted Label', color='white')
        
        # Annotate Cells with Counts
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        return fig

    def get_detailed_predictions(self, model_name):
        """
        Returns a detailed list of predictions for the GUI table.
        Format: Tuple (Trial_ID, True_Label, Predicted_Label, Status)
        """
        if model_name not in self.trained_models:
            return []
            
        model = self.trained_models[model_name]
        y_pred = model.predict(self.X_test)
        
        details = []
        for i in range(len(self.y_test)):
            true_lbl = "Left" if self.y_test[i] == 0 else "Right"
            pred_lbl = "Left" if y_pred[i] == 0 else "Right"
            status = "CORRECT" if self.y_test[i] == y_pred[i] else "WRONG"
            
            # Using i+1 as a relative Trial ID for the test set
            details.append((i+1, true_lbl, pred_lbl, status))
            
        return details

# =========================================================
# Standalone Testing Block
# =========================================================
if __name__ == "__main__":
    print(">> RUNNING ML_ANALYSIS STANDALONE TEST...")
    
    # 1. Generate Synthetic Data (3 Channels, 100 trials, 500 samples)
    # This simulates a small dataset to verify pipeline stability.
    X = np.random.randn(100, 3, 500)
    y = np.array([0]*50 + [1]*50)
    
    # Inject Signal Differences (to make it learnable)
    # Class 0: High Variance Ch0, Low Ch2
    X[0:50, 0, :] *= 5.0 
    
    # Class 1: Low Variance Ch0, High Ch2
    X[50:, 2, :] *= 5.0 
    
    # 2. Initialize Pipeline
    pipeline = ML_Pipeline()
    
    # 3. Run Comparison
    print(">> Training Models...")
    metrics = pipeline.run_full_comparison(X, y, test_size=0.3)
    
    print("\n>> METRICS SUMMARY:")
    for name, m in metrics.items():
        print(f"{name:<25}: ACC={m['Accuracy']*100:.2f}%")
    
    # 4. Test Learning Curve Generation
    print("\n>> Generating Learning Curve...")
    fig = pipeline.generate_learning_curve(pipeline.best_model_name)
    if fig:
        plt.show()
        
    print(">> TEST COMPLETE.")