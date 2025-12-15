"""
ml_analysis.py

Purpose:
    - Machine Learning Pipeline (Feature Extraction + Classification).
    - Robust comparison of 8 models using Scikit-Learn.
    - Replaced TensorFlow with Scikit-Learn MLPClassifier to fix DLL/Runtime errors.
    - Updated: Increased max_iter to 3000 to fix ConvergenceWarning.

Dependencies:
    - numpy, matplotlib, sklearn
    - csp_scratch (Custom Module)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Scikit-Learn (Classical ML & Neural Network)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier  # Neural Network
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Custom CSP
import csp_scratch

class ML_Pipeline:
    def __init__(self):
        self.csp = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Store trained models and their results
        self.trained_models = {}
        self.results = {}
        self.best_model_name = None

    def prepare_data(self, raw_data, events, fs, tmin=0.5, tmax=2.5):
        """
        Segments raw EEG data into epochs based on Left/Right cues.
        Window: Typically Motor Imagery happens 0.5s to 2.5s after cue.
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
            
            # Filter strictly for Left (769) and Right (770)
            if eid not in [EV_LEFT, EV_RIGHT]:
                continue
            
            start = idx + offset_start
            end = idx + offset_end
            
            # Boundary check
            if start < 0 or end > n_samples_raw:
                continue
                
            # Extract Epoch
            trials.append(raw_data[:, start:end])
            
            # Map Labels: 769->0 (Left), 770->1 (Right)
            labels.append(0 if eid == EV_LEFT else 1)
            
        return np.array(trials), np.array(labels)

    def run_full_comparison(self, epochs, labels, test_size=0.25):
        """
        Extracts CSP features and trains ALL 8 models.
        Returns a dictionary of accuracies.
        """
        # 1. Split Data
        # Stratify ensures balanced classes in train/test split
        self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
            epochs, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 2. CSP Feature Extraction (From Scratch Module)
        # We assume 3 channels, so n_components=2 is optimal (1 filter per class)
        self.csp = csp_scratch.CSP_Scratch(n_components=2)
        self.csp.fit(self.X_train_raw, self.y_train)
        
        X_train_feat = self.csp.transform(self.X_train_raw)
        X_test_feat = self.csp.transform(self.X_test_raw)
        
        # 3. Standardization (Critical for SVM, LogReg, and MLP)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train_feat)
        self.X_test = scaler.transform(X_test_feat)
        
        print(f"[ML INFO] Feature Shape: {self.X_train.shape}")
        
        # 4. Define All 8 Models (Scikit-Learn Implementation)
        # Note: MLP max_iter increased to 3000 to prevent ConvergenceWarning
        models = {
            'SVM (Linear)': SVC(kernel='linear', C=1.0),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Naive Bayes': GaussianNB(),
            'Kernel SVM (RBF)': SVC(kernel='rbf', C=1.0, gamma='scale'),
            'RF (Tuned)': RandomForestClassifier(n_estimators=200, max_depth=10, criterion='entropy', random_state=42),
            'MLP (Neural Network)': MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=3000, activation='relu', solver='adam', random_state=42)
        }
        
        self.results = {}
        self.trained_models = {}

        # 5. Train and Evaluate Loop
        for name, clf in models.items():
            try:
                print(f"[ML] Training {name}...")
                clf.fit(self.X_train, self.y_train)
                
                # Predict
                pred = clf.predict(self.X_test)
                
                # Calculate Accuracy
                acc = accuracy_score(self.y_test, pred)
                
                # Store Results
                self.results[name] = acc
                self.trained_models[name] = clf
                
            except Exception as e:
                print(f"[ML ERROR] Failed to train {name}: {e}")
                self.results[name] = 0.0

        # 6. Determine Best Model
        if self.results:
            self.best_model_name = max(self.results, key=self.results.get)
            best_acc = self.results[self.best_model_name]
            print(f"\n[ML RESULT] Best Model: {self.best_model_name} ({best_acc*100:.2f}%)")
        
        return self.results

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
# Standalone Testing Block (if __name__ == "__main__")
# =========================================================
if __name__ == "__main__":
    print(">> RUNNING ML_ANALYSIS STANDALONE TEST...")
    
    # 1. Generate Synthetic Data (Simulating EEG)
    # 60 Trials, 3 Channels, 500 Samples
    n_trials = 60
    n_ch = 3
    n_samples = 500
    fs = 250
    
    trials = []
    labels = []
    
    for i in range(n_trials):
        label = np.random.randint(0, 2)
        labels.append(label)
        
        signal = np.random.randn(n_ch, n_samples)
        
        # Inject Variance Difference (Simulating ERD)
        if label == 0: 
            signal[0, :] *= 5.0 # High variance Ch 0
        else:
            signal[2, :] *= 5.0 # High variance Ch 2
            
        trials.append(signal)
        
    X_syn = np.array(trials)
    y_syn = np.array(labels)
    
    print(f"Synthetic Data Created: {X_syn.shape}")
    
    # 2. Init Pipeline
    pipeline = ML_Pipeline()
    
    # 3. Run Comparison
    print(">> STARTING MODEL COMPARISON...")
    results = pipeline.run_full_comparison(X_syn, y_syn, test_size=0.3)
    
    print("\n>> FINAL RESULTS:")
    print("-" * 40)
    print(f"{'MODEL':<30} | {'ACCURACY':<10}")
    print("-" * 40)
    for name, acc in results.items():
        print(f"{name:<30} | {acc*100:.2f}%")
    print("-" * 40)
    
    print(">> STANDALONE TEST COMPLETE.")