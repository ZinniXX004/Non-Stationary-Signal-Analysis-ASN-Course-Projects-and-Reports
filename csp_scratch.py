"""
csp_scratch.py

Purpose:
    - Implement Common Spatial Pattern (CSP) algorithm from scratch.
    - Implement Temporal Feature Extraction (Mean, Var, Skew, Kurtosis).
    - Provide COMPLETE VISUALIZATION tools (Scatter Plots & Box Plots).
    - Allow users to visually inspect feature separability before Machine Learning.

Dependencies:
    - numpy
    - matplotlib
    - scipy (for skewness and kurtosis calculation)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# =========================================================
# 1. Context & Description Helper
# =========================================================
def get_csp_description():
    """
    Returns a descriptive string explaining the CSP algorithm.
    
    Returns:
        str: Educational text about Spatial Filtering and Variance.
    """
    description = (
        "--- FEATURE EXTRACTION: COMMON SPATIAL PATTERN (CSP) ---\n\n"
        "1. OBJECTIVE:\n"
        "   To design spatial filters (W) that transform the EEG signals into a new space\n"
        "   where the variance (power) is optimal for discriminating between two classes.\n\n"
        "2. THE LOGIC (Variance Discrimination):\n"
        "   - During Motor Imagery, ERD (Desynchronization) occurs.\n"
        "   - ERD = Drop in Power = Change in Variance.\n"
        "   - CSP finds a projection matrix 'W' such that:\n"
        "     * Signal projected to Component 1 has MAX variance for Class 1 (Left Hand)\n"
        "       and MIN variance for Class 2 (Right Hand).\n"
        "     * Signal projected to Component Last has MIN variance for Class 1\n"
        "       and MAX variance for Class 2.\n\n"
        "3. MATHEMATICAL STEPS:\n"
        "   a. Calculate Covariance Matrix (R) for each trial.\n"
        "   b. Average Covariances for Class 1 (R1) and Class 2 (R2).\n"
        "   c. Solve Generalized Eigenvalue Problem: R1 * w = lambda * R2 * w.\n"
        "   d. The resulting eigenvectors (w) are the Spatial Filters.\n\n"
        "4. OUTPUT FEATURE:\n"
        "   - We do not use the raw projected signal as a feature.\n"
        "   - We calculate the Log-Variance of the projected signal: f = log(var(Z)).\n"
        "   - This small vector (e.g., 2 values) is fed into the Machine Learning classifier.\n"
    )
    return description

# =========================================================
# 2. CSP Implementation (Spatial Features)
# =========================================================
class CSP_Scratch:
    """
    Common Spatial Pattern (CSP) algorithm.
    Extracts spatial features that maximize variance discrimination.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.filters_ = None   # Matrix W
        self.patterns_ = None  # Matrix A (Inverse W)

    def fit(self, X, y):
        """
        Computes the CSP filters based on training data.
        X: (n_trials, n_channels, n_samples)
        y: (n_trials,)
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes. Found: {len(classes)}")
            
        X_class1 = X[y == classes[0]]
        X_class2 = X[y == classes[1]]
        
        # 1. Compute Average Covariance Matrices
        C1 = self._compute_avg_covariance(X_class1)
        C2 = self._compute_avg_covariance(X_class2)
        
        # 2. Composite Covariance
        C_sum = C1 + C2
        
        # 3. Eigen Decomposition for Whitening
        evals, evecs = np.linalg.eigh(C_sum)
        
        # Sort descending
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]
        
        # 4. Whitening Matrix P
        P = np.dot(np.diag(1.0 / np.sqrt(evals + 1e-10)), evecs.T)
        
        # 5. Transform Covariances
        S1 = np.dot(P, np.dot(C1, P.T))
        
        # 6. Generalized Eigen Decomposition
        b_evals, B = np.linalg.eigh(S1)
        idx = np.argsort(b_evals)[::-1]
        B = B[:, idx]
        
        # 7. Projection Matrix (W)
        self.filters_ = np.dot(B.T, P)
        
        # 8. Spatial Patterns (A) for Topography
        self.patterns_ = np.linalg.pinv(self.filters_)
        
        return self

    def transform(self, X):
        """
        Extracts Log-Variance features.
        Returns: (n_trials, n_components)
        """
        if self.filters_ is None:
            raise RuntimeError("CSP not fitted. Run fit() first.")
            
        n_trials = X.shape[0]
        n_filters = self.filters_.shape[0]
        
        # Select first m and last m filters
        pick_idx = [i for i in range(self.n_components // 2)] + \
                   [n_filters - 1 - i for i in range(self.n_components // 2)]
                   
        W_selected = self.filters_[pick_idx, :]
        features = np.zeros((n_trials, len(pick_idx)))
        
        for i in range(n_trials):
            # Project: Z = W * X
            Z = np.dot(W_selected, X[i])
            # Variance
            var_Z = np.var(Z, axis=1)
            # Log-Transform
            features[i, :] = np.log(var_Z + 1e-10)
            
        return features

    def _compute_avg_covariance(self, X_epochs):
        n_trials, n_channels, n_samples = X_epochs.shape
        cov_sum = np.zeros((n_channels, n_channels))
        for i in range(n_trials):
            epoch = X_epochs[i]
            epoch = epoch - np.mean(epoch, axis=1, keepdims=True)
            cov = np.dot(epoch, epoch.T)
            cov = cov / np.trace(cov)
            cov_sum += cov
        return cov_sum / n_trials

    def plot_feature_scatter(self, features, labels, class_names=['Left', 'Right']):
        """
        Generates a 2D Scatter Plot of the extracted CSP features.
        """
        if features.shape[1] < 2:
            return None

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        
        # Plot Class 0
        ax.scatter(features[labels==0, 0], features[labels==0, 1], 
                   color='cyan', label=class_names[0], alpha=0.7, edgecolors='white', s=60)
        
        # Plot Class 1
        ax.scatter(features[labels==1, 0], features[labels==1, 1], 
                   color='magenta', label=class_names[1], alpha=0.7, edgecolors='white', s=60)
        
        ax.set_title("CSP FEATURE SPACE (SPATIAL)", color='white', fontweight='bold')
        ax.set_xlabel("Log-Variance (Component 1)", color='white')
        ax.set_ylabel("Log-Variance (Component 2)", color='white')
        
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('white')
        ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white')
        ax.grid(True, color='#333', linestyle='--')
        
        fig.tight_layout()
        return fig

# =========================================================
# 2. Temporal Feature Extraction (Statistical Features)
# =========================================================
class TemporalFeatureExtractor:
    """
    Extracts statistical time-domain features from raw EEG signals.
    Metrics: Mean, Variance, StdDev, Skewness, Kurtosis.
    """
    def __init__(self):
        self.feature_names = ['Mean', 'Variance', 'StdDev', 'Skewness', 'Kurtosis']

    def transform(self, X):
        """
        Extracts temporal features from X.
        Args:
            X: (n_trials, n_channels, n_samples)
        Returns:
            features: (n_trials, n_channels * 5) -> Flattened feature vector
        """
        n_trials, n_channels, n_samples = X.shape
        
        # Initialize output matrix
        # We extract 5 metrics per channel
        n_metrics = len(self.feature_names)
        features = np.zeros((n_trials, n_channels * n_metrics))
        
        for i in range(n_trials):
            trial_feats = []
            for c in range(n_channels):
                signal = X[i, c, :]
                
                # 1. Mean
                mean_val = np.mean(signal)
                
                # 2. Variance
                var_val = np.var(signal)
                
                # 3. Standard Deviation
                std_val = np.std(signal)
                
                # 4. Skewness
                skew_val = skew(signal)
                
                # 5. Kurtosis
                kurt_val = kurtosis(signal)
                
                trial_feats.extend([mean_val, var_val, std_val, skew_val, kurt_val])
                
            features[i, :] = np.array(trial_feats)
            
        return features

    def plot_feature_boxplot_all_metrics(self, features, labels, channel_idx=0, 
                                         class_names=['Left', 'Right'], ch_name="C3"):
        """
        Generates 5 Box Plots (one for each metric) for a specific channel.
        Visualizes which statistical property best separates the classes.
        """
        n_metrics = len(self.feature_names)
        
        # Create subplots (1 row, 5 columns)
        fig, axes = plt.subplots(1, n_metrics, figsize=(15, 4))
        fig.patch.set_facecolor('#0d0d0d')
        
        colors = ['cyan', 'magenta']
        
        for m_idx, ax in enumerate(axes):
            ax.set_facecolor('#0d0d0d')
            metric_name = self.feature_names[m_idx]
            
            # Calculate column index in flattened feature matrix
            # Structure: [Ch1_Metrics..., Ch2_Metrics..., Ch3_Metrics...]
            col_idx = (channel_idx * n_metrics) + m_idx
            
            data_c0 = features[labels==0, col_idx]
            data_c1 = features[labels==1, col_idx]
            
            # FIX: Used 'tick_labels' instead of 'labels' to support Matplotlib 3.9+
            '''box = ax.boxplot([data_c0, data_c1], tick_labels=class_names, patch_artist=True,
                             medianprops=dict(color="white"))'''
            box = ax.boxplot([data_c0, data_c1], 
                 tick_labels=class_names, 
                 patch_artist=True, 
                 medianprops=dict(color="white"),
                 whiskerprops=dict(color="white"),  # Makes vertical lines white
                 capprops=dict(color="white"))      # Makes end caps white
            
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                
            ax.set_title(metric_name, color='white', fontsize=10, fontweight='bold')
            
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('white')
            ax.grid(True, color='#333', axis='y', linestyle='--')
            
        fig.suptitle(f"TEMPORAL FEATURES DISTRIBUTION ({ch_name})", color='white', y=1.05)
        fig.tight_layout()
        return fig

# =========================================================
# Standalone Test
# =========================================================
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: csp_scratch.py")
    
    # 1. Print Description
    print("-" * 60)
    print(get_csp_description())
    print("-" * 60)
    
    # 2. Generate Synthetic Data (3 Channels, 100 trials)
    n_trials = 100
    n_ch = 3
    n_samp = 500
    X = np.random.randn(n_trials, n_ch, n_samp)
    y = np.array([0]*50 + [1]*50)
    
    # Inject differences
    # Class 0: High Variance Ch0 (Left Hemi)
    X[0:50, 0, :] *= 3.0
    # Class 1: High Variance Ch2 (Right Hemi)
    X[50:, 2, :] *= 3.0
    
    # 3. Test CSP
    print("\n[TEST] CSP Training & Visualization...")
    csp = CSP_Scratch(n_components=2)
    csp.fit(X, y)
    csp_feats = csp.transform(X)
    print(f"CSP Features Shape: {csp_feats.shape}")
    
    fig1 = csp.plot_feature_scatter(csp_feats, y)
    if fig1:
        fig1.suptitle("TEST CSP PLOT", color='white', y=0.98)
    
    # 4. Test Temporal Extraction
    print("\n[TEST] Temporal Extraction & Visualization...")
    temp_extractor = TemporalFeatureExtractor()
    temp_feats = temp_extractor.transform(X)
    print(f"Temporal Features Shape: {temp_feats.shape}")
    print(f"(Expected: {n_trials} x {n_ch * 5} = {n_trials} x 15)")
    
    # Generate Multi-Metric Box Plot for Channel 0 (C3)
    fig2 = temp_extractor.plot_feature_boxplot_all_metrics(temp_feats, y, channel_idx=0, 
                                                           ch_name="C3 (Simulated)")
    
    plt.show()
    print(">> TEST COMPLETE.")