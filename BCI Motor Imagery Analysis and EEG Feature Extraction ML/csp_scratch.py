"""
csp_scratch.py

Purpose:
    - Implement Common Spatial Pattern (CSP) algorithm from scratch.
    - Implement Temporal Feature Extraction (Mean, Variance, Skewness, Kurtosis).
    - Provide COMPLETE VISUALIZATION tools (Scatter Plots & Box Plots).
    - specialized for Motor Imagery setups including C3, Cz, and C4 channels.

Dependencies:
    - numpy
    - matplotlib
    - scipy (for skewness and kurtosis calculation)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# ==========================================
# 1. Context & Description Helper
# ==========================================
def get_csp_description():
    """
    Returns a descriptive string explaining the CSP algorithm and the channel setup.
    
    Returns:
        str: Educational text about Spatial Filtering and Variance.
    """
    description = (
        "--- FEATURE EXTRACTION: COMMON SPATIAL PATTERN (CSP) ---\n\n"
        "1. OBJECTIVE:\n"
        "   To design spatial filters (W) that transform the EEG signals into a new space\n"
        "   where the variance (power) is optimal for discriminating between two classes.\n"
        "   This is applied to Motor Imagery channels (e.g., C3, Cz, C4).\n\n"
        "2. THE LOGIC (Variance Discrimination):\n"
        "   - During Motor Imagery, ERD (Event-Related Desynchronization) occurs.\n"
        "   - ERD = Drop in Power = Change in Variance.\n"
        "   - CSP finds a projection matrix 'W' such that:\n"
        "     * Signal projected to Component 1 has MAX variance for Class 1 (Left Hand)\n"
        "       and MIN variance for Class 2 (Right Hand).\n"
        "     * Signal projected to Last Component has MIN variance for Class 1\n"
        "       and MAX variance for Class 2.\n\n"
        "3. MATHEMATICAL STEPS:\n"
        "   a. Calculate Covariance Matrix (R) for each trial.\n"
        "   b. Average Covariances for Class 1 (R1) and Class 2 (R2).\n"
        "   c. Solve Generalized Eigenvalue Problem: R1 * w = lambda * R2 * w.\n"
        "   d. The resulting eigenvectors (w) are the Spatial Filters.\n\n"
        "4. OUTPUT FEATURE:\n"
        "   - We do not use the raw projected signal as a feature.\n"
        "   - We calculate the Log-Variance of the projected signal: f = log(var(Z)).\n"
        "   - This small vector is fed into the Machine Learning classifier.\n"
    )
    return description


# ==========================================
# 2. CSP Implementation (Spatial Features)
# ==========================================
class CSP_Scratch:
    """
    Common Spatial Pattern (CSP) algorithm.
    Extracts spatial features that maximize variance discrimination between two classes.
    """
    def __init__(self, n_components=2):
        """
        Initialize the CSP object.
        
        Args:
            n_components (int): Number of filters to keep (usually 2 for binary classification).
        """
        self.n_components = n_components
        self.filters_ = None   # Matrix W (Spatial Filters)
        self.patterns_ = None  # Matrix A (Inverse of W, used for topographical plotting)

    def fit(self, X, y):
        """
        Computes the CSP filters based on training data.
        
        Args:
            X (numpy.ndarray): EEG Data of shape (n_trials, n_channels, n_samples).
                               Includes channels C3, Cz, C4.
            y (numpy.ndarray): Labels of shape (n_trials,).
            
        Returns:
            self: The fitted instance.
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes. Found: {len(classes)}")
            
        # Separate data by class
        X_class1 = X[y == classes[0]]
        X_class2 = X[y == classes[1]]
        
        # 1. Compute Average Covariance Matrices for each class
        C1 = self._compute_avg_covariance(X_class1)
        C2 = self._compute_avg_covariance(X_class2)
        
        # 2. Compute Composite Covariance
        C_sum = C1 + C2
        
        # 3. Eigen Decomposition for Whitening (P)
        # We solve C_sum = U * Lambda * U.T
        evals, evecs = np.linalg.eigh(C_sum)
        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(evals)[::-1]
        evals = evals[sorted_indices]
        evecs = evecs[:, sorted_indices]
        
        # 4. Construct Whitening Matrix P
        # P = diag(lambda^-0.5) * U.T
        P = np.dot(np.diag(1.0 / np.sqrt(evals + 1e-10)), evecs.T)
        
        # 5. Transform Covariance of Class 1 using P
        # S1 = P * C1 * P.T
        S1 = np.dot(P, np.dot(C1, P.T))
        
        # 6. Generalized Eigen Decomposition of S1
        # S1 = B * Lambda_b * B.T
        b_evals, B = np.linalg.eigh(S1)
        
        # Sort in descending order
        sorted_indices_b = np.argsort(b_evals)[::-1]
        B = B[:, sorted_indices_b]
        
        # 7. Compute Final Projection Matrix (W)
        # W = B.T * P
        self.filters_ = np.dot(B.T, P)
        
        # 8. Compute Spatial Patterns (A) for Topography Visualization
        # Patterns are the pseudo-inverse of Filters
        self.patterns_ = np.linalg.pinv(self.filters_)
        
        return self

    def transform(self, X):
        """
        Projects the data using filters and extracts Log-Variance features.
        
        Args:
            X (numpy.ndarray): EEG Data of shape (n_trials, n_channels, n_samples).
            
        Returns:
            numpy.ndarray: Feature matrix of shape (n_trials, n_components).
        """
        if self.filters_ is None:
            raise RuntimeError("CSP not fitted. Run fit() first.")
            
        n_trials = X.shape[0]
        n_filters = self.filters_.shape[0]
        
        # Select first m/2 and last m/2 filters
        # The first filters maximize variance for Class 1.
        # The last filters maximize variance for Class 2.
        n_pick = self.n_components // 2
        pick_indices = [i for i in range(n_pick)] + \
                       [n_filters - 1 - i for i in range(n_pick)]
                   
        W_selected = self.filters_[pick_indices, :]
        features = np.zeros((n_trials, len(pick_indices)))
        
        for i in range(n_trials):
            # 1. Project Signal: Z = W * X
            # X[i] shape is (n_channels, n_samples)
            Z = np.dot(W_selected, X[i])
            
            # 2. Calculate Variance of Projected Signal
            var_Z = np.var(Z, axis=1)
            
            # 3. Log-Transform (Normalization)
            features[i, :] = np.log(var_Z + 1e-10)
            
        return features

    def _compute_avg_covariance(self, X_epochs):
        """
        Helper to compute average covariance matrix over multiple trials.
        """
        n_trials, n_channels, n_samples = X_epochs.shape
        covariance_sum = np.zeros((n_channels, n_channels))
        
        for i in range(n_trials):
            epoch = X_epochs[i]
            
            # Center the data (mean subtraction)
            epoch = epoch - np.mean(epoch, axis=1, keepdims=True)
            
            # Compute Covariance: (X * X.T)
            covariance = np.dot(epoch, epoch.T)
            
            # Normalize by trace (sum of diagonal elements)
            covariance = covariance / np.trace(covariance)
            
            covariance_sum += covariance
            
        return covariance_sum / n_trials

    def plot_feature_scatter(self, features, labels, class_names=['Left', 'Right']):
        """
        Generates a 2D Scatter Plot of the extracted CSP features.
        Useful to see if classes are separable.
        """
        if features.shape[1] < 2:
            print("Not enough features to plot scatter.")
            return None

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        
        # Plot Class 0 (e.g., Left Hand)
        ax.scatter(features[labels==0, 0], features[labels==0, 1], 
                   color='cyan', label=class_names[0], alpha=0.7, edgecolors='white', s=60)
        
        # Plot Class 1 (e.g., Right Hand)
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


# ==========================================
# 3. Temporal Feature Extraction
# ==========================================
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
        
        # We extract 5 metrics per channel
        n_metrics = len(self.feature_names)
        features = np.zeros((n_trials, n_channels * n_metrics))
        
        for i in range(n_trials):
            trial_features = []
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
                
                trial_features.extend([mean_val, var_val, std_val, skew_val, kurt_val])
                
            features[i, :] = np.array(trial_features)
            
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
            # Structure: 
            # [Ch1_Metrics(0-4), Ch2_Metrics(5-9), Ch3_Metrics(10-14)...]
            col_idx = (channel_idx * n_metrics) + m_idx
            
            data_c0 = features[labels==0, col_idx]
            data_c1 = features[labels==1, col_idx]
            
            # Create Boxplot
            box = ax.boxplot([data_c0, data_c1], 
                             tick_labels=class_names, 
                             patch_artist=True, 
                             medianprops=dict(color="white"),
                             whiskerprops=dict(color="white"),
                             capprops=dict(color="white"))
            
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


# ==========================================
# 4. Standalone Test (Simulation)
# ==========================================
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: csp_scratch.py")
    print(">> Configuration: 3 Channels (C3, Cz, C4)")
    
    # 1. Print Description
    print("-" * 60)
    print(get_csp_description())
    print("-" * 60)
    
    # 2. Generate Synthetic Data
    # Setup: 100 trials, 3 Channels (C3, Cz, C4), 500 samples
    n_trials = 100
    n_channels = 3  # Index 0=C3, Index 1=Cz, Index 2=C4
    n_samples = 500
    
    # Random noise base
    X_synthetic = np.random.randn(n_trials, n_channels, n_samples)
    y_synthetic = np.array([0]*50 + [1]*50) # 50 Left, 50 Right
    
    # -----------------------------------------------------------------
    # SIMULATION LOGIC:
    # Class 0 (Left Hand Imagery):
    #   - C4 (Right Hemisphere) should Desynchronize (Power Drop).
    #   - C3 (Left Hemisphere) might stay neutral or increase slightly.
    #   - Here, we simulate simple Variance differences for classification.
    #
    # Class 1 (Right Hand Imagery):
    #   - C3 (Left Hemisphere) should Desynchronize.
    # -----------------------------------------------------------------
    
    # Inject Artificial Variance Differences
    # Make C3 (Index 0) highly active for Class 0
    X_synthetic[0:50, 0, :] *= 2.5 
    
    # Make Cz (Index 1) moderately active for both (Central Reference)
    # Usually Cz is less discriminative, but we give it some noise variation
    X_synthetic[:, 1, :] *= 1.2
    
    # Make C4 (Index 2) highly active for Class 1
    X_synthetic[50:, 2, :] *= 2.5 
    
    
    # 3. Test CSP (Spatial Filtering)
    print("\n[TEST] CSP Training & Visualization...")
    csp = CSP_Scratch(n_components=2)
    csp.fit(X_synthetic, y_synthetic)
    csp_features = csp.transform(X_synthetic)
    
    print(f"CSP Features Shape: {csp_features.shape}")
    print("Top row features (log-variance):")
    print(csp_features[0:3, :])
    
    # Visualize CSP Scatter
    fig_csp = csp.plot_feature_scatter(csp_features, y_synthetic)
    if fig_csp:
        fig_csp.suptitle("TEST CSP PLOT (Separability of Classes)", color='white', y=0.98)
    
    
    # 4. Test Temporal Extraction (Statistical)
    print("\n[TEST] Temporal Extraction & Visualization...")
    temp_extractor = TemporalFeatureExtractor()
    temp_features = temp_extractor.transform(X_synthetic)
    
    print(f"Temporal Features Shape: {temp_features.shape}")
    expected_dim = n_trials * (n_channels * 5)
    print(f"(Expected columns: {n_channels} channels * 5 metrics = 15)")
    
    # -----------------------------------------------------
    # Visualization: Inspect Specific Channels
    # -----------------------------------------------------
    channel_map = {0: 'C3 (Left)', 1: 'Cz (Center)', 2: 'C4 (Right)'}
    
    # Plot for C3
    fig_c3 = temp_extractor.plot_feature_boxplot_all_metrics(
        temp_features, y_synthetic, channel_idx=0, ch_name=channel_map[0]
    )
    
    # Plot for Cz (NEWLY ADDED REQUEST)
    fig_cz = temp_extractor.plot_feature_boxplot_all_metrics(
        temp_features, y_synthetic, channel_idx=1, ch_name=channel_map[1]
    )
    
    # Plot for C4
    fig_c4 = temp_extractor.plot_feature_boxplot_all_metrics(
        temp_features, y_synthetic, channel_idx=2, ch_name=channel_map[2]
    )
    
    plt.show()
    print(">> TEST COMPLETE.")