"""
csp_scratch.py

Purpose:
    - Implement Common Spatial Pattern (CSP) algorithm from scratch.
    - Pure NumPy implementation (Linear Algebra).
    
Dependencies:
    - numpy
"""

import numpy as np

class CSP_Scratch:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.filters_ = None
        self.patterns_ = None 

    def fit(self, X, y):
        # 1. Separate data by class
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly 2 classes.")
            
        X_class1 = X[y == classes[0]]
        X_class2 = X[y == classes[1]]
        
        # 2. Compute Average Covariance Matrix
        C1 = self._compute_avg_covariance(X_class1)
        C2 = self._compute_avg_covariance(X_class2)
        
        # 3. Composite Covariance
        C_sum = C1 + C2
        
        # 4. Eigen Decomposition
        evals, evecs = np.linalg.eigh(C_sum)
        
        # Sort descending
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        # 5. Whitening Transformation Matrix (P)
        P = np.dot(np.diag(1.0 / np.sqrt(evals + 1e-10)), evecs.T)
        
        # 6. Transform class covariances
        S1 = np.dot(P, np.dot(C1, P.T))
        
        # 7. Generalized Eigen Decomposition on S1
        b_evals, B = np.linalg.eigh(S1)
        
        # Sort descending
        idx = np.argsort(b_evals)[::-1]
        B = B[:, idx]
        
        # 8. Compute Projection Matrix (W)
        self.filters_ = np.dot(B.T, P)
        self.patterns_ = np.linalg.pinv(self.filters_)
        
        return self

    def transform(self, X):
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
            # Project and Compute Log-Variance
            Z = np.dot(W_selected, X[i])
            var_Z = np.var(Z, axis=1)
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