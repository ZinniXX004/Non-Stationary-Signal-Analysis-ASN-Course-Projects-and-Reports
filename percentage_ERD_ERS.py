"""
percentage_ERD_ERS.py

Purpose:
    - Calculate the relative change in power compared to a baseline reference.
    - Formula: ERD% = ((A - R) / R) * 100
    
Dependencies:
    - numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_erd_percent(smoothed_data, fs, tmin, tmax, ref_interval=(-1.0, 0.0)):
    """
    Computes the ERD/ERS percentage time course.

    Args:
        smoothed_data (np.array): 2D array (n_channels x n_time_points).
        fs (float): Sampling frequency.
        tmin (float): Start time of epoch.
        tmax (float): End time of epoch.
        ref_interval (tuple): (start, end) time for baseline.

    Returns:
        erd_percentage (np.array): Result in Percentage.
        time_axis (np.array): Time vector.
    """
    n_channels, n_samples = smoothed_data.shape
    
    # 1. Generate Time Axis
    time_axis = np.linspace(tmin, tmax, n_samples)
    
    # 2. Identify Baseline Indices
    t_ref_start, t_ref_end = ref_interval
    
    # Find indices
    idx_start = np.searchsorted(time_axis, t_ref_start)
    idx_end = np.searchsorted(time_axis, t_ref_end)
    
    # Safety Check
    if idx_start >= idx_end:
        print("[WARN] Invalid reference interval. Defaulting to first 10 samples.")
        idx_start = 0
        idx_end = 10
        
    if idx_end > n_samples:
        idx_end = n_samples

    # 3. Calculate Reference Power (R)
    # Mean across the time dimension (axis 1) for the baseline period
    R = np.mean(smoothed_data[:, idx_start:idx_end], axis=1, keepdims=True)
    
    # Prevent Division by Zero
    R[R == 0] = 1e-9
    
    # 4. Apply Formula
    erd_percentage = ((smoothed_data - R) / R) * 100.0
    
    return erd_percentage, time_axis

# =========================================================
# Unit Test
# =========================================================
if __name__ == "__main__":
    # Create dummy data
    data = np.random.rand(3, 1000) * 10
    fs = 250.0
    
    print("Testing calculate_erd_percent...")
    res, t = calculate_erd_percent(data, fs, -1, 3, (-1, 0))
    
    if res.shape == data.shape:
        print("[PASS] Function exists and returns correct shape.")
    else:
        print("[FAIL] Shape mismatch.")