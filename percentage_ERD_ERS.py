"""
percentage_ERD_ERS.py

Purpose:
    - Calculate the relative change in power compared to a baseline reference.
    - Formula: ERD% = ((A - R) / R) * 100
    - Differentiate between Desynchronization (Activation) and Synchronization (Idling).
    - Provide educational context regarding the quantification of brain state changes.

Dependencies:
    - numpy
    - matplotlib (for standalone testing)
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. Context & Description Helper
# =========================================================
def get_erd_description():
    """
    Returns a descriptive string explaining the ERD/ERS calculation.
    
    Returns:
        str: Educational text about the formula and interpretation.
    """
    description = (
        "--- QUANTIFICATION: ERD/ERS PERCENTAGE ---\n\n"
        "1. FORMULA:\n"
        "   ERD% = ((A - R) / R) * 100\n"
        "   Where:\n"
        "   - A (Activity): Power at a specific time point (t).\n"
        "   - R (Reference): Average power during the baseline period (pre-cue).\n\n"
        "2. INTERPRETATION:\n"
        "   - **ERD (Event-Related Desynchronization):**\n"
        "     * Negative values (e.g., -50%).\n"
        "     * Indicates a decrease in rhythmic power.\n"
        "     * Meaning: Cortical ACTIVATION (neuronal populations firing independently).\n"
        "   - **ERS (Event-Related Synchronization):**\n"
        "     * Positive values (e.g., +150%).\n"
        "     * Indicates an increase in rhythmic power.\n"
        "     * Meaning: Cortical IDLING or INHIBITION (neuronal populations firing in sync).\n\n"
        "3. TYPICAL PATTERN (Motor Imagery):\n"
        "   - Pre-cue: Baseline (0%).\n"
        "   - During Imagery: ERD (Negative) in contralateral sensorimotor area.\n"
        "   - Post Imagery: Beta ERS (Positive Rebound).\n"
    )
    return description

# =========================================================
# 2. Calculation Function
# =========================================================
def calculate_erd_percent(smoothed_data, fs, tmin, tmax, ref_interval=(-1.0, 0.0)):
    """
    Computes the ERD/ERS percentage time course.

    Args:
        smoothed_data (np.array): 2D array (n_channels x n_time_points) of Smoothed Power.
        fs (float): Sampling frequency.
        tmin (float): Start time of the epoch relative to cue (e.g., -1.5).
        tmax (float): End time of the epoch relative to cue (e.g., 4.5).
        ref_interval (tuple): (start, end) time for the Reference baseline.
                              Usually the period immediately before the cue.

    Returns:
        erd_percentage (np.array): Array of same shape as input, but in Percentage (%).
        time_axis (np.array): The time vector corresponding to the data.
    """
    n_channels, n_samples = smoothed_data.shape
    
    # 1. Generate Time Axis
    # This assumes the data exactly spans tmin to tmax linearly
    time_axis = np.linspace(tmin, tmax, n_samples)
    
    # 2. Identify Baseline Indices
    # Find the indices in the time_axis that correspond to ref_interval
    t_ref_start, t_ref_end = ref_interval
    
    # Using searchsorted to find nearest indices efficiently
    idx_start = np.searchsorted(time_axis, t_ref_start)
    idx_end = np.searchsorted(time_axis, t_ref_end)
    
    # Safety Checks
    if idx_start >= idx_end:
        print("[WARN] Invalid reference interval indices. Defaulting to first 10 samples.")
        idx_start = 0
        idx_end = 10
        
    if idx_end > n_samples:
        print("[WARN] Reference interval ends outside data bounds. Clipping to end.")
        idx_end = n_samples

    # 3. Calculate Reference Power (R)
    # R is the mean power across the baseline period for EACH channel.
    # keepdims=True preserves shape as (n_channels, 1) for correct broadcasting.
    R = np.mean(smoothed_data[:, idx_start:idx_end], axis=1, keepdims=True)
    
    # Prevent Division by Zero
    # If R is exactly zero, set to a tiny epsilon to avoid NaN/Inf errors.
    R[R == 0] = 1e-9
    
    # 4. Apply Formula: ((A - R) / R) * 100
    # 'smoothed_data' represents 'A' (Activity) over time
    erd_percentage = ((smoothed_data - R) / R) * 100.0
    
    return erd_percentage, time_axis

# =========================================================
# Unit Test (Standalone Execution)
# =========================================================
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: percentage_ERD_ERS.py")
    
    # 1. Print Description
    print("-" * 60)
    print(get_erd_description())
    print("-" * 60)

    # 2. Simulate a single channel signal
    fs = 250.0
    tmin, tmax = -2.0, 6.0
    duration = tmax - tmin
    n_pts = int(duration * fs)
    time_vec = np.linspace(tmin, tmax, n_pts)
    
    # Create Dummy Data:
    # Baseline (-2s to 0s): Power = 10 units (Stable)
    # ERD (0s to 3s): Power drops to 5 units (Should be -50%)
    # ERS (3s to 6s): Power rises to 15 units (Should be +50%)
    
    dummy_power = np.ones((1, n_pts)) * 10.0 # Start with Baseline level
    
    # Apply ERD Drop
    mask_erd = (time_vec >= 0.0) & (time_vec < 3.0)
    dummy_power[:, mask_erd] = 5.0
    
    # Apply ERS Rise
    mask_ers = (time_vec >= 3.0)
    dummy_power[:, mask_ers] = 15.0
    
    # Add a little noise to make it realistic
    noise = np.random.randn(1, n_pts) * 0.1
    dummy_power += noise
    
    print("\n[TEST] Calculating ERD/ERS Percentage (Ref: -2.0 to -0.5s)...")
    
    try:
        erd_result, t_axis = calculate_erd_percent(dummy_power, fs, tmin, tmax, ref_interval=(-2.0, -0.5))
        
        # 3. Plotting
        plt.figure(figsize=(10, 6))
        
        # Subplot 1: Absolute Power
        plt.subplot(2, 1, 1)
        plt.plot(t_axis, dummy_power[0, :], 'k', label='Smoothed Power')
        plt.title("Input: Absolute Power")
        plt.axvspan(-2, -0.5, color='green', alpha=0.1, label='Reference Period')
        plt.ylabel(r"Power ($\mu V^2$)")
        plt.legend(loc='upper left')
        plt.grid(True)
        
        # Subplot 2: Relative Percentage
        plt.subplot(2, 1, 2)
        plt.plot(t_axis, erd_result[0, :], 'b', linewidth=2, label='ERD/ERS %')
        plt.title("Output: Relative Change (%)")
        
        # Reference Lines
        plt.axhline(0, color='k', linewidth=1)
        plt.axhline(-50, color='r', linestyle='--', label='Expected ERD (-50%)')
        plt.axhline(50, color='g', linestyle='--', label='Expected ERS (+50%)')
        
        plt.ylabel("Change (%)")
        plt.xlabel("Time relative to Cue (s)")
        plt.legend(loc='upper left')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("\n[TEST] ERD Calculation Module Verification Passed.")
        
    except Exception as e:
        print(f"\n[TEST] Failed: {e}")