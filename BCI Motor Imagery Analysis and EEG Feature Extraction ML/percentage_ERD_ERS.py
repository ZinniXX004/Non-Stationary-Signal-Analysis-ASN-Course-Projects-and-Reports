"""
percentage_ERD_ERS.py

Purpose:
    - Calculate the relative change in power compared to a baseline reference.
    - Formula: ERD% = ((A - R) / R) * 100
    - Differentiate between Desynchronization (Activation) and Synchronization (Idling).
    - Support Multi-Channel (C3, Cz, C4) processing for Motor Imagery analysis.
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
    # This means R will be [R_c3, R_cz, R_c4] column vector.
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
    print(">> RUNNING STANDALONE TEST: percentage_ERD_ERS.py (Multi-Channel)")
    
    # 1. Print Description
    print("-" * 60)
    print(get_erd_description())
    print("-" * 60)

    # 2. Simulate 3 Channels
    fs = 250.0
    tmin, tmax = -2.0, 6.0
    duration = tmax - tmin
    n_pts = int(duration * fs)
    time_vec = np.linspace(tmin, tmax, n_pts)
    
    # Baseline power level
    baseline_power = 10.0
    
    # Initialize array (3 Channels x N Samples)
    dummy_power = np.ones((3, n_pts)) * baseline_power
    
    # Mask for Task Period (0s to 3s) and Post-Task (3s to 5s)
    mask_task = (time_vec >= 0.0) & (time_vec < 3.0)
    mask_post = (time_vec >= 3.0) & (time_vec < 5.0)
    
    # Channel 0 (C3): Simulate ERD (Left Hemi activation for Right Hand)
    # Drop to 50% power
    dummy_power[0, mask_task] = 5.0 
    # Rebound ERS
    dummy_power[0, mask_post] = 15.0 
    
    # Channel 1 (Cz): Constant (No significant change)
    # Stays around 10.0
    
    # Channel 2 (C4): Simulate ERS (Right Hemi inhibition/idling)
    # Increase to 150% power
    dummy_power[2, mask_task] = 15.0 
    
    # Add noise
    noise = np.random.randn(3, n_pts) * 0.5
    dummy_power += noise
    
    print("\n[TEST] Calculating ERD/ERS Percentage (Ref: -2.0 to -0.5s)...")
    
    try:
        erd_result, t_axis = calculate_erd_percent(dummy_power, fs, tmin, tmax, ref_interval=(-2.0, -0.5))
        
        # 3. Numeric Validation
        print("\n--- NUMERIC RESULTS (Average during Task 0-3s) ---")
        ch_names = ['C3 (Simulated ERD)', 'Cz (No Change)', 'C4 (Simulated ERS)']
        
        task_indices = (t_axis >= 0.0) & (t_axis < 3.0)
        
        for i in range(3):
            avg_erd = np.mean(erd_result[i, task_indices])
            print(f"{ch_names[i]}: {avg_erd:.2f}%")
            
        # 4. Plotting
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        
        for i, ax in enumerate(axes):
            ax.plot(t_axis, erd_result[i, :], color='blue', linewidth=2, label='ERD/ERS %')
            ax.axhline(0, color='k', linewidth=1.5)
            ax.axvline(0, color='gray', linestyle=':', label='Cue Onset')
            
            # Highlight ERD (Negative) vs ERS (Positive) areas
            ax.fill_between(t_axis, erd_result[i, :], 0, where=(erd_result[i, :] < 0), 
                            color='blue', alpha=0.3, label='ERD (Activation)')
            ax.fill_between(t_axis, erd_result[i, :], 0, where=(erd_result[i, :] > 0), 
                            color='red', alpha=0.3, label='ERS (Idling)')
            
            ax.set_title(f"Channel: {ch_names[i]}")
            ax.set_ylabel("Change (%)")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True)
            
        axes[-1].set_xlabel("Time relative to Cue (s)")
        plt.tight_layout()
        plt.show()
        
        print("\n[TEST] Multi-Channel ERD Calculation Passed.")
        
    except Exception as e:
        print(f"\n[TEST] Failed: {e}")