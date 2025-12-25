"""
squaring_EEG.py (REVISED v3.0)

Purpose:
    - Convert EEG Amplitude to Instantaneous Power (Energy).
    - Formula: P(t) = x(t)^2.
    - Support Multi-Channel Processing (C3, Cz, C4).
    - This is a critical step before Averaging to prevent positive/negative cancelation.

Dependencies:
    - numpy
    - matplotlib (for standalone testing)
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Context & Description Helper
def get_squaring_description():
    """
    Returns a descriptive string explaining the Squaring step.
    
    Returns:
        str: Educational text about Instantaneous Power.
    """
    description = (
        "--- SIGNAL PROCESSING: SQUARING (POWER CALCULATION) ---\n\n"
        "1. OBJECTIVE:\n"
        "   To convert the raw EEG amplitude (Volts) into Instantaneous Power (Energy)\n"
        "   for all three channels (C3, Cz, C4).\n"
        "   Formula: P(t) = x(t)^2\n\n"
        "2. WHY IS THIS NECESSARY?\n"
        "   - **Rectification:** EEG signals oscillate between positive and negative values.\n"
        "     If we simply averaged the raw signal across trials, the peaks and troughs\n"
        "     would cancel each other out (summing to near zero).\n"
        "   - **Energy Quantification:** Squaring ensures all values are non-negative,\n"
        "     representing the 'energy' of the neural population's activity.\n\n"
        "3. RELEVANCE TO BCI:\n"
        "   - This allows us to quantify the magnitude of Mu/Beta rhythms.\n"
        "   - High Power = Synchronization (Idling).\n"
        "   - Low Power = Desynchronization (Active Calculation/Movement).\n"
    )
    return description

# 2. Squaring Function
def square_signal(eeg_data):
    """
    Computes the square of the EEG signal (Instantaneous Power).
    
    Args:
        eeg_data (np.array): Input signal (Filtered Amplitude). 
                             Shape: (n_channels, n_samples).
                             Typically contains C3, Cz, and C4.
    
    Returns:
        power_data (np.array): Signal Power (Amplitude squared).
                               Shape: Same as input.
    """
    # Using numpy's optimized element-wise power operation.
    # This works automatically for 1D arrays or 2D matrices (3 channels).
    power_data = np.square(eeg_data)
    
    return power_data

# Unit Test (Standalone Execution)
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: squaring_EEG.py (Multi-Channel)")
    
    # 1. Print Description
    print("-" * 60)
    print(get_squaring_description())
    print("-" * 60)

    # 2. Create Dummy Data for 3 Channels
    fs = 250.0
    t = np.linspace(0, 1, int(fs))
    
    # C3: High Amplitude Sine (Simulating Strong Rhythm)
    sig_c3 = np.sin(2 * np.pi * 10 * t) * 10.0
    
    # Cz: Medium Amplitude
    sig_cz = np.sin(2 * np.pi * 10 * t) * 5.0
    
    # C4: Low Amplitude (Simulating ERD/Attenuated Rhythm)
    sig_c4 = np.sin(2 * np.pi * 10 * t) * 1.0
    
    # Stack into (3, samples)
    data_3ch = np.vstack([sig_c3, sig_cz, sig_c4])
    
    print(f"\n[TEST] Squaring signal array of shape {data_3ch.shape}...")
    power_3ch = square_signal(data_3ch)
    
    # 3. Validation and Plotting
    # Check if all values are non-negative
    if np.all(power_3ch >= 0):
        print("[PASS] All power values are non-negative (Rectification successful).")
    else:
        print("[FAIL] Negative values detected in power signal.")

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    channel_names = ['C3 (High Power)', 'Cz (Medium Power)', 'C4 (Low Power)']
    
    for i, ax in enumerate(axes):
        # Plot Original Amplitude (Dotted)
        ax.plot(t, data_3ch[i, :], label=r'Amplitude ($\mu V$)', 
                color='gray', linestyle='--', alpha=0.7)
        
        # Plot Squared Power (Solid)
        ax.plot(t, power_3ch[i, :], label=r'Power ($\mu V^2$)', 
                color='orange', linewidth=2)
        
        ax.set_title(f"Channel {channel_names[i]}")
        ax.legend(loc='upper right')
        ax.grid(True)
    
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    
    print("\n[TEST] Squaring Module Verification Passed.")