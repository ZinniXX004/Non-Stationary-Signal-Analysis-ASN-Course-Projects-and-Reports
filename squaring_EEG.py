"""
squaring_EEG.py

Purpose:
    - Convert EEG Amplitude to Instantaneous Power.
    - Formula: P(t) = x(t)^2
    - This is the prerequisite for calculating ERD/ERS.
    - Provide educational context regarding Signal Rectification.

Dependencies:
    - numpy
    - matplotlib (for standalone testing)
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. Context & Description Helper
# =========================================================
def get_squaring_description():
    """
    Returns a descriptive string explaining the Squaring step.
    
    Returns:
        str: Educational text about Instantaneous Power.
    """
    description = (
        "--- SIGNAL PROCESSING: SQUARING (POWER CALCULATION) ---\n\n"
        "1. OBJECTIVE:\n"
        "   To convert the raw EEG amplitude (Volts) into Instantaneous Power (Energy).\n"
        "   Formula: P(t) = x(t)^2\n\n"
        "2. WHY IS THIS NECESSARY?\n"
        "   - **Rectification:** EEG signals oscillate between positive and negative values.\n"
        "     If we simply averaged the raw signal across trials, the peaks and troughs\n"
        "     would cancel each other out (summing to near zero).\n"
        "   - **Energy Quantification:** Squaring ensures all values are non-negative,\n"
        "     representing the 'energy' of the neural population's activity.\n\n"
        "3. INTERPRETATION:\n"
        "   - High values indicate high synchronization (high amplitude oscillations).\n"
        "   - Low values indicate desynchronization (low amplitude).\n"
        "   - This step prepares the data for the 'Grand Average' across trials.\n"
    )
    return description

# =========================================================
# 2. Squaring Function
# =========================================================
def square_signal(eeg_data):
    """
    Computes the square of the EEG signal.
    
    Args:
        eeg_data (np.array): Input signal (Filtered Amplitude). 
                             Can be 1D (one channel) or 2D (channels x samples).
    
    Returns:
        power_data (np.array): Signal Power (Amplitude squared).
    """
    # Using numpy's optimized element-wise power operation
    # This is much faster than a standard python loop
    power_data = np.square(eeg_data)
    
    return power_data

# =========================================================
# Unit Test (Standalone Execution)
# =========================================================
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: squaring_EEG.py")
    
    # 1. Print Description
    print("-" * 60)
    print(get_squaring_description())
    print("-" * 60)

    # 2. Create a dummy signal: 10Hz Sine Wave
    fs = 250.0
    t = np.linspace(0, 1, int(fs))
    amplitude = np.sin(2 * np.pi * 10 * t)
    
    print("\n[TEST] Squaring signal...")
    power = square_signal(amplitude)
    
    # Validation: 
    # Squaring a sine wave sin(wt) results in (1 - cos(2wt))/2.
    # The output should be all positive.
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    # Note: Using raw string r'' to fix invalid escape sequence warning
    plt.plot(t, amplitude, label=r'Amplitude ($\mu V$)')
    plt.title("Original Signal (Amplitude)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(t, power, color='orange', label=r'Power ($\mu V^2$)')
    plt.title("Squared Signal (Instantaneous Power)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Check if all values are non-negative
    if np.all(power >= 0):
        print("[PASS] All power values are non-negative.")
    else:
        print("[FAIL] Negative values detected in power signal.")