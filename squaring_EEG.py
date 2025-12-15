"""
squaring_EEG.py

Purpose:
    - Convert EEG Amplitude to Instantaneous Power.
    - Formula: P(t) = x(t)^2
    - This is the prerequisite for calculating ERD/ERS.

Dependencies:
    - numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

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
    power_data = np.square(eeg_data)
    
    return power_data

# =========================================================
# Unit Test
# =========================================================
if __name__ == "__main__":
    # Create a dummy signal: 10Hz Sine Wave
    fs = 250.0
    t = np.linspace(0, 1, int(fs))
    amplitude = np.sin(2 * np.pi * 10 * t)
    
    print("Squaring signal...")
    power = square_signal(amplitude)
    
    # Validation: 
    # Squaring a sine wave sin(wt) results in (1 - cos(2wt))/2.
    # The output should be all positive.
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, amplitude, label='Amplitude ($\mu V$)')
    plt.title("Original Signal (Amplitude)")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(t, power, color='orange', label='Power ($\mu V^2$)')
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