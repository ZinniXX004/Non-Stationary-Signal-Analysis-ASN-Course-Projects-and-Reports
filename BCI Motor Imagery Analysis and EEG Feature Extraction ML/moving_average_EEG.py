"""
moving_average_EEG.py (REVISED v3.0)

Purpose:
    - Apply a Moving Average (Sliding Window) filter to smooth the Power signal.
    - Extract the 'Envelope' of the ERD/ERS response.
    - Support Multi-Channel Processing (C3, Cz, C4).
    - Offload computationally intensive loop to C++ backend.

Dependencies:
    - ctypes
    - numpy
    - matplotlib (for standalone testing)
    - os
"""

import ctypes
import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Load C++ Library
dll_name = "eeg_processing.dll"
dll_path = os.path.abspath(dll_name)

try:
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"DLL not found at: {dll_path}")
    
    lib = ctypes.CDLL(dll_path)

    # Function Signature:
    # void moving_average(double* input, int length, int window_size, double* output)
    lib.moving_average.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # input array
        ctypes.c_int,     # length
        ctypes.c_int,     # window_size (samples)
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')  # output array
    ]
    lib.moving_average.restype = None

    print(f"[INFO] C++ Library loaded successfully: {dll_name}")

except Exception as e:
    print(f"[ERROR] Failed to load DLL: {e}")
    lib = None

# 2. Context & Description Helper
def get_smoothing_description():
    """
    Returns a descriptive string explaining the Moving Average step.
    
    Returns:
        str: Educational text about Envelope Extraction.
    """
    description = (
        "--- POST-PROCESSING: MOVING AVERAGE (SMOOTHING) ---\n\n"
        "1. OBJECTIVE:\n"
        "   To extract the smooth 'envelope' of the power signal for C3, Cz, and C4.\n\n"
        "2. WHY IS THIS NECESSARY?\n"
        "   - The Squared Signal (Instantaneous Power) is extremely jagged and noisy.\n"
        "   - To visualize the slow cortical potential changes (ERD/ERS), we must\n"
        "     average out the rapid fluctuations.\n\n"
        "3. METHOD (Sliding Window):\n"
        "   - We calculate the arithmetic mean of data points within a sliding window.\n"
        "   - Standard Window: 0.5 to 1.0 seconds.\n"
        "   - This acts as a Low-Pass Filter in the time domain.\n"
    )
    return description

# 3. Smoothing Function Wrapper
def apply_moving_average(power_data, fs, window_sec=0.5):
    """
    Smooths the power data using the C++ backend.
    
    Args:
        power_data (np.array): Input array (Channels x Samples).
        fs (float): Sampling frequency.
        window_sec (float): Length of the smoothing window in seconds.
    
    Returns:
        smoothed_data (np.array): Smoothed signal with same shape as input.
    """
    if lib is None:
        raise RuntimeError("C++ Library not loaded. Cannot perform Smoothing.")

    # Handle 1D input by reshaping to 2D (1, N) temporarily
    is_1d = False
    if power_data.ndim == 1:
        power_data = power_data.reshape(1, -1)
        is_1d = True

    n_channels, n_samples = power_data.shape
    window_samples = int(window_sec * fs)
    
    # Initialize output array
    smoothed_data = np.zeros_like(power_data, dtype=np.float64)
    
    # Process each channel individually
    for i in range(n_channels):
        # 1. Prepare C-compatible arrays (Contiguous memory)
        channel_in = np.ascontiguousarray(power_data[i], dtype=np.float64)
        channel_out = np.zeros(n_samples, dtype=np.float64)
        
        # 2. Call C++ Function
        lib.moving_average(channel_in, n_samples, window_samples, channel_out)
        
        # 3. Store result
        smoothed_data[i, :] = channel_out

    # Restore original shape if input was 1D
    if is_1d:
        return smoothed_data.flatten()

    return smoothed_data

# Unit Test (Standalone Execution)
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: moving_average_EEG.py (Multi-Channel)")
    
    # 1. Print Description
    print("-" * 60)
    print(get_smoothing_description())
    print("-" * 60)

    # 2. Create Noisy Data for 3 Channels
    fs = 250.0
    duration = 4.0
    t = np.linspace(0, duration, int(duration * fs))
    
    # C3: Step Function (Simulating ERS Rebound)
    clean_c3 = np.zeros_like(t)
    clean_c3[t >= 2.0] = 10.0
    
    # Cz: Sine Wave Envelope
    clean_cz = np.sin(2 * np.pi * 0.5 * t) * 5.0 + 5.0
    
    # C4: Impulse (Simulating Artifact)
    clean_c4 = np.zeros_like(t)
    clean_c4[(t > 1.9) & (t < 2.1)] = 20.0
    
    # Stack and Add Noise
    clean_3ch = np.vstack([clean_c3, clean_cz, clean_c4])
    noise = np.random.randn(3, len(t)) * 2.0
    noisy_3ch = clean_3ch + noise
    
    print(f"\n[TEST] Applying Moving Average to 3 Channels (Window = 0.5s)...")
    
    try:
        smoothed_3ch = apply_moving_average(noisy_3ch, fs, window_sec=0.5)
        
        # 3. Plot Comparison
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        ch_names = ['C3 (Step)', 'Cz (Sine)', 'C4 (Impulse)']
        
        for i, ax in enumerate(axes):
            ax.plot(t, noisy_3ch[i, :], color='lightgray', label='Noisy Input')
            ax.plot(t, clean_3ch[i, :], 'k--', label='True Envelope')
            ax.plot(t, smoothed_3ch[i, :], 'r', linewidth=2, label='Smoothed Output')
            
            ax.set_title(f"Smoothing Result: {ch_names[i]}")
            ax.set_ylabel("Power")
            ax.legend(loc='upper right')
            ax.grid(True)
            
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
        
        print("\n[TEST] Multi-Channel Smoothing Verification Passed.")
        
    except Exception as e:
        print(f"\n[TEST] Failed: {e}")