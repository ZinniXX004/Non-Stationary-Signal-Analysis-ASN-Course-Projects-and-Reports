"""
moving_average_EEG.py

Purpose:
    - Apply a Moving Average (Sliding Window) filter to smooth the Power signal.
    - Offload the loop iteration to 'eeg_processing.dll' (C++).
    - Window size is defined in seconds (converted to samples based on fs).

Dependencies:
    - ctypes, numpy, matplotlib
"""

import ctypes
import numpy as np
import os
import matplotlib.pyplot as plt

# =========================================================
# 1. Load C++ Library
# =========================================================
dll_name = "eeg_processing.dll"
dll_path = os.path.abspath(dll_name)

try:
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"DLL not found at: {dll_path}")
    
    lib = ctypes.CDLL(dll_path)

    # void moving_average(double* input, int length, int window_size, double* output)
    lib.moving_average.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # input
        ctypes.c_int,     # length
        ctypes.c_int,     # window_size (samples)
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')  # output
    ]
    lib.moving_average.restype = None

    print(f"[INFO] C++ Library loaded successfully: {dll_name}")

except Exception as e:
    print(f"[ERROR] Failed to load DLL: {e}")
    lib = None

# =========================================================
# 2. Smoothing Function Wrapper
# =========================================================
def apply_moving_average(power_data, fs, window_sec=0.5):
    """
    Smooths the power data using the C++ backend.
    
    Args:
        power_data (np.array): 2D array (Channels x Samples).
        fs (float): Sampling frequency.
        window_sec (float): Length of the smoothing window in seconds.
                            Standard for ERD is often 0.25s to 1.0s.
    
    Returns:
        smoothed_data (np.array): Same shape as input.
    """
    if lib is None:
        raise RuntimeError("C++ Library not loaded.")

    n_channels, n_samples = power_data.shape
    window_samples = int(window_sec * fs)
    
    # Initialize output array
    smoothed_data = np.zeros_like(power_data, dtype=np.float64)
    
    # Process each channel individually
    # The C++ function is designed for 1D arrays.
    for i in range(n_channels):
        # 1. Prepare C-compatible arrays
        channel_in = np.ascontiguousarray(power_data[i], dtype=np.float64)
        channel_out = np.zeros(n_samples, dtype=np.float64)
        
        # 2. Call C++
        lib.moving_average(channel_in, n_samples, window_samples, channel_out)
        
        # 3. Store result
        smoothed_data[i, :] = channel_out

    return smoothed_data

# =========================================================
# Unit Test
# =========================================================
if __name__ == "__main__":
    # Create a noisy Step Function
    fs = 250.0
    t = np.linspace(0, 4, int(4*fs))
    
    # Signal: 0 until t=2, then jumps to 10 (Simulating ERS rebound)
    # Add heavy random noise
    clean_signal = np.zeros_like(t)
    clean_signal[t >= 2.0] = 10.0
    noise = np.random.randn(len(t)) * 3.0
    noisy_signal = clean_signal + noise
    
    # We must reshape to (1, samples) because the function expects 2D
    input_2d = noisy_signal.reshape(1, -1)
    
    print("Applying Moving Average (Window = 1.0s)...")
    
    try:
        smoothed_2d = apply_moving_average(input_2d, fs, window_sec=1.0)
        smoothed_signal = smoothed_2d[0, :]
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(t, noisy_signal, color='lightgray', label='Noisy Input')
        plt.plot(t, clean_signal, 'k--', label='True Underlying Trend')
        plt.plot(t, smoothed_signal, 'r', linewidth=2, label='Smoothed (C++ Output)')
        
        plt.title("Test: Moving Average Smoothing")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Test Failed: {e}")