"""
moving_average_EEG.py

Purpose:
    - Apply a Moving Average (Sliding Window) filter to smooth the Power signal.
    - Offload the loop iteration to 'eeg_processing.dll' (C++).
    - Window size is defined in seconds (converted to samples based on fs).
    - Provide educational context regarding Signal Smoothing/Envelope Extraction.

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

# =========================================================
# 1. Load C++ Library
# =========================================================
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

# =========================================================
# 2. Context & Description Helper
# =========================================================
def get_smoothing_description():
    """
    Returns a descriptive string explaining the Moving Average step.
    
    Returns:
        str: Educational text about Envelope Extraction.
    """
    description = (
        "--- POST-PROCESSING: MOVING AVERAGE (SMOOTHING) ---\n\n"
        "1. OBJECTIVE:\n"
        "   To extract the 'envelope' or the general trend of the power signal.\n\n"
        "2. WHY IS THIS NECESSARY?\n"
        "   - The Squared Signal (Instantaneous Power) is extremely jagged and noisy.\n"
        "   - We are interested in the *time course* of energy changes over seconds,\n"
        "     not the rapid millisecond-by-millisecond fluctuations.\n\n"
        "3. METHOD (Sliding Window):\n"
        "   - We calculate the arithmetic mean of data points within a sliding window.\n"
        "   - Window Size matters: \n"
        "     * Too small: Signal remains noisy.\n"
        "     * Too large: Temporal resolution is lost (blurs the start/end of ERD).\n"
        "   - Standard Window: 0.5 to 1.0 seconds for Motor Imagery analysis.\n"
    )
    return description

# =========================================================
# 3. Smoothing Function Wrapper
# =========================================================
def apply_moving_average(power_data, fs, window_sec=0.5):
    """
    Smooths the power data using the C++ backend.
    
    Args:
        power_data (np.array): Input array (Channels x Samples) or (Samples,).
        fs (float): Sampling frequency.
        window_sec (float): Length of the smoothing window in seconds.
                            Standard for ERD is often 0.25s to 1.0s.
    
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
    # The C++ function is designed for 1D arrays (double*)
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

# =========================================================
# Unit Test (Standalone Execution)
# =========================================================
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: moving_average_EEG.py")
    
    # 1. Print Description
    print("-" * 60)
    print(get_smoothing_description())
    print("-" * 60)

    # 2. Create a noisy Step Function
    fs = 250.0
    duration = 4.0
    t = np.linspace(0, duration, int(duration * fs))
    
    # Signal: 0 until t=2, then jumps to 10 (Simulating ERS rebound)
    clean_signal = np.zeros_like(t)
    clean_signal[t >= 2.0] = 10.0
    
    # Add heavy random noise
    noise = np.random.randn(len(t)) * 3.0
    noisy_signal = clean_signal + noise
    
    print(f"\n[TEST] Applying Moving Average (Window = 1.0s, fs={fs}Hz)...")
    
    try:
        smoothed_signal = apply_moving_average(noisy_signal, fs, window_sec=1.0)
        
        # 3. Plot Comparison
        plt.figure(figsize=(10, 5))
        plt.plot(t, noisy_signal, color='lightgray', label='Noisy Input (Squared Signal)')
        plt.plot(t, clean_signal, 'k--', label='True Underlying Trend')
        plt.plot(t, smoothed_signal, 'r', linewidth=2, label='Smoothed Output (Envelope)')
        
        plt.title("Test: Moving Average Smoothing")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("\n[TEST] Smoothing Module Verification Passed.")
        
    except Exception as e:
        print(f"\n[TEST] Failed: {e}")