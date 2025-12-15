"""
filtering_BPF_EEG.py

Purpose:
    - Apply Bandpass Filter (IIR Butterworth) to EEG signals.
    - Calculate Filter Coefficients (b, a) using pure Numpy (No Scipy).
    - Offload the filtering loop to 'eeg_processing.dll' (C++).

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

    # void apply_filter(double* input, int length, double* b, int b_len, 
    #                   double* a, int a_len, double* output)
    lib.apply_filter.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # input
        ctypes.c_int,     # length
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # b
        ctypes.c_int,     # b_len
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # a
        ctypes.c_int,     # a_len
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')  # output
    ]
    lib.apply_filter.restype = None
    
    print(f"[INFO] C++ Library loaded successfully: {dll_name}")

except Exception as e:
    print(f"[ERROR] Failed to load DLL: {e}")
    lib = None

# =========================================================
# 2. Pure Numpy Filter Design (Math Helper)
# =========================================================
def design_butter_bandpass_2nd_order(lowcut, highcut, fs):
    """
    Designs a 2nd-order Butterworth Bandpass Filter using the Bilinear Transform.
    This replaces scipy.signal.butter.
    
    Formula Reference: Robert Bristow-Johnson's Audio EQ Cookbook (BPF constant peak gain).
    
    Args:
        lowcut (float): Lower frequency (Hz)
        highcut (float): Higher frequency (Hz)
        fs (float): Sampling rate (Hz)
        
    Returns:
        b (np.array): Numerator coefficients
        a (np.array): Denominator coefficients
    """
    w0 = 2 * np.pi * np.sqrt(lowcut * highcut) / fs
    bw_hz = highcut - lowcut
    Q = np.sqrt(lowcut * highcut) / bw_hz
    
    # Intermediate variables for Bilinear Transform
    alpha = np.sin(w0) / (2 * Q)
    cos_w0 = np.cos(w0)
    
    # Calculate coefficients
    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    
    a0 = 1 + alpha
    a1 = -2 * cos_w0
    a2 = 1 - alpha
    
    # Normalize by a0 so that a[0] becomes 1
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    return b, a

# =========================================================
# 3. Filtering Function Wrapper
# =========================================================
def run_filter(eeg_data, fs, low=0.5, high=30.0, order=2):
    """
    Applies the Bandpass filter to the data via C++.
    
    Args:
        eeg_data (np.array): 1D Raw Signal.
        fs (float): Sampling Rate.
        low, high (float): Cutoff frequencies.
        order (int): 2 or 4. If 4, applies the 2nd order filter twice (cascade).
        
    Returns:
        filtered_data (np.array)
    """
    if lib is None:
        raise RuntimeError("C++ Library not loaded.")

    # 1. Calculate Coefficients (Pure Numpy)
    b, a = design_butter_bandpass_2nd_order(low, high, fs)
    
    # Prepare Ctypes arrays for coefficients
    b_c = np.ascontiguousarray(b, dtype=np.float64)
    a_c = np.ascontiguousarray(a, dtype=np.float64)
    
    # 2. Prepare Data
    temp_input = np.ascontiguousarray(eeg_data, dtype=np.float64)
    output = np.zeros_like(temp_input)
    
    # 3. First Pass (2nd Order)
    lib.apply_filter(temp_input, len(temp_input), 
                     b_c, len(b_c), 
                     a_c, len(a_c), 
                     output)
    
    # 4. (Optional) Second Pass for Steepness (Pseudo-4th Order)
    if order >= 4:
        # Swap input/output for the next pass
        temp_input = output.copy()
        lib.apply_filter(temp_input, len(temp_input), 
                         b_c, len(b_c), 
                         a_c, len(a_c), 
                         output)

    return output

# =========================================================
# Unit Test
# =========================================================
if __name__ == "__main__":
    # Create a noisy signal
    fs = 250.0
    t = np.linspace(0, 2, int(2*fs))
    
    # Signal: 10Hz (Wanted) + 0.2Hz (Drift) + 50Hz (Noise)
    raw = (np.sin(2 * np.pi * 10 * t) + 
           np.sin(2 * np.pi * 0.2 * t) * 2.0 + 
           np.sin(2 * np.pi * 50 * t) * 0.5)
    
    print("Applying Filter: 0.5 - 30 Hz...")
    
    try:
        filtered = run_filter(raw, fs, low=0.5, high=30.0, order=4)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(t, raw, label='Raw (Drift + 50Hz Noise)', alpha=0.5)
        plt.plot(t, filtered, label='Filtered (0.5-30 Hz)', linewidth=2)
        plt.title("Test: IIR Bandpass Filter (C++ Backend)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Test Failed: {e}")