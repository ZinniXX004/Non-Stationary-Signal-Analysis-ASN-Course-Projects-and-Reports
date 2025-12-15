"""
CWT.py

Purpose:
    - Perform Continuous Wavelet Transform on EEG data.
    - Supports 'Morlet' and 'Mexican Hat' wavelets via C++ backend.
    
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

    # -- BINDING FOR MORLET --
    lib.compute_cwt_magnitude.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    lib.compute_cwt_magnitude.restype = None

    # -- BINDING FOR MEXICAN HAT --
    lib.compute_cwt_mexican_hat.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    ]
    lib.compute_cwt_mexican_hat.restype = None

    print(f"[INFO] C++ Library loaded successfully: {dll_name}")

except Exception as e:
    print(f"[ERROR] Could not load C++ library. CWT will fail. {e}")
    lib = None

# =========================================================
# 2. CWT Function Wrapper
# =========================================================
def run_cwt(eeg_data, fs, f_min=4, f_max=40, f_step=1.0, wavelet_type='morlet'):
    """
    Computes the CWT Magnitude using the C++ backend.

    Args:
        eeg_data (np.array): 1D array of EEG signal.
        fs (float): Sampling frequency.
        f_min, f_max, f_step: Frequency range (Hz).
        wavelet_type (str): 'morlet' or 'mexican_hat'.

    Returns:
        tfr_data (np.array): 2D array [Frequencies x Time].
        freqs (np.array): Array of frequencies used.
    """
    if lib is None:
        raise RuntimeError("C++ Library is not loaded.")

    n_samples = len(eeg_data)
    
    # 1. Define Frequencies and Scales
    freqs = np.arange(f_min, f_max + 0.1, f_step)
    
    # Scale conversion depends on the Wavelet type
    # For Morlet (w0=6), Scale ~= fs / freq
    # For Mexican Hat, Scale ~= (sqrt(2.5)*fs) / (2*pi*freq) approx.
    # To keep it comparable for visualization, we use the standard f = 1/s relationship.
    scales = fs / freqs
    n_scales = len(scales)

    # 2. Prepare Ctypes Arrays
    input_c = np.ascontiguousarray(eeg_data, dtype=np.float64)
    scales_c = np.ascontiguousarray(scales, dtype=np.float64)
    output_c = np.zeros(n_scales * n_samples, dtype=np.float64)

    # 3. Call C++ Function based on Type
    if wavelet_type == 'mexican_hat':
        lib.compute_cwt_mexican_hat(input_c, n_samples, scales_c, n_scales, fs, output_c)
    else:
        # Default to Morlet
        lib.compute_cwt_magnitude(input_c, n_samples, scales_c, n_scales, fs, output_c)

    # 4. Reshape Output
    tfr_data = output_c.reshape((n_scales, n_samples))
    
    return tfr_data, freqs

# =========================================================
# Unit Test
# =========================================================
if __name__ == "__main__":
    fs = 250.0
    t = np.linspace(0, 1, int(fs))
    signal = np.sin(2 * np.pi * 10 * t) # 10 Hz
    
    try:
        print("Testing Morlet...")
        tfr1, _ = run_cwt(signal, fs, wavelet_type='morlet')
        print(f"Morlet Shape: {tfr1.shape}")

        print("Testing Mexican Hat...")
        tfr2, _ = run_cwt(signal, fs, wavelet_type='mexican_hat')
        print(f"Mexican Hat Shape: {tfr2.shape}")
        
    except Exception as e:
        print(f"Test Failed: {e}")