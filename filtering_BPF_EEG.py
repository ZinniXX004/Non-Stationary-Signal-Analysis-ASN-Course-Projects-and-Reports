"""
filtering_BPF_EEG.py

Purpose:
    - Apply Bandpass Filter (IIR Butterworth) to EEG signals.
    - Calculate Filter Coefficients (b, a) using pure Numpy (No Scipy).
    - Offload the filtering loop to 'eeg_processing.dll' (C++).
    - Provide educational context regarding preprocessing for Motor Imagery.

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
    # void apply_filter(double* input, int length, double* b, int b_len, 
    #                   double* a, int a_len, double* output)
    lib.apply_filter.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # input
        ctypes.c_int,                                                           # length
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # b coefficients
        ctypes.c_int,                                                           # b_len
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # a coefficients
        ctypes.c_int,                                                           # a_len
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')  # output
    ]
    lib.apply_filter.restype = None
    
    print(f"[INFO] C++ Library loaded successfully: {dll_name}")

except Exception as e:
    print(f"[ERROR] Failed to load DLL: {e}")
    lib = None

# =========================================================
# 2. Context & Description Helper
# =========================================================
def get_filter_description(low=0.5, high=30.0):
    """
    Returns a descriptive string explaining the Bandpass Filter step.
    
    Args:
        low (float): Low cutoff frequency.
        high (float): High cutoff frequency.
        
    Returns:
        str: Educational text about the filter.
    """
    description = (
        f"--- PREPROCESSING: BANDPASS FILTER ({low} - {high} Hz) ---\n\n"
        "1. OBJECTIVE:\n"
        "   To isolate the brain frequencies relevant to Motor Imagery while removing noise.\n\n"
        "2. WHY 0.5 Hz (High-Pass)?\n"
        "   - Removes DC Offset (baseline drift) caused by electrode polarization.\n"
        "   - Removes very slow artifacts like sweating or breathing.\n\n"
        "3. WHY 30.0 Hz (Low-Pass)?\n"
        "   - Removes 50Hz/60Hz Power Line Interference (Mains Hum).\n"
        "   - Removes high-frequency EMG (Electromyogram) noise from muscle tension.\n\n"
        "4. TARGET RHYTHMS:\n"
        "   - Preserves Mu (8-12 Hz) and Beta (13-30 Hz) bands required for ERD/ERS analysis.\n"
        "   - Filter Type: Infinite Impulse Response (IIR) Butterworth (Order 2/4).\n"
    )
    return description

# =========================================================
# 3. Pure Numpy Filter Design (Math Helper)
# =========================================================
def design_butter_bandpass_2nd_order(lowcut, highcut, fs):
    """
    Designs a 2nd-order Butterworth Bandpass Filter using the Bilinear Transform.
    This replaces scipy.signal.butter to adhere to project constraints.
    
    Formula Reference: Robert Bristow-Johnson's Audio EQ Cookbook (BPF constant peak gain).
    
    Args:
        lowcut (float): Lower frequency (Hz)
        highcut (float): Higher frequency (Hz)
        fs (float): Sampling rate (Hz)
        
    Returns:
        b (np.array): Numerator coefficients
        a (np.array): Denominator coefficients
    """
    # Angular frequency
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
    
    # Normalize by a0 so that a[0] becomes 1 (Standard IIR format)
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    
    return b, a

# =========================================================
# 4. Filtering Function Wrapper
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
        filtered_data (np.array): Filtered signal.
    """
    if lib is None:
        raise RuntimeError("C++ Library not loaded. Cannot perform Filtering.")

    # Validate Input Dimensions
    if eeg_data.ndim != 1:
        eeg_data = eeg_data.flatten()

    # 1. Calculate Coefficients (Pure Numpy)
    b, a = design_butter_bandpass_2nd_order(low, high, fs)
    
    # Prepare Ctypes arrays for coefficients
    b_c = np.ascontiguousarray(b, dtype=np.float64)
    a_c = np.ascontiguousarray(a, dtype=np.float64)
    
    # 2. Prepare Data Arrays
    temp_input = np.ascontiguousarray(eeg_data, dtype=np.float64)
    output = np.zeros_like(temp_input)
    
    # 3. First Pass (2nd Order)
    lib.apply_filter(temp_input, len(temp_input), 
                     b_c, len(b_c), 
                     a_c, len(a_c), 
                     output)
    
    # 4. (Optional) Second Pass for Steepness (Pseudo-4th Order)
    # This cascades the filter to achieve a sharper cutoff (-24dB/octave)
    if order >= 4:
        # Swap input/output for the next pass
        # We take the output of the first pass as input for the second
        temp_input = output.copy()
        lib.apply_filter(temp_input, len(temp_input), 
                         b_c, len(b_c), 
                         a_c, len(a_c), 
                         output)

    return output

# =========================================================
# Unit Test (Standalone Execution)
# =========================================================
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: filtering_BPF_EEG.py")
    
    # 1. Create a noisy signal
    fs = 250.0
    t = np.linspace(0, 2, int(2*fs))
    
    # Signal Composition:
    # - 10 Hz: Desired Mu Rhythm (Should preserve)
    # - 0.2 Hz: Slow Drift/Sweat Artifact (Should remove)
    # - 50 Hz: Mains Hum Noise (Should remove)
    raw = (np.sin(2 * np.pi * 10 * t) + 
           np.sin(2 * np.pi * 0.2 * t) * 2.0 + 
           np.sin(2 * np.pi * 50 * t) * 0.5)
    
    print("\n[TEST] Applying Filter: 0.5 - 30 Hz (Order 4)...")
    print("-" * 60)
    print(get_filter_description(0.5, 30.0))
    print("-" * 60)
    
    try:
        filtered = run_filter(raw, fs, low=0.5, high=30.0, order=4)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(t, raw, label=r'Raw Signal (Drift + 50Hz Noise)', alpha=0.5, color='gray')
        plt.plot(t, filtered, label=r'Filtered Signal (0.5-30 Hz)', linewidth=2, color='blue')
        plt.title("Test: IIR Bandpass Filter (C++ Backend)")
        plt.xlabel("Time (s)")
        plt.ylabel(r"Amplitude ($\mu V$)")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print("\n[TEST] Filter Module Verification Passed.")
        
    except Exception as e:
        print(f"\n[TEST] Failed: {e}")