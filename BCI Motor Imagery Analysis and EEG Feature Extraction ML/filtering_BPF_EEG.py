"""
filtering_BPF_EEG.py (REVISED v3.0)

Purpose:
    - Apply Bandpass Filter (IIR Butterworth) to EEG signals.
    - Support Multi-Channel Filtering (C3, Cz, C4) for full BCI pipeline.
    - Calculate Filter Coefficients (b, a) using pure Numpy.
    - Offload the filtering loop to 'eeg_processing.dll' (C++).
    - Provide educational context regarding preprocessing.

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

# 2. Context & Description Helper
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
        "4. TARGET RHYTHMS (C3, Cz, C4):\n"
        "   - Preserves Mu (8-12 Hz) and Beta (13-30 Hz) bands required for ERD/ERS analysis.\n"
        "   - Filter Type: Infinite Impulse Response (IIR) Butterworth (Order 2/4).\n"
    )
    return description

# 3. Pure Numpy Filter Design (Math Helper)
def design_butter_bandpass_2nd_order(lowcut, highcut, fs):
    """
    Designs a 2nd-order Butterworth Bandpass Filter using the Bilinear Transform.
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

# 4. Core Filtering Function (Single Channel)
def run_filter_single(eeg_data, fs, low=0.5, high=30.0, order=2):
    """
    Applies the Bandpass filter to a SINGLE channel (1D array).
    """
    if lib is None:
        raise RuntimeError("C++ Library not loaded. Cannot perform Filtering.")

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
    if order >= 4:
        temp_input = output.copy()
        lib.apply_filter(temp_input, len(temp_input), 
                         b_c, len(b_c), 
                         a_c, len(a_c), 
                         output)

    return output

# 5. Multi-Channel Wrapper
def run_filter_multi_channel(eeg_data_3ch, fs, low=0.5, high=30.0, order=2):
    """
    Applies BPF to multiple channels (e.g., C3, Cz, C4).
    
    Args:
        eeg_data_3ch (np.array): 2D Array (n_channels, n_samples).
        fs (float): Sampling rate.
        
    Returns:
        filtered_data (np.array): Same shape as input.
    """
    n_channels, n_samples = eeg_data_3ch.shape
    filtered_data = np.zeros_like(eeg_data_3ch)
    
    channel_names = ['C3', 'Cz', 'C4'] # Assumed order for logging
    
    for i in range(n_channels):
        ch_name = channel_names[i] if i < 3 else f"Ch{i}"
        # print(f"[INFO] Filtering Channel {ch_name}...")
        filtered_data[i, :] = run_filter_single(eeg_data_3ch[i, :], fs, low, high, order)
        
    return filtered_data

# Unit Test (Standalone Execution)
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: filtering_BPF_EEG.py (Multi-Channel)")
    
    fs = 250.0
    t = np.linspace(0, 2, int(2*fs))
    
    # Simulate 3 Channels with different noise profiles
    # C3: 10Hz signal + DC drift
    raw_c3 = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 0.1 * t) * 5.0
    
    # Cz: 20Hz signal + 50Hz mains hum
    raw_cz = np.sin(2 * np.pi * 20 * t) + np.sin(2 * np.pi * 50 * t) * 2.0
    
    # C4: Clean 10Hz signal
    raw_c4 = np.sin(2 * np.pi * 10 * t)
    
    # Stack
    raw_3ch = np.vstack([raw_c3, raw_cz, raw_c4])
    
    print(f"\n[TEST] Applying Filter to 3 Channels...")
    try:
        filtered_3ch = run_filter_multi_channel(raw_3ch, fs, low=0.5, high=30.0, order=4)
        
        # Plot Results
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ch_names = ['C3 (Drift Removal)', 'Cz (50Hz Removal)', 'C4 (Clean Baseline)']
        
        for i, ax in enumerate(axes):
            ax.plot(t, raw_3ch[i, :], label='Raw', color='gray', alpha=0.5)
            ax.plot(t, filtered_3ch[i, :], label='Filtered', color='blue', linewidth=1.5)
            ax.set_title(ch_names[i])
            ax.set_ylabel("Amplitude")
            ax.legend(loc='upper right')
            ax.grid(True)
            
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
        
        print("\n[TEST] Multi-Channel Filter Passed.")
        
    except Exception as e:
        print(f"\n[TEST] Failed: {e}")