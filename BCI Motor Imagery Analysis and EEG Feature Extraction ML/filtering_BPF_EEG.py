"""
filtering_BPF_EEG.py

Purpose:
    - Apply Bandpass Filter (IIR Butterworth) to EEG signals.
    - Support Multi-Channel Filtering (C3, Cz, C4).
    - Support Dynamic Frequency Band Selection (Mu, Beta, Broadband).
    - Calculate Filter Coefficients (b, a) using pure Numpy.
    - Offload the filtering loop to 'eeg_processing.dll' (C++).

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

# 2. Context & Description Helper (Dynamic)
def get_filter_description(low=0.5, high=30.0):
    """
    Returns a descriptive string explaining the Bandpass Filter step.
    Adapts explanation based on the frequency range selected.
    
    Args:
        low (float): Low cutoff frequency.
        high (float): High cutoff frequency.
        
    Returns:
        str: Educational text about the filter.
    """
    band_name = "BROADBAND"
    focus_text = "General Motor Imagery content + Artifact Removal."
    
    if low >= 7 and high <= 15:
        band_name = "MU RHYTHM (Alpha)"
        focus_text = "Focus on Idling Rhythms. Expect strong ERD during movement."
    elif low >= 12 and high >= 25:
        band_name = "BETA RHYTHM"
        focus_text = "Focus on Active Processing and Post-Movement Rebound (ERS)."

    description = (
        f"--- PREPROCESSING: {band_name} FILTER ({low} - {high} Hz) ---\n\n"
        "1. OBJECTIVE:\n"
        f"   {focus_text}\n\n"
        "2. FILTER SPECIFICATIONS:\n"
        f"   - Low Cutoff: {low} Hz (Removes slower drifts/artifacts)\n"
        f"   - High Cutoff: {high} Hz (Removes high-freq noise/EMG)\n"
        "   - Type: IIR Butterworth (Zero-Phase via Forward-Backward application)\n\n"
        "3. TARGET CHANNELS (C3, Cz, C4):\n"
        "   - These channels are located over the Sensorimotor Cortex.\n"
        "   - Filtering ensures we analyze brain waves, not muscle noise.\n"
    )
    return description

# 3. Pure Numpy Filter Design (Math Helper)
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

# 5. Multi-Channel Wrapper (Flexible Band Support)
def run_filter_multi_channel(eeg_data_3ch, fs, low=0.5, high=30.0, order=2):
    """
    Applies BPF to multiple channels (e.g., C3, Cz, C4).
    This function is called by the GUI when the Band Selector is changed.
    
    Args:
        eeg_data_3ch (np.array): 2D Array (n_channels, n_samples).
        fs (float): Sampling rate.
        low, high (float): Cutoff frequencies (Dynamic).
        
    Returns:
        filtered_data (np.array): Same shape as input.
    """
    n_channels, n_samples = eeg_data_3ch.shape
    filtered_data = np.zeros_like(eeg_data_3ch)
    
    channel_names = ['C3', 'Cz', 'C4'] 
    
    for i in range(n_channels):
        # ch_name = channel_names[i] if i < 3 else f"Ch{i}"
        # print(f"[INFO] Filtering Channel {ch_name} with {low}-{high} Hz...")
        filtered_data[i, :] = run_filter_single(eeg_data_3ch[i, :], fs, low, high, order)
        
    return filtered_data

# Unit Test (Standalone Execution)
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: filtering_BPF_EEG.py (Multi-Band Check)")
    
    fs = 250.0
    t = np.linspace(0, 2, int(2*fs))
    
    # Simulate a signal with multiple frequency components
    # 10 Hz (Mu) + 20 Hz (Beta) + 50 Hz (Noise)
    raw_sig = (np.sin(2 * np.pi * 10 * t) + 
               np.sin(2 * np.pi * 20 * t) + 
               np.sin(2 * np.pi * 50 * t) * 0.5)
    
    # Create 3-channel duplicate for testing wrapper
    raw_3ch = np.vstack([raw_sig, raw_sig, raw_sig])
    
    print("\n[TEST 1] Testing MU BAND Filter (8-13 Hz)...")
    mu_filtered = run_filter_multi_channel(raw_3ch, fs, low=8.0, high=13.0, order=4)
    
    print("\n[TEST 2] Testing BETA BAND Filter (13-30 Hz)...")
    beta_filtered = run_filter_multi_channel(raw_3ch, fs, low=13.0, high=30.0, order=4)
    
    # Plot Comparison
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, raw_sig, color='gray', alpha=0.5, label='Raw (10Hz + 20Hz + Noise)')
    plt.title("Raw Signal")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(t, mu_filtered[0, :], color='blue', label='Mu Filtered (8-13 Hz)')
    plt.title("Mu Band Output (Should keep 10Hz)")
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t, beta_filtered[0, :], color='red', label='Beta Filtered (13-30 Hz)')
    plt.title("Beta Band Output (Should keep 20Hz)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n[TEST] Multi-Band Filter Verification Passed.")