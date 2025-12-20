"""
CWT.py

Purpose:
    - Perform Continuous Wavelet Transform (CWT) on EEG data.
    - Interface with the C++ backend ('eeg_processing.dll') for high-performance convolution.
    - Provide educational context and interpretation for Motor Imagery analysis.

Dependencies:
    - ctypes
    - numpy
    - matplotlib (for standalone testing only)
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

    # -- BINDING FOR MORLET WAVELET --
    # void compute_cwt_magnitude(double* input, int length, double* scales, int num_scales, double fs, double* output)
    lib.compute_cwt_magnitude.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # input signal
        ctypes.c_int,                                                           # signal length
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # scales array
        ctypes.c_int,                                                           # number of scales
        ctypes.c_double,                                                        # sampling frequency
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')  # output flattened array
    ]
    lib.compute_cwt_magnitude.restype = None

    # -- BINDING FOR MEXICAN HAT WAVELET --
    # void compute_cwt_mexican_hat(double* input, int length, double* scales, int num_scales, double fs, double* output)
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
    print(f"[ERROR] Could not load C++ library. CWT will fail. Details: {e}")
    lib = None

# =========================================================
# 2. Context & Description Helper
# =========================================================
def get_cwt_interpretation(channel_name="Unknown"):
    """
    Returns a descriptive string explaining the CWT analysis for the GUI.
    
    Args:
        channel_name (str): The name of the EEG channel being analyzed (e.g., 'C3').
        
    Returns:
        str: Educational text about ERD/ERS and Motor Imagery.
    """
    description = (
        f"--- ANALYSIS CONTEXT FOR CHANNEL {channel_name} ---\n\n"
        "1. OBJECTIVE:\n"
        "   To identify Event-Related Desynchronization (ERD) or Synchronization (ERS) "
        "   in the time-frequency domain.\n\n"
        "2. FREQUENCY BANDS OF INTEREST:\n"
        "   - Mu Rhythm (8-13 Hz): Primary motor cortex oscillation.\n"
        "   - Beta Rhythm (13-30 Hz): Associated with active motor planning.\n\n"
        "3. VISUAL INTERPRETATION:\n"
        "   - BLUE / COOL COLORS: Represent ERD (Power Decrease). This indicates neuronal ACTIVATION "
        "     in the underlying cortical area.\n"
        "   - RED / WARM COLORS: Represent ERS (Power Increase). This indicates neuronal IDLING "
        "     or INHIBITION (often seen as a 'Rebound' after movement).\n\n"
        "4. LATERALIZATION LOGIC:\n"
        "   - Right Hand Movement -> Expect ERD (Blue) primarily in Channel C3 (Left Hemisphere).\n"
        "   - Left Hand Movement  -> Expect ERD (Blue) primarily in Channel C4 (Right Hemisphere).\n"
    )
    return description

# =========================================================
# 3. CWT Function Wrapper
# =========================================================
def run_cwt(eeg_data, fs, f_min=4, f_max=40, f_step=1.0, wavelet_type='morlet'):
    """
    Computes the CWT Magnitude using the C++ backend.

    Args:
        eeg_data (np.array): 1D array of EEG signal (Time domain).
                             Must be a single channel (e.g., just C3).
        fs (float): Sampling frequency (e.g., 250.0).
        f_min, f_max, f_step: Frequency range of interest (Hz).
        wavelet_type (str): 'morlet' or 'mexican_hat'.

    Returns:
        tfr_data (np.array): 2D array [Frequencies x Time].
        freqs (np.array): Array of frequencies used.
    """
    if lib is None:
        raise RuntimeError("C++ Library is not loaded. Cannot perform CWT.")

    # Validate Input
    if eeg_data.ndim != 1:
        # If user passed (1, N) or (N, 1), flatten it.
        eeg_data = eeg_data.flatten()

    n_samples = len(eeg_data)
    
    # 1. Define Frequencies and Convert to Scales
    # We create a frequency vector from min to max.
    # Note: High frequency = Small Scale. Low frequency = Large Scale.
    freqs = np.arange(f_min, f_max + 0.1, f_step)
    
    # Scale conversion formula (Approximation f = 1/s for visualization alignment)
    # The exact relationship depends on the central frequency of the wavelet, 
    # but for standard Time-Frequency mapping, s = fs / f is the standard approach.
    with np.errstate(divide='ignore'):
        scales = fs / freqs
    
    n_scales = len(scales)

    # 2. Prepare Ctypes Arrays (Memory Allocation)
    # Ensure data is float64 (double) and contiguous in memory for C++ pointer access
    input_c = np.ascontiguousarray(eeg_data, dtype=np.float64)
    scales_c = np.ascontiguousarray(scales, dtype=np.float64)
    
    # Allocate flat output array (Size: Scales * Samples)
    output_c = np.zeros(n_scales * n_samples, dtype=np.float64)

    # 3. Call C++ Function based on Wavelet Type
    if wavelet_type == 'mexican_hat':
        lib.compute_cwt_mexican_hat(input_c, n_samples, scales_c, n_scales, fs, output_c)
    else:
        # Default to Morlet (Complex Wavelet Magnitude)
        lib.compute_cwt_magnitude(input_c, n_samples, scales_c, n_scales, fs, output_c)

    # 4. Reshape Output
    # C++ returns a flat array. We reshape it to (Frequencies, Time).
    tfr_data = output_c.reshape((n_scales, n_samples))
    
    return tfr_data, freqs

# =========================================================
# Unit Test (Standalone Execution)
# =========================================================
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: CWT.py")
    
    # 1. Simulate a dummy signal
    # 4 seconds at 250 Hz
    fs = 250.0
    duration = 4.0
    times = np.linspace(0, duration, int(fs * duration))
    
    # Signal: 10 Hz (Alpha) constant + 20 Hz (Beta) burst between 1s and 3s
    signal = np.sin(2 * np.pi * 10 * times) * 5.0 
    beta_burst = np.sin(2 * np.pi * 20 * times) * 10.0
    
    # Add burst
    mask = (times >= 1.0) & (times <= 3.0)
    signal[mask] += beta_burst[mask]
    
    print(f"Generated Test Signal: {len(signal)} samples.")
    
    try:
        # 2. Run Morlet CWT
        print("\n[TEST] Computing Morlet CWT...")
        tfr, f_axis = run_cwt(signal, fs, f_min=5, f_max=30, f_step=0.5, wavelet_type='morlet')
        print(f"CWT Output Shape: {tfr.shape} (Freqs x Time)")
        
        # 3. Print Interpretation
        print("\n[TEST] Fetching Interpretation Text for Channel 'C3':")
        print("-" * 60)
        print(get_cwt_interpretation("C3"))
        print("-" * 60)
        
        # 4. Simple Visualization (if matplotlib available)
        plt.figure(figsize=(10, 6))
        extent = [times[0], times[-1], f_axis[0], f_axis[-1]]
        plt.imshow(tfr, extent=extent, aspect='auto', origin='lower', cmap='jet')
        plt.colorbar(label='Power')
        plt.title("Test CWT: 10Hz Continuous + 20Hz Burst")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.show()
        
        print("\n[TEST] CWT Module Verification Passed.")
        
    except Exception as e:
        print(f"\n[TEST] Failed: {e}")