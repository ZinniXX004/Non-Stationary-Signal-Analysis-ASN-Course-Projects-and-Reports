"""
CWT.py (FIXED v3.1)

Purpose:
    - Perform Continuous Wavelet Transform (CWT) on EEG data.
    - Support Multi-Channel Analysis (C3, Cz, C4).
    - Interface with C++ backend ('eeg_processing.dll').
    - FIXED: get_cwt_interpretation now accepts channel_name argument.

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

# 1. Load C++ Library
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

# 2. Context & Description Helper (FIXED)
def get_cwt_interpretation(channel_name="Unknown"):
    """
    Returns a descriptive string explaining the Multi-Channel CWT analysis.
     Accepts 'channel_name' to provide specific context for C3/Cz/C4.
    
    Args:
        channel_name (str): Name of the channel (e.g., 'C3').
    
    Returns:
        str: Educational text about ERD/ERS.
    """
    
    # Specific context based on channel
    lateral_info = ""
    if "C3" in channel_name:
        lateral_info = "   - **C3 (Left Hemi):** Expect ERD (Blue) during RIGHT hand movement."
    elif "C4" in channel_name:
        lateral_info = "   - **C4 (Right Hemi):** Expect ERD (Blue) during LEFT hand movement."
    elif "Cz" in channel_name:
        lateral_info = "   - **Cz (Vertex):** Involved in foot movement or general idling."
        
    description = (
        f"--- TIME-FREQUENCY ANALYSIS: CHANNEL {channel_name} ---\n\n"
        "1. OBJECTIVE:\n"
        "   To visualize the Event-Related Desynchronization (ERD) and Synchronization (ERS)\n"
        "   patterns across the Sensorimotor Cortex.\n\n"
        "2. PHYSIOLOGICAL EXPECTATION (Contralateral Effect):\n"
        f"{lateral_info}\n"
        "   - **ERD (Blue):** Active cortical processing (Desynchronization).\n"
        "   - **ERS (Red):** Idling or Inhibition (Synchronization).\n\n"
        "3. FREQUENCY BANDS:\n"
        "   - **Mu (8-13 Hz):** Primary motor rhythm.\n"
        "   - **Beta (13-30 Hz):** Active processing and post-movement rebound.\n"
    )
    return description

# 3. Core CWT Function (Single Channel)
def run_cwt_single(eeg_data, fs, f_min=4, f_max=40, f_step=1.0, wavelet_type='morlet'):
    """
    Computes CWT for a single 1D array. Helper function.
    """
    if lib is None:
        raise RuntimeError("C++ Library is not loaded.")

    n_samples = len(eeg_data)
    freqs = np.arange(f_min, f_max + 0.1, f_step)
    
    # Scale conversion (s = fs / f)
    with np.errstate(divide='ignore'):
        scales = fs / freqs
    
    n_scales = len(scales)

    # Prepare Ctypes Arrays
    input_c = np.ascontiguousarray(eeg_data, dtype=np.float64)
    scales_c = np.ascontiguousarray(scales, dtype=np.float64)
    output_c = np.zeros(n_scales * n_samples, dtype=np.float64)

    # Call C++
    if wavelet_type == 'mexican_hat':
        lib.compute_cwt_mexican_hat(input_c, n_samples, scales_c, n_scales, fs, output_c)
    else:
        lib.compute_cwt_magnitude(input_c, n_samples, scales_c, n_scales, fs, output_c)

    # Reshape (Freqs x Time)
    tfr_data = output_c.reshape((n_scales, n_samples))
    
    return tfr_data, freqs

# 4. Multi-Channel Wrapper (Preserved)
def run_cwt_multi_channel(eeg_data_3ch, fs, f_min=4, f_max=40, f_step=1.0, wavelet_type='morlet'):
    """
    Computes CWT for C3, Cz, and C4 sequentially.

    Args:
        eeg_data_3ch (np.array): 2D array (3 channels, n_samples).
                                 Order MUST be [C3, Cz, C4].
        fs (float): Sampling rate.
        
    Returns:
        list: [tfr_C3, tfr_Cz, tfr_C4] - Three 2D matrices.
        np.array: Frequency axis.
    """
    if eeg_data_3ch.shape[0] != 3:
        print(f"[WARN] Expected 3 channels, got {eeg_data_3ch.shape[0]}. Processing anyway.")

    tfr_results = []
    freqs = None

    channel_names = ['C3', 'Cz', 'C4']

    for i in range(eeg_data_3ch.shape[0]):
        # print(f"[INFO] Computing CWT for Channel {channel_names[i]}...")
        tfr, f_axis = run_cwt_single(eeg_data_3ch[i, :], fs, f_min, f_max, f_step, wavelet_type)
        tfr_results.append(tfr)
        freqs = f_axis

    return tfr_results, freqs

# 5. Wrapper for Single Channel Call (Legacy Support)
def run_cwt(eeg_data, fs, f_min=4, f_max=40, f_step=1.0, wavelet_type='morlet'):
    """
    Wrapper to allow GUI to call run_cwt directly on a single channel segment.
    This maintains compatibility with the GUI's single-channel plotting logic.
    """
    return run_cwt_single(eeg_data, fs, f_min, f_max, f_step, wavelet_type)

# Unit Test (Standalone Execution)
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: CWT.py (Multi-Channel)")
    
    # 1. Simulate 3-Channel Signal
    fs = 250.0
    duration = 4.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # C3: 10Hz (Mu) - Disappears at t=2s (Simulated ERD)
    sig_c3 = np.sin(2 * np.pi * 10 * t) * 5.0
    sig_c3[t > 2.0] *= 0.2 
    
    # Cz: 20Hz (Beta) - Constant
    sig_cz = np.sin(2 * np.pi * 20 * t) * 3.0
    
    # C4: 10Hz (Mu) - Increases at t=2s (Simulated ERS/Idle)
    sig_c4 = np.sin(2 * np.pi * 10 * t) * 5.0
    sig_c4[t > 2.0] *= 2.0 
    
    # Stack into (3, samples)
    data_3ch = np.vstack([sig_c3, sig_cz, sig_c4])
    
    try:
        # 2. Run Multi-Channel CWT
        print("\n[TEST] Processing 3 Channels...")
        results, freqs = run_cwt_multi_channel(data_3ch, fs)
        
        # 3. Test Description Function
        print(f"\n[TEST] Description for C3: {get_cwt_interpretation('C3')}")
        
        # 4. Visualize
        print("\n[TEST] Plotting Results...")
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        ch_names = ['C3 (Simulated ERD)', 'Cz (Constant Beta)', 'C4 (Simulated ERS)']
        
        extent = [t[0], t[-1], freqs[0], freqs[-1]]
        
        for i, ax in enumerate(axes):
            im = ax.imshow(results[i], extent=extent, aspect='auto', origin='lower', cmap='jet')
            ax.set_title(f"Spectrogram: {ch_names[i]}")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, label='Power')
            
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()
        
        print("\n[TEST] Multi-Channel CWT Passed.")
        
    except Exception as e:
        print(f"\n[TEST] Failed: {e}")