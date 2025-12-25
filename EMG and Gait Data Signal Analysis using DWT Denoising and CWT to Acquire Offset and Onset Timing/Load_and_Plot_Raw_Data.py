import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

# Target Channel Name Constants (According to Physionet Header)
# Use a list of strings to allow flexible matching (handling trailing spaces etc.)
TARGET_CHANNELS = {
    'FS': ['baso RT FOOT', 'baso RT FOOT '],   # Foot Switch (Right Foot)
    'GL': ['semg RT LAT.G', 'semg RT LAT.G '], # Gastrocnemius Lateralis (Right)
    'VL': ['semg RT LAT.V', 'semg RT LAT.V ']  # Vastus Lateralis (Right)
}

def get_channel_index(signal_names, target_variations):
    for i, name in enumerate(signal_names):
        # Clean whitespace and match
        if name.strip() in [t.strip() for t in target_variations]:
            return i
    return -1

def load_raw_data(record_path):
    """
    Args:
        record_path (str): Path to the record file (without extension), e.g., 'dataset/S01'
        
    Returns:
        dict: Dictionary containing raw signals (GL, VL, FS), time array, and sampling rate
    """
    try:
        # Read record using wfdb library
        # rdrecord will read both the header (.hea) and the signal (.dat)
        record = wfdb.rdrecord(record_path)
        
        signal_names = record.sig_name
        fs = record.fs  # Sampling frequency
        
        print(f"[-] Loading Record: {record.record_name}")
        print(f"[-] Sampling Rate: {fs} Hz")
        print(f"[-] Total Duration: {record.sig_len / fs:.2f} seconds")

        # Dynamically find channel indices
        idx_fs = get_channel_index(signal_names, TARGET_CHANNELS['FS'])
        idx_gl = get_channel_index(signal_names, TARGET_CHANNELS['GL'])
        idx_vl = get_channel_index(signal_names, TARGET_CHANNELS['VL'])

        # Validate if channels are found
        if -1 in [idx_fs, idx_gl, idx_vl]:
            missing = []
            if idx_fs == -1: missing.append("Foot Switch (FS)")
            if idx_gl == -1: missing.append("Gastrocnemius (GL)")
            if idx_vl == -1: missing.append("Vastus (VL)")
            raise ValueError(f"The following channels were not found in {record_path}: {', '.join(missing)}")

        print(f"[-] Channels found at indices: FS={idx_fs}, GL={idx_gl}, VL={idx_vl}")

        # Extract Signals
        # record.p_signal is a numpy array (samples x channels)
        raw_fs = record.p_signal[:, idx_fs]
        raw_gl = record.p_signal[:, idx_gl]
        raw_vl = record.p_signal[:, idx_vl]
        
        # Create time array
        total_samples = len(raw_fs)
        time = np.arange(total_samples) / fs

        # Return structured dictionary
        data_dict = {
            'record_name': record.record_name,
            'fs': fs,
            'time': time,
            'signal_fs': raw_fs,
            'signal_gl': raw_gl,
            'signal_vl': raw_vl,
            'indices': {'FS': idx_fs, 'GL': idx_gl, 'VL': idx_vl}
        }
        
        return data_dict

    except Exception as e:
        print(f"[!] Error loading data: {e}")
        return None

def plot_raw_signals(data_dict):
    if data_dict is None:
        return

    time = data_dict['time']
    fs_sig = data_dict['signal_fs']
    gl_sig = data_dict['signal_gl']
    vl_sig = data_dict['signal_vl']

    plt.figure(figsize=(12, 8))
    
    # Plot Foot Switch
    plt.subplot(3, 1, 1)
    plt.plot(time, fs_sig, color='black', label='Foot Switch (Basography)')
    plt.title(f"Raw Signals - Record: {data_dict['record_name']}")
    plt.ylabel('Amplitude (V)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    # Plot Gastrocnemius Lateralis
    plt.subplot(3, 1, 2)
    plt.plot(time, gl_sig, color='#1f77b4', label='EMG - Gastrocnemius Lat (GL)') # Standard Blue
    plt.ylabel('Amplitude (mV/uV)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    # Plot Vastus Lateralis
    plt.subplot(3, 1, 3)
    plt.plot(time, vl_sig, color='#d62728', label='EMG - Vastus Lat (VL)') # Red
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (mV/uV)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# <<<< Standalone testing >>>>
if __name__ == "__main__":
    test_file = "S01" 
    
    # Check if file exists in current directory (for testing)
    if os.path.exists(test_file + ".hea"):
        print("File found, processing...")
        data = load_raw_data(test_file)
        plot_raw_signals(data)
    else:
        print(f"File {test_file}.hea not found. Ensure path is correct during testing.")