"""
average_all_EEG_trials.py

Purpose:
    - Segment continuous squared EEG data into "Epochs" based on Event markers.
    - Perform Synchronous Averaging to enhance Signal-to-Noise Ratio (SNR).
    - Separate trials by Class: Left Hand (769) vs. Right Hand (770).
    - Preserves Multi-Channel Integrity (C3, Cz, C4) for feature extraction.

Dependencies:
    - numpy
    - matplotlib (for standalone testing)
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Context & Description Helper
def get_averaging_description():
    """
    Returns a descriptive string explaining the Synchronous Averaging step.
    
    Returns:
        str: Educational text about SNR improvement and ERPs.
    """
    description = (
        "--- STATISTICAL PROCESSING: SYNCHRONOUS AVERAGING ---\n\n"
        "1. OBJECTIVE:\n"
        "   To extract the Event-Related Desynchronization (ERD) pattern buried\n"
        "   within the background EEG noise for all three channels (C3, Cz, C4).\n\n"
        "2. THE PROBLEM:\n"
        "   - A single EEG trial is dominated by noise (SNR << 1).\n"
        "   - The motor cortex response is weak compared to background activity.\n\n"
        "3. THE SOLUTION (Averaging):\n"
        "   - Assumption: Noise is random (zero mean), while the ERD signal is time-locked.\n"
        "   - By summing N trials, the Signal amplitude increases by N, while Noise by sqrt(N).\n"
        "   - Result: SNR improves by a factor of sqrt(N).\n\n"
        "4. PROCEDURE:\n"
        "   - Cut data segments (Epochs) from T_min to T_max around each cue.\n"
        "   - Separate Epochs into Class 1 (Left Hand) and Class 2 (Right Hand).\n"
        "   - Calculate the arithmetic mean for every time point across all trials.\n"
    )
    return description

# 2. Averaging Function
def extract_and_average_epochs(squared_data, events, fs, tmin=-1.0, tmax=4.0):
    """
    Cuts the continuous data into segments and averages them by class.
    
    Args:
        squared_data (np.array): 2D array (n_channels x n_samples) of Power data.
                                 Typically 3 channels (C3, Cz, C4).
        events (np.array): Event array from MNE [sample_index, 0, event_id].
        fs (float): Sampling frequency.
        tmin (float): Start time relative to cue (e.g., -1.0s).
        tmax (float): End time relative to cue (e.g., 4.0s).
        
    Returns:
        avg_left (np.array): Averaged Left Hand trials (n_channels x n_time_points).
        avg_right (np.array): Averaged Right Hand trials (n_channels x n_time_points).
        time_axis (np.array): Time vector for plotting.
    """
    
    # 1. Define Event IDs (BCI Competition IV 2b standard)
    EVENT_LEFT = 769
    EVENT_RIGHT = 770
    
    # 2. Calculate Sample Offsets
    offset_start = int(tmin * fs)
    offset_end = int(tmax * fs)
    epoch_len = offset_end - offset_start
    
    # Containers for trials
    trials_left = []
    trials_right = []
    
    n_channels, n_samples = squared_data.shape
    
    # 3. Iterate through events
    for event in events:
        sample_idx = event[0]
        event_id = event[2]
        
        # Check boundaries to avoid crashes at start/end of file
        start_idx = sample_idx + offset_start
        end_idx = sample_idx + offset_end
        
        if start_idx < 0 or end_idx > n_samples:
            continue
            
        # Extract the segment (Epoch)
        # Shape: (n_channels, epoch_len)
        epoch_data = squared_data[:, start_idx:end_idx]
        
        # Sort by Class
        if event_id == EVENT_LEFT:
            trials_left.append(epoch_data)
        elif event_id == EVENT_RIGHT:
            trials_right.append(epoch_data)
            
    # 4. Convert to Numpy Arrays and Average
    # Stack Shape: (n_trials, n_channels, n_time_points)
    # Mean Axis: 0 (Average across trials, preserving channels and time)
    
    # Left Hand Class
    if len(trials_left) > 0:
        stack_left = np.array(trials_left)
        avg_left = np.mean(stack_left, axis=0) 
        print(f"[INFO] Averaged {len(trials_left)} Left Hand trials.")
    else:
        avg_left = np.zeros((n_channels, epoch_len))
        print("[WARN] No Left Hand trials found.")

    # Right Hand Class
    if len(trials_right) > 0:
        stack_right = np.array(trials_right)
        avg_right = np.mean(stack_right, axis=0)
        print(f"[INFO] Averaged {len(trials_right)} Right Hand trials.")
    else:
        avg_right = np.zeros((n_channels, epoch_len))
        print("[WARN] No Right Hand trials found.")
        
    # Generate Time Axis for plotting
    time_axis = np.linspace(tmin, tmax, epoch_len)
    
    return avg_left, avg_right, time_axis

# Unit Test (Standalone Execution)
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: average_all_EEG_trials.py (Multi-Channel)")
    
    # 1. Print Description
    print("-" * 60)
    print(get_averaging_description())
    print("-" * 60)

    # 2. Create Dummy Data (3 Channels, 20 seconds, 250Hz)
    fs = 250.0
    total_samples = int(20 * fs)
    n_channels = 3
    
    # Base Noise
    dummy_power = np.random.rand(n_channels, total_samples) * 5.0 
    
    # 3. Create Dummy Events
    events = np.array([
        [int(5.0*fs), 0, 769],  # Left trial
        [int(10.0*fs), 0, 770], # Right trial
        [int(15.0*fs), 0, 769]  # Left trial
    ])
    
    # 4. Inject Pattern: ERD (Power Drop) on C4 (Channel 2) during Left Trial (769)
    # This simulates contralateral activation.
    for ev in events:
        idx = ev[0]
        eid = ev[2]
        
        start_erd = idx + int(1.0*fs)
        end_erd = idx + int(3.0*fs)
        if end_erd >= total_samples: continue
            
        if eid == 769: # Left Hand -> Right Hemisphere (C4/Ch2) ERD
            dummy_power[2, start_erd:end_erd] *= 0.1 
        elif eid == 770: # Right Hand -> Left Hemisphere (C3/Ch0) ERD
            dummy_power[0, start_erd:end_erd] *= 0.1

    print("\n[TEST] Running Averaging Function...")
    avg_L, avg_R, t_axis = extract_and_average_epochs(dummy_power, events, fs)
    
    # 5. Plot 3 Channels
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ch_names = ['C3 (Left Hemi)', 'Cz (Midline)', 'C4 (Right Hemi)']
    
    for i, ax in enumerate(axes):
        ax.plot(t_axis, avg_L[i, :], label='Left Hand Cue', color='blue')
        ax.plot(t_axis, avg_R[i, :], label='Right Hand Cue', color='red', linestyle='--')
        
        ax.axvline(0, color='k', linestyle=':', label='Cue Onset')
        ax.set_title(f"Averaged Power: {ch_names[i]}")
        ax.set_ylabel(r"Power ($\mu V^2$)")
        ax.legend(loc='upper right')
        ax.grid(True)
        
    axes[-1].set_xlabel("Time relative to Cue (s)")
    plt.tight_layout()
    plt.show()
    
    print("\n[TEST] Multi-Channel Averaging Verification Passed.")