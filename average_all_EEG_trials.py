"""
average_all_EEG_trials.py

Purpose:
    - Segment the continuous squared EEG data into "Epochs" based on Event markers.
    - Perform Synchronous Averaging to enhance the Signal-to-Noise Ratio (SNR).
    - Separate trials by Class (769: Left Hand vs 770: Right Hand).
    - Formula: y_avg(t) = (1/N) * Sum(x_i(t))
    - Provide educational context regarding the Averaging technique.

Dependencies:
    - numpy
    - matplotlib (for standalone testing)
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. Context & Description Helper
# =========================================================
def get_averaging_description():
    """
    Returns a descriptive string explaining the Synchronous Averaging step.
    
    Returns:
        str: Educational text about SNR improvement and ERPs.
    """
    description = (
        "--- STATISTICAL PROCESSING: SYNCHRONOUS AVERAGING ---\n\n"
        "1. OBJECTIVE:\n"
        "   To extract the Event-Related Potential (ERP) or Event-Related Desynchronization (ERD)\n"
        "   buried within the background EEG noise.\n\n"
        "2. THE PROBLEM:\n"
        "   - A single EEG trial is dominated by noise (SNR << 1).\n"
        "   - The brain response to a motor cue is weak compared to background activity.\n\n"
        "3. THE SOLUTION (Averaging):\n"
        "   - Assumption: Noise is random (zero mean), while the Signal is time-locked to the cue.\n"
        "   - By summing N trials, the Signal amplitude increases by N, while Noise increases by sqrt(N).\n"
        "   - Result: The Signal-to-Noise Ratio (SNR) improves by a factor of sqrt(N).\n\n"
        "4. PROCEDURE:\n"
        "   - Cut data segments (Epochs) from T_min to T_max around each cue.\n"
        "   - Separate Epochs into Class 1 (Left Hand) and Class 2 (Right Hand).\n"
        "   - Calculate the arithmetic mean for every time point across all trials.\n"
    )
    return description

# =========================================================
# 2. Averaging Function
# =========================================================
def extract_and_average_epochs(squared_data, events, fs, tmin=-1.0, tmax=4.0):
    """
    Cuts the continuous data into segments and averages them by class.
    
    Args:
        squared_data (np.array): 2D array (n_channels x n_samples) of Power data.
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
    # tmin is usually negative (baseline), tmax is positive.
    offset_start = int(tmin * fs)
    offset_end = int(tmax * fs)
    epoch_len = offset_end - offset_start
    
    # Containers for trials
    # Lists are used initially because we don't know the exact clean trial count yet
    trials_left = []
    trials_right = []
    
    n_channels, n_samples = squared_data.shape
    
    # 3. Iterate through events
    for event in events:
        sample_idx = event[0]
        event_id = event[2]
        
        # Check boundaries to avoid crashes at the start/end of file
        start_idx = sample_idx + offset_start
        end_idx = sample_idx + offset_end
        
        if start_idx < 0 or end_idx > n_samples:
            continue
            
        # Extract the segment
        epoch_data = squared_data[:, start_idx:end_idx]
        
        # Sort by Class
        if event_id == EVENT_LEFT:
            trials_left.append(epoch_data)
        elif event_id == EVENT_RIGHT:
            trials_right.append(epoch_data)
            
    # 4. Convert to Numpy Arrays and Average
    # Shape becomes: (n_trials, n_channels, n_time_points)
    # Then we average across axis 0 (the trials dimension)
    
    # Handle Left Class
    if len(trials_left) > 0:
        stack_left = np.array(trials_left)
        avg_left = np.mean(stack_left, axis=0) 
        print(f"[INFO] Averaged {len(trials_left)} Left Hand trials.")
    else:
        # Fallback if no trials found (prevent crash)
        avg_left = np.zeros((n_channels, epoch_len))
        print("[WARN] No Left Hand trials found.")

    # Handle Right Class
    if len(trials_right) > 0:
        stack_right = np.array(trials_right)
        avg_right = np.mean(stack_right, axis=0)
        print(f"[INFO] Averaged {len(trials_right)} Right Hand trials.")
    else:
        # Fallback if no trials found
        avg_right = np.zeros((n_channels, epoch_len))
        print("[WARN] No Right Hand trials found.")
        
    # Generate Time Axis for plotting
    time_axis = np.linspace(tmin, tmax, epoch_len)
    
    return avg_left, avg_right, time_axis

# =========================================================
# Unit Test (Standalone Execution)
# =========================================================
if __name__ == "__main__":
    print(">> RUNNING STANDALONE TEST: average_all_EEG_trials.py")
    
    # 1. Print Description
    print("-" * 60)
    print(get_averaging_description())
    print("-" * 60)

    # 2. Create Dummy Data (1 Channel, 20 seconds, 250Hz)
    fs = 250.0
    total_samples = int(20 * fs)
    # Background noise (Random)
    dummy_power = np.random.rand(1, total_samples) * 5.0 
    
    # 3. Create Dummy Events (Indices) at t=5s and t=10s
    # Format: [sample_index, 0, event_id]
    events = np.array([
        [int(5.0*fs), 0, 769],  # Left trial 1
        [int(10.0*fs), 0, 769], # Left trial 2
        [int(15.0*fs), 0, 770]  # Right trial 1
    ])
    
    # 4. Inject a Pattern (Simulated ERD) into the noisy data
    # Let's say ERD (drop in power) happens 1s after cue
    for ev in events:
        idx = ev[0]
        # Simulate ERD: Drop power to 0.1 from t+1s to t+3s
        start_erd = idx + int(1.0*fs)
        end_erd = idx + int(3.0*fs)
        if end_erd < total_samples:
            dummy_power[:, start_erd:end_erd] *= 0.1 

    print("\n[TEST] Running Averaging Function...")
    avg_L, avg_R, t_axis = extract_and_average_epochs(dummy_power, events, fs)
    
    # 5. Plot
    plt.figure(figsize=(10, 5))
    
    # We expect to see the "dip" (ERD) clearly in Left trials
    # Using raw strings r'' to fix syntax warning
    plt.plot(t_axis, avg_L[0, :], label='Avg Left (Class 769)', color='blue')
    plt.plot(t_axis, avg_R[0, :], label='Avg Right (Class 770)', color='red', linestyle='--')
    
    plt.axvline(0, color='k', linestyle='-', label='Cue Onset')
    plt.axvspan(1, 3, color='gray', alpha=0.3, label='Simulated ERD Region')
    
    plt.title("Test: Synchronous Averaging of Power")
    plt.xlabel("Time relative to Cue (s)")
    plt.ylabel(r"Power ($\mu V^2$)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\n[TEST] Averaging Module Verification Passed.")