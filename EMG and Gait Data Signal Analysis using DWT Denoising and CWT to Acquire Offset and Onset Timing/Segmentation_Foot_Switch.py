import numpy as np
import matplotlib.pyplot as plt

def detect_events_manual(fs_signal, fs, min_dist_sec=0.4):
    """
    Manually detects Heel Strike (HS) and Toe Off (TO) events.
    
    Logic:
    1. Dynamic Thresholding (average of min and max).
    2. Binarization (High/Low state).
    3. Edge Detection (Rising/Falling).
    4. Manual Debouncing (Removing double detections within a short window).
    
    Returns: (heel_strikes, toe_offs)
    """
    # 1. Dynamic Thresholding
    t_min = np.min(fs_signal)
    t_max = np.max(fs_signal)
    threshold = (t_max + t_min) / 2
    
    # 2. Binarization
    # Assumption: High Value = Stance (Contact), Low Value = Swing
    # Cast boolean result to int (0 or 1)
    binary_signal = (np.abs(fs_signal) > np.abs(threshold)).astype(int)
    
    # 3. Edge Detection
    # np.diff calculates out[i] = a[i+1] - a[i]
    # +1 indicates 0 -> 1 transition (Rising Edge / Heel Strike)
    # -1 indicates 1 -> 0 transition (Falling Edge / Toe Off)
    edges = np.diff(binary_signal)
    
    raw_hs = np.where(edges == 1)[0]
    raw_to = np.where(edges == -1)[0]
    
    # 4. Manual Debouncing
    def debounce(indices, min_samples):
        if len(indices) == 0: return np.array([])
        clean = [indices[0]]
        for i in range(1, len(indices)):
            # Only accept if distance from last accepted event > min_samples
            if (indices[i] - clean[-1]) > min_samples:
                clean.append(indices[i])
        return np.array(clean)
    
    min_samples = int(min_dist_sec * fs)
    clean_hs = debounce(raw_hs, min_samples)
    clean_to = debounce(raw_to, min_samples)
            
    return clean_hs, clean_to

def segment_data(data_dict):
    """
    Segments data into individual cycles (Toe Off to Toe Off).
    Detects the Heel Strike occurring within the cycle.
    """
    if data_dict is None:
        print("[!] Data is empty.")
        return []
    
    fs_signal = data_dict['signal_fs']
    fs = data_dict['fs']
    
    print("[-] Performing segmentation (TO to TO) & Heel Strike detection...")
    
    # Detect Events
    heel_strikes, toe_offs = detect_events_manual(fs_signal, fs, min_dist_sec=0.5)
    
    segments = []
    
    # Cycle = Toe Off [i] -> Toe Off [i+1]
    # This captures Swing Phase -> Stance Phase
    for i in range(len(toe_offs) - 1):
        start_idx = toe_offs[i]
        end_idx = toe_offs[i+1]
        
        # Find Heel Strike within this cycle (Start < HS < End)
        # The Heel Strike marks the transition from Swing to Stance
        cycle_hs_candidates = heel_strikes[(heel_strikes > start_idx) & (heel_strikes < end_idx)]
        current_hs_idx = cycle_hs_candidates[0] if len(cycle_hs_candidates) > 0 else None
        
        # Validate cycle duration (e.g., 0.5s to 2.5s)
        duration = (end_idx - start_idx) / fs
        if 0.5 < duration < 2.5:
            # Slice the signals
            seg_dict = {
                'cycle_id': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'hs_idx': current_hs_idx, # Global Index of Heel Strike
                
                # Sliced signals
                'time': data_dict['time'][start_idx:end_idx],
                'fs_segment': data_dict['signal_fs'][start_idx:end_idx],
                'gl_segment': data_dict['signal_gl'][start_idx:end_idx],
                'vl_segment': data_dict['signal_vl'][start_idx:end_idx],
                'fs': fs
            }
            
            # Calculate relative Heel Strike time for plotting
            if current_hs_idx is not None:
                seg_dict['hs_idx_rel'] = current_hs_idx - start_idx
                seg_dict['hs_time_rel'] = (current_hs_idx - start_idx) / fs
            else:
                seg_dict['hs_idx_rel'] = None
                seg_dict['hs_time_rel'] = None
                
            segments.append(seg_dict)
            
    print(f"[-] Found {len(segments)} valid walking cycles.")
    return segments

def plot_segmentation(data_dict, segments):
    """
    Visualizes Full Segmentation (Standalone Test).
    Red Dashed = Start (Toe Off)
    Cyan Dotted = Internal Event (Heel Strike)
    """
    if not segments: return

    time = data_dict['time']
    fs_signal = data_dict['signal_fs']
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, fs_signal, color='white', alpha=0.7, label='Raw Foot Switch')
    
    for i, seg in enumerate(segments):
        # Plot Start (Toe Off)
        start_t = time[seg['start_idx']]
        plt.axvline(x=start_t, color='#ff5252', linestyle='--', alpha=0.9)
        
        # Plot Internal Heel Strike
        if seg['hs_idx'] is not None:
            hs_t = time[seg['hs_idx']]
            plt.axvline(x=hs_t, color='#18ffff', linestyle=':', linewidth=1.5, alpha=0.8)

    plt.title(f"Segmentation Result (TO-TO): {len(segments)} Cycles")
    ax = plt.gca()
    ax.set_facecolor('#1e1e2e')
    plt.show()

if __name__ == "__main__":
    try:
        import Load_and_Plot_Raw_Data as Loader
        data = Loader.load_raw_data("S01")
        if data:
            gait_cycles = segment_data(data)
            plot_segmentation(data, gait_cycles)
    except ImportError:
        print("Modul Load_and_Plot_Raw_Data not found.")