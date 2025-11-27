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
    # Calculate the midpoint between the signal's minimum and maximum
    t_min = np.min(fs_signal)
    t_max = np.max(fs_signal)
    threshold = (t_max + t_min) / 2
    
    # 2. Binarization
    # Assumption: High Value = Stance (Contact), Low Value = Swing
    # We cast boolean result to int (0 or 1)
    binary_signal = (np.abs(fs_signal) > np.abs(threshold)).astype(int)
    
    # 3. Edge Detection
    # np.diff calculates out[i] = a[i+1] - a[i]
    # +1 indicates 0 -> 1 transition (Rising Edge / Heel Strike)
    # -1 indicates 1 -> 0 transition (Falling Edge / Toe Off)
    edges = np.diff(binary_signal)
    
    raw_hs = np.where(edges == 1)[0]
    raw_to = np.where(edges == -1)[0]
    
    # 4. Manual Debouncing
    # Filters out events that occur too close to the previous one
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
    Segments data into individual cycles (Heel Strike to Heel Strike).
    """
    if data_dict is None:
        print("[!] Data is empty.")
        return []
    
    fs_signal = data_dict['signal_fs']
    fs = data_dict['fs']
    
    print("[-] Performing segmentation (HS to HS) & Toe Off detection...")
    
    # Detect Events
    heel_strikes, toe_offs = detect_events_manual(fs_signal, fs, min_dist_sec=0.5)
    
    segments = []
    
    # Iterate through heel strikes to define cycles
    for i in range(len(heel_strikes) - 1):
        start_idx = heel_strikes[i]
        end_idx = heel_strikes[i+1]
        
        # Find Toe Off within this specific cycle (Start < TO < End)
        # Toe Off marks the end of the Stance phase
        cycle_to_candidates = toe_offs[(toe_offs > start_idx) & (toe_offs < end_idx)]
        current_to_idx = cycle_to_candidates[0] if len(cycle_to_candidates) > 0 else None
        
        # Validate cycle duration (e.g., 0.5s to 2.5s for normal walking)
        duration = (end_idx - start_idx) / fs
        if 0.5 < duration < 2.5:
            # Slice the signals (Numpy slicing preserves values)
            seg_dict = {
                'cycle_id': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'to_idx': current_to_idx, # Global Index (Important for absolute plots)
                
                # Sliced signals
                'time': data_dict['time'][start_idx:end_idx],
                'fs_segment': data_dict['signal_fs'][start_idx:end_idx],
                'gl_segment': data_dict['signal_gl'][start_idx:end_idx],
                'vl_segment': data_dict['signal_vl'][start_idx:end_idx],
                'fs': fs
            }
            
            # Calculate relative Toe Off time (seconds from start of cycle)
            if current_to_idx is not None:
                seg_dict['to_idx_rel'] = current_to_idx - start_idx
                seg_dict['to_time_rel'] = (current_to_idx - start_idx) / fs
            else:
                seg_dict['to_idx_rel'] = None
                seg_dict['to_time_rel'] = None
                
            segments.append(seg_dict)
            
    print(f"[-] Found {len(segments)} valid walking cycles.")
    return segments

def plot_segmentation(data_dict, segments):
    """
    Visualizes Full Segmentation (For standalone testing).
    """
    if not segments: return

    time = data_dict['time']
    fs_signal = data_dict['signal_fs']
    
    plt.figure(figsize=(12, 6))
    plt.plot(time, fs_signal, color='white', alpha=0.7, label='Raw Foot Switch')
    
    for i, seg in enumerate(segments):
        start_t = time[seg['start_idx']]
        plt.axvline(x=start_t, color='#ff5252', linestyle='--', alpha=0.9)
        
        if seg['to_idx'] is not None:
            to_t = time[seg['to_idx']]
            plt.axvline(x=to_t, color='#18ffff', linestyle=':', linewidth=1.5, alpha=0.8)

    plt.title(f"Segmentation Result: {len(segments)} Cycles")
    ax = plt.gca()
    ax.set_facecolor('#1e1e2e')
    plt.show()

# --- Block for standalone testing ---
if __name__ == "__main__":
    try:
        import Load_and_Plot_Raw_Data as Loader
        # Change this string to match your local file name if testing directly
        data = Loader.load_raw_data("S01")
        if data:
            gait_cycles = segment_data(data)
            plot_segmentation(data, gait_cycles)
    except ImportError:
        print("Modul Load_and_Plot_Raw_Data not found.")