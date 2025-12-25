import numpy as np
import matplotlib.pyplot as plt

def get_envelope(scalogram):
    """
    Args:
        scalogram (array): 2D array of energy density
        
    Returns:
        array: 1D array of total energy over time
    """
    # axis 0 is frequency, axis 1 is time
    return np.sum(scalogram, axis=0)

def detect_bursts(energy_profile, fs, threshold_ratio=0.01):
    """
    Logic:
    1. Thresholding: Detect areas above 1% of peak energy
    2. Merging: Combine bursts separated by gaps < 30ms
    3. Discarding: Remove bursts shorter than 30ms
    
    Args:
        energy_profile (array): 1D energy array
        fs (float): Sampling frequency
        threshold_ratio (float): Threshold level relative to peak (default 0.01)
        
    Returns:
        list: List of tuples [(start_idx, end_idx), ...] representing activations
    """
    # 1. Determine Absolute Threshold
    peak_energy = np.max(energy_profile)
    threshold_val = threshold_ratio * peak_energy
    
    # 2. Binarization (0 or 1)
    is_active = (energy_profile > threshold_val).astype(int)
    
    # 3. Edge Detection (Rising & Falling Edges)
    # Pad with 0 at both ends to detect edges at boundaries safely
    diff = np.diff(np.pad(is_active, (1, 1), 'constant'))
    
    # Find indices: Rising (+1) is Onset, Falling (-1) is Offset
    # Correction -1 because of padding at the start
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    
    if len(onsets) == 0:
        return []

    # Combine into candidate list [start, end]
    candidates = []
    for on, off in zip(onsets, offsets):
        candidates.append([on, off])
        
    # Constraint Processing (Debouncing)
    # Constraint: 30 ms limit
    # samples = 0.03 * fs
    min_samples = int(0.03 * fs) 
    
    # Phase A: Merging (Join if gap is too short)
    if not candidates: return []
    
    merged = [candidates[0]]
    
    for i in range(1, len(candidates)):
        curr_start, curr_end = candidates[i]
        last_start, last_end = merged[-1]
        
        gap = curr_start - last_end
        
        if gap < min_samples:
            # Merge: Update the end of the last candidate
            merged[-1][1] = curr_end
        else:
            # Gap is sufficient, add as new activation
            merged.append([curr_start, curr_end])
            
    # Phase B: Discarding (Remove if duration is too short)
    final_activations = []
    for start, end in merged:
        duration = end - start
        if duration >= min_samples:
            final_activations.append((start, end))
            
    return final_activations

def apply_threshold(segments):
    if not segments: return []
    
    print(f"[-] Detecting Onset/Offset (Threshold 1%, Min Duration/Gap 30ms)...")
    
    processed_segments = []
    
    for seg in segments:
        new_seg = seg.copy()
        fs = seg['fs']
        
        # Process Gastrocnemius (GL)
        if 'cwt_gl' in seg:
            E_gl = seg['cwt_gl']['E']
            profile_gl = get_envelope(E_gl)
            
            activations_gl = detect_bursts(profile_gl, fs)
            
            # Store results (indices and absolute times)
            res_gl = []
            for start, end in activations_gl:
                # Ensure end index is within bounds
                end_idx_safe = min(end, len(seg['time']))
                # Convert index to time using the segment's time array
                # Use end_idx_safe - 1 to stay within array bounds for indexing
                end_time_idx = end_idx_safe - 1 if end_idx_safe > 0 else 0
                
                res_gl.append({
                    'start_idx': start,
                    'end_idx': end,
                    'start_t': seg['time'][start],
                    'end_t': seg['time'][end_time_idx]
                })
            new_seg['activations_gl'] = res_gl
            # Save profile for visualization
            new_seg['energy_profile_gl'] = profile_gl

        # Process Vastus Lateralis (VL)
        if 'cwt_vl' in seg:
            E_vl = seg['cwt_vl']['E']
            profile_vl = get_envelope(E_vl)
            
            activations_vl = detect_bursts(profile_vl, fs)
            
            res_vl = []
            for start, end in activations_vl:
                end_idx_safe = min(end, len(seg['time']))
                end_time_idx = end_idx_safe - 1 if end_idx_safe > 0 else 0
                
                res_vl.append({
                    'start_idx': start,
                    'end_idx': end,
                    'start_t': seg['time'][start],
                    'end_t': seg['time'][end_time_idx]
                })
            new_seg['activations_vl'] = res_vl
            new_seg['energy_profile_vl'] = profile_vl
            
        processed_segments.append(new_seg)
        
    return processed_segments

def plot_threshold_result(segments, cycle_idx=0, muscle='GL'):
    if not segments: return
    
    seg = segments[cycle_idx]
    
    if muscle == 'GL':
        profile = seg.get('energy_profile_gl')
        activations = seg.get('activations_gl')
        title = "Gastrocnemius (GL)"
    else:
        profile = seg.get('energy_profile_vl')
        activations = seg.get('activations_vl')
        title = "Vastus (VL)"
        
    if profile is None:
        print("Activation data not computed.")
        return
        
    time = seg['time']
    # Normalize time to start at 0 for this plot
    t_plot = time - time[0]
    
    plt.figure(figsize=(10, 5))
    
    # Plot Energy Profile
    plt.plot(t_plot, profile, color='black', linewidth=1, label='Integrated Energy (CWT)')
    
    # Plot Threshold Line
    th_val = 0.01 * np.max(profile)
    plt.axhline(th_val, color='orange', linestyle='--', label='Threshold (1%)')
    
    # Highlight Active Areas
    if activations:
        for i, act in enumerate(activations):
            t_start = act['start_t'] - time[0]
            t_end = act['end_t'] - time[0]
            
            plt.axvspan(t_start, t_end, color='green', alpha=0.3, 
                        label='Detected Burst' if i == 0 else "")
            
            # Mark Onset/Offset
            plt.axvline(t_start, color='green', linestyle='-', linewidth=0.5)
            plt.axvline(t_end, color='red', linestyle='-', linewidth=0.5)
            
    plt.title(f"Muscle Activation Detection - {title} (Cycle {seg['cycle_id']})")
    plt.xlabel("Time (s)")
    plt.ylabel("Total Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# <<<< Standalone Test Block >>>>
if __name__ == "__main__":
    # Simulate Energy Profile
    fs = 2000
    t = np.linspace(0, 1, fs)
    
    # Create fake energy signal
    # Burst 1: Valid (length 100ms)
    # Burst 2: Noise short (length 10ms) -> Should be Discarded
    # Burst 3: Close to Burst 1 (gap 10ms) -> Should be Merged with Burst 1
    
    energy = np.zeros_like(t)
    
    # Main Burst (0.2s - 0.3s)
    energy[400:600] = 100 
    
    # Noise close to main burst (0.31s - 0.35s) -> Gap 10ms < 30ms -> MERGE
    energy[620:700] = 80
    
    # Short standalone noise (0.8s - 0.81s) -> Duration 10ms < 30ms -> DISCARD
    energy[1600:1620] = 90
    
    # Normalize input structure
    dummy_seg = [{
        'cycle_id': 1,
        'fs': fs,
        'time': t,
        'cwt_gl': {'E': np.expand_dims(energy, axis=0)}, # Fake 2D matrix
    }]
    
    res = apply_threshold(dummy_seg)
    activations = res[0]['activations_gl']
    
    print(f"Found {len(activations)} activations.")
    for i, act in enumerate(activations):
        dur = act['end_t'] - act['start_t']
        print(f"Activation {i+1}: {act['start_t']:.3f}s - {act['end_t']:.3f}s (Duration: {dur*1000:.1f} ms)")
        
    plot_threshold_result(res, muscle='GL')