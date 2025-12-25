"""
load_data_eeg_mne.py (REVISED v3.1)

Purpose:
    - Load GDF files from BCI Competition IV Dataset 2b.
    - Designed to support Sequential Loading for Multiple Training Files.
    - Load ALL channels initially for inspection, then identify Motor Channels.
    - Map hexadecimal/decimal event codes to standard integers.
    - Detect Session Type (Screening vs. Smiley Feedback).
    - Provide detailed metadata for GUI logic.

Dependencies:
    - mne
    - numpy
    - os
    - sys
"""

import mne
import numpy as np
import os
import sys

# Suppress heavy MNE logging to keep the console clean and performance high
mne.set_log_level('WARNING')

def load_eeg_data(filepath):
    """
    Loads a single EEG GDF file, extracts events, and prepares metadata.
    
    This function is designed to be called iteratively by the GUI when 
    loading multiple training files (e.g., B0101T + B0102T).

    Args:
        filepath (str): Full path to the .gdf file.

    Returns:
        raw (mne.io.Raw): The complete MNE raw object (All channels).
        events (np.array): The extracted events matrix [sample, 0, id].
        event_id_map (dict): Mapping of event names to IDs.
        fs (float): Sampling frequency.
        session_info (dict): Comprehensive metadata regarding the session.
    """
    # 1. Validation
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None, None, None, None, None

    filename = os.path.basename(filepath)
    print(f"[INFO] Loading File: {filename}")

    # 2. Read Raw GDF (Preload required for filtering)
    try:
        raw = mne.io.read_raw_gdf(filepath, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"[ERROR] MNE Load Failed for {filename}: {e}")
        return None, None, None, None, None

    # 3. Rename Channels (Standardize BCI IV 2b naming)
    # The dataset typically contains 3 EEG channels (Indices 0,1,2) and 3 EOG channels.
    original_names = raw.ch_names
    rename_map = {}
    
    # Heuristic mapping based on channel index positions described in the dataset PDF
    # EEG 1-3 are C3, Cz, C4. EOG 4-6 are EOGs.
    if len(original_names) >= 3:
        rename_map[original_names[0]] = 'C3'
        rename_map[original_names[1]] = 'Cz'
        rename_map[original_names[2]] = 'C4'
    
    if len(original_names) >= 6:
        rename_map[original_names[3]] = 'EOG:01'
        rename_map[original_names[4]] = 'EOG:02'
        rename_map[original_names[5]] = 'EOG:03'

    raw.rename_channels(rename_map)
    
    # 4. Set Montage (Standard 10-20 System)
    # This associates 3D coordinates with the channel names for topographic plots.
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"[WARN] Montage setting failed (Non-critical): {e}")

    # 5. Extract Events with Comprehensive Mapping
    # Standard codes from BCI Competition IV 2b Description (Table 2)
    standard_bci_map = {
        # Cues / Trials
        '768': 768,   # Start of a trial
        '769': 769,   # Cue onset LEFT (Class 1)
        '770': 770,   # Cue onset RIGHT (Class 2)
        '781': 781,   # BCI Feedback (Continuous)
        '783': 783,   # Cue unknown
        
        # Artifacts / States
        '276': 276,   # Idling EEG (Eyes Open)
        '277': 277,   # Idling EEG (Eyes Closed)
        '1023': 1023, # Rejected trial (Artifact)
        '32766': 32766, # Start of a new run
        
        # Hexadecimal Variants (MNE sometimes reads annotations as hex strings)
        '0x0300': 768,
        '0x0301': 769,
        '0x0302': 770,
        '0x030D': 781,
        '0x03FF': 1023,
        '0x7FFE': 32766
    }
    
    # Scan file annotations
    annotations = raw.annotations
    unique_desc = np.unique(annotations.description)
    
    # Build a specific mapping for this file
    used_map = {}
    for desc in unique_desc:
        desc_str = str(desc).strip()
        
        # Check strict string match
        if desc_str in standard_bci_map:
            used_map[desc_str] = standard_bci_map[desc_str]
        else:
            # Try integer conversion fallback
            try:
                val = int(desc_str)
                if val in standard_bci_map.values():
                    used_map[desc_str] = val
            except ValueError:
                continue

    if not used_map:
        print(f"[CRITICAL WARNING] No standard BCI events found in {filename}.")
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    else:
        print(f"[INFO] Applied Event Map for {filename}: {used_map}")
        events, event_id_map = mne.events_from_annotations(raw, event_id=used_map, verbose=False)

    # 6. Determine Session Type (Screening vs Feedback)
    # Feedback sessions (03T, 04E, 05E) contain event 781. Screening (01T, 02T) do not.
    has_feedback = 781 in event_id_map.values()
    session_type = "Smiley Feedback" if has_feedback else "Screening (No Feedback)"
    
    # 7. Channel Validation
    # Ensure the critical motor channels exist
    available_channels = raw.ch_names
    motor_channels = ['C3', 'Cz', 'C4']
    missing_channels = [ch for ch in motor_channels if ch not in available_channels]
    
    if missing_channels:
        print(f"[CRITICAL ERROR] Missing required motor channels in {filename}: {missing_channels}")
        # We return None to signal the GUI to abort loading this specific file
        return None, None, None, None, None

    # 8. Build Metadata Dictionary
    duration_sec = raw.times[-1]
    
    # Count specific trials
    count_left = np.sum(events[:, 2] == 769)
    count_right = np.sum(events[:, 2] == 770)
    count_artifact = np.sum(events[:, 2] == 1023)
    
    session_info = {
        "filename": filename,
        "type": session_type,
        "has_feedback": has_feedback,
        "total_events": len(events),
        "count_left": int(count_left),
        "count_right": int(count_right),
        "count_artifact": int(count_artifact),
        "sampling_rate": raw.info['sfreq'],
        "duration_sec": duration_sec,
        "duration_str": f"{int(duration_sec // 60)} min {int(duration_sec % 60)} sec",
        "all_channels": available_channels,
        "motor_channels": motor_channels
    }

    print(f"[INFO] Loaded {filename} | Type: {session_type} | Trials: {count_left+count_right}")
    
    return raw, events, event_id_map, raw.info['sfreq'], session_info

# ==========================================
# Standalone Test Execution
# ==========================================
if __name__ == "__main__":
    print(">> STANDALONE TEST MODE: load_data_eeg_mne.py")
    print(">> This test simulates loading multiple datasets (Training Set).\n")

    # List to store loaded data to verify memory handling
    loaded_datasets = []

    while True:
        # 1. Input File
        target_file = input("Enter .gdf filename (e.g., B0101T.gdf) or 'q' to quit: ").strip()
        target_file = target_file.replace('"', '').replace("'", "") # Clean quotes

        if target_file.lower() == 'q':
            break

        if not target_file:
            continue

        # 2. Run Load Function
        print(f"\n[TEST] Loading: {target_file} ...")
        raw, events, event_map, fs, meta = load_eeg_data(target_file)

        # 3. Report Results
        if raw is not None:
            print("\n" + "="*50)
            print(f"       DATASET LOADED: {meta['filename']}")
            print("="*50)
            print(f"Session Type  : {meta['type']}")
            print(f"Channels      : {meta['all_channels']}")
            print(f"Motor Chans   : {meta['motor_channels']}")
            print("-" * 50)
            print(f"TRIALS: Left={meta['count_left']}, Right={meta['count_right']}, Artifacts={meta['count_artifact']}")
            print("="*50 + "\n")
            
            # Store for simulation
            loaded_datasets.append(raw)
        else:
            print("\n[TEST] Load Failed.")

    # 4. Summary of Multi-File Load
    if loaded_datasets:
        print("\n>> MULTI-FILE LOAD SUMMARY:")
        print(f"Total Datasets Loaded: {len(loaded_datasets)}")
        print("Ready for Merging (Concatenation) in GUI Pipeline.")
    else:
        print("No datasets loaded.")