"""
load_data_eeg_mne.py (REVISED v2.1)

Purpose:
    - Load GDF files from BCI Competition IV Dataset 2b.
    - Rename channels to standard 10-20 system (C3, Cz, C4).
    - Map hexadecimal/decimal event codes to integers based on Dataset Description.
    - Detect Session Type (Screening vs. Smiley Feedback).
    - Includes Standalone Test Block for debugging.

Dependencies:
    - mne
    - numpy
    - os
"""

import mne
import numpy as np
import os
import sys

# Suppress heavy MNE logging to keep the console clean
mne.set_log_level('WARNING')

def load_eeg_data(filepath):
    """
    Loads EEG data and extracts standard BCI events.

    Args:
        filepath (str): Path to the .gdf file.

    Returns:
        raw (mne.io.Raw): The raw MNE object (containing C3, Cz, C4).
        events (np.array): The extracted events matrix.
        event_id_map (dict): Mapping of event names to IDs.
        fs (float): Sampling frequency.
        session_info (dict): Metadata regarding the session type (Screening/Feedback).
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None, None, None, None, None

    print(f"[INFO] Loading File: {os.path.basename(filepath)}")

    # 1. Read Raw GDF
    # preload=True is required for subsequent filtering and cropping in memory.
    try:
        raw = mne.io.read_raw_gdf(filepath, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"[ERROR] MNE Load Failed: {e}")
        return None, None, None, None, None

    # 2. Rename Channels (Standardize BCI IV 2b naming)
    # The dataset typically contains 3 EEG channels (Indices 0,1,2) and 3 EOG channels.
    # We explicitly map them to ensure consistency across different subjects.
    original_names = raw.ch_names
    rename_map = {}
    
    # Check if we have at least 3 channels
    if len(original_names) >= 3:
        rename_map[original_names[0]] = 'C3'
        rename_map[original_names[1]] = 'Cz'
        rename_map[original_names[2]] = 'C4'
    
    # Map EOG channels if they exist (Indices 3, 4, 5)
    if len(original_names) >= 6:
        rename_map[original_names[3]] = 'EOG:01'
        rename_map[original_names[4]] = 'EOG:02'
        rename_map[original_names[5]] = 'EOG:03'

    raw.rename_channels(rename_map)
    
    # 3. Set Montage (Standard 10-20 System)
    # This provides 3D coordinates for topographic plotting (CSP/scalp maps).
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"[WARN] Montage setting failed: {e}")

    # 4. Pick EEG Channels Only for Processing
    # We isolate the brain signals. EOG is typically used for artifact removal, 
    # but for this specific assignment scope, we focus on the 3 motor leads.
    raw.pick(['C3', 'Cz', 'C4'])

    # 5. Extract Events with Comprehensive Mapping
    # Based on Table 2 of the BCI Competition 2008 Description.
    # We must map potential Hex strings or alternative descriptions to Integers.
    
    standard_bci_map = {
        # Trigger / Cue Codes
        '768': 768,   # Start of a trial
        '769': 769,   # Cue onset LEFT (Class 1)
        '770': 770,   # Cue onset RIGHT (Class 2)
        '781': 781,   # BCI Feedback (Continuous)
        '783': 783,   # Cue unknown
        
        # Artifact / State Codes
        '276': 276,   # Idling EEG (Eyes Open)
        '277': 277,   # Idling EEG (Eyes Closed)
        '1023': 1023, # Rejected trial (Artifact)
        '32766': 32766, # Start of a new run
        
        # Hexadecimal Variants (Sometimes MNE reads them as hex strings)
        '0x0300': 768,
        '0x0301': 769,
        '0x0302': 770,
        '0x030D': 781,
        '0x03FF': 1023,
        '0x7FFE': 32766
    }
    
    # Scan annotations present in the file
    annotations = raw.annotations
    unique_desc = np.unique(annotations.description)
    # print(f"[DEBUG] Raw Annotation Descriptions found: {unique_desc}")
    
    # Build a specific mapping for this file based on what is available
    used_map = {}
    for desc in unique_desc:
        desc_str = str(desc).strip()
        
        # 1. Check strict string match
        if desc_str in standard_bci_map:
            used_map[desc_str] = standard_bci_map[desc_str]
        else:
            # 2. Try integer conversion fallback
            try:
                val = int(desc_str)
                if val in standard_bci_map.values():
                    used_map[desc_str] = val
            except ValueError:
                continue

    if not used_map:
        print("[CRITICAL WARNING] No standard BCI events found. Defaulting to MNE auto-mapping.")
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)
    else:
        print(f"[INFO] Applied Event Map: {used_map}")
        events, event_id_map = mne.events_from_annotations(raw, event_id=used_map, verbose=False)

    # 6. Determine Session Type (Screening vs Feedback)
    # Logic: Feedback sessions (03T, 04E, 05E) contain event 781 (BCI Feedback).
    # Screening sessions (01T, 02T) do not.
    
    has_feedback = 781 in event_id_map.values()
    
    session_type = "Smiley Feedback" if has_feedback else "Screening (No Feedback)"
    
    session_info = {
        "filename": os.path.basename(filepath),
        "type": session_type,
        "has_feedback": has_feedback,
        "total_trials": len(events),
        "sampling_rate": raw.info['sfreq']
    }

    print(f"[INFO] Detected Session Type: {session_type}")
    print(f"[INFO] Sampling Rate: {raw.info['sfreq']} Hz")
    
    return raw, events, event_id_map, raw.info['sfreq'], session_info

# ==========================================
# Standalone Test Execution
# ==========================================
if __name__ == "__main__":
    print(">> STANDALONE TEST MODE: load_data_eeg_mne.py")
    print(">> This mode allows you to test GDF loading without the GUI.\n")

    # 1. Input File
    target_file = input("Please enter the filename/path of a .gdf file (e.g., B0401T.gdf): ").strip()

    # Remove quotes if user copied path as string
    target_file = target_file.replace('"', '').replace("'", "")

    if not target_file:
        print("[TEST] No file provided. Exiting.")
        sys.exit()

    # 2. Run the Function
    print(f"\n[TEST] Attempting to load: {target_file} ...")
    raw, events, event_map, fs, session_meta = load_eeg_data(target_file)

    # 3. Report Results
    if raw is not None:
        print("\n" + "="*40)
        print("       DATA LOAD SUCCESSFUL")
        print("="*40)
        print(f"1. Filename      : {session_meta['filename']}")
        print(f"2. Session Type  : {session_meta['type']}")
        print(f"3. Sampling Rate : {fs} Hz")
        print(f"4. Total Events  : {session_meta['total_trials']}")
        print(f"5. Channels      : {raw.ch_names}")
        print(f"6. Duration      : {raw.times[-1]:.2f} seconds")
        print("-" * 40)
        print("Event Mapping Found:")
        for desc, code in event_map.items():
            print(f"  - '{desc}' -> ID {code}")
        print("="*40 + "\n")
        
        # Optional: Print first 5 events
        print("[TEST] First 5 Events (Sample Index, 0, Event ID):")
        print(events[:5])
    else:
        print("\n[TEST] Loading Failed. Please check the error messages above.")