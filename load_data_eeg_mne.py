"""
load_data_eeg_mne.py (REVISED)

Purpose:
    - Load GDF files.
    - FORCE MAPPING of events to standard BCI codes (769, 770).
"""

import mne
import numpy as np
import os

# Suppress heavy MNE logging
mne.set_log_level('WARNING')

def load_eeg_data(filepath):
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None, None, None, None

    print(f"[INFO] Loading: {os.path.basename(filepath)}")

    try:
        # 1. Read Raw GDF
        raw = mne.io.read_raw_gdf(filepath, preload=True, verbose='ERROR')
    except Exception as e:
        print(f"[ERROR] MNE Load Failed: {e}")
        return None, None, None, None

    # 2. Rename Channels
    # Dataset 2b typically has EEG:C3, EEG:Cz, EEG:C4 as first 3 channels
    original_names = raw.ch_names
    rename_map = {}
    
    # Safety check: ensure we map correctly even if channel names vary slightly
    for i, name in enumerate(original_names):
        if i == 0: rename_map[name] = 'C3'
        elif i == 1: rename_map[name] = 'Cz'
        elif i == 2: rename_map[name] = 'C4'
        elif 'EOG' in name or i > 2: rename_map[name] = f'EOG:{i}'

    raw.rename_channels(rename_map)
    
    # 3. Apply Montage
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
    except:
        print("[WARN] Montage setting failed (Minor issue).")

    # 4. Pick EEG Channels Only
    raw.pick(['C3', 'Cz', 'C4'])

    # 5. Extract Events with FORCED MAPPING
    # GDF events are often stored as annotations like '769', '770', '32766'
    # We want to map these strings specifically to their integer values.
    
    # Define the map we WANT
    custom_mapping = {
        '769': 769,  # Left Hand
        '770': 770,  # Right Hand
        '1023': 1023 # Rejected
    }
    
    # Check what annotations exist in the file
    annotations = raw.annotations
    unique_desc = np.unique(annotations.description)
    print(f"[DEBUG] Found Annotation descriptions: {unique_desc}")
    
    # Create a valid map based on what exists in the file
    used_map = {}
    for desc in unique_desc:
        # Sometimes description is 'Event 769' or just '769'
        # Convert to string and strip potential whitespace
        desc_str = str(desc).strip()
        
        if desc_str in custom_mapping:
            used_map[desc_str] = custom_mapping[desc_str]
        else:
            # Handle hex codes if present (e.g., '0x0301')
            try:
                # Attempt to map standard decimal strings
                val = int(desc_str)
                if val in [769, 770, 1023]:
                    used_map[desc_str] = val
            except:
                pass

    if not used_map:
        print("[CRITICAL WARN] No standard BCI events (769/770) found in annotations!")
        # Fallback: Let MNE decide, but print warning
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)
        print(f"[DEBUG] Fallback MNE Map: {event_id_map}")
    else:
        # Use our enforced map
        print(f"[INFO] Enforcing Event Map: {used_map}")
        events, event_id_map = mne.events_from_annotations(raw, event_id=used_map, verbose=False)

    sfreq = raw.info['sfreq']
    print(f"[INFO] Sampling Rate: {sfreq} Hz")
    print(f"[INFO] Total Extracted Events: {len(events)}")
    
    return raw, events, event_id_map, sfreq