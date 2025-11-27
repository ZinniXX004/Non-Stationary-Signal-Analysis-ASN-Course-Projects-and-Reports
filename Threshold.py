import numpy as np
import matplotlib.pyplot as plt

def get_envelope(scalogram):
    """
    Mengubah Scalogram 2D (Freq x Time) menjadi 1D Energy Profile.
    Metode: Menjumlahkan energi di seluruh pita frekuensi (Marginal Integral).
    Ini merepresentasikan total instantaneous energy dari otot.
    """
    # axis 0 adalah frekuensi, axis 1 adalah waktu
    return np.sum(scalogram, axis=0)

def detect_bursts(energy_profile, fs, threshold_ratio=0.01):
    """
    Algoritma Deteksi Onset/Offset dengan Double Constraint (Merge & Discard).
    """
    # 1. Tentukan Threshold Absolut
    peak_energy = np.max(energy_profile)
    threshold_val = threshold_ratio * peak_energy
    
    # 2. Binarisasi (0 atau 1)
    is_active = (energy_profile > threshold_val).astype(int)
    
    # 3. Deteksi Tepi (Rising & Falling Edges)
    # Pad dengan 0 di kedua ujung agar deteksi tepi aman di batas data
    diff = np.diff(np.pad(is_active, (1, 1), 'constant'))
    
    # Indeks onset (perubahan 0 ke 1) dan offset (1 ke 0)
    # Koreksi indeks -1 karena padding di awal
    onsets = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    
    if len(onsets) == 0:
        return []

    # Gabungkan menjadi list of [start, end]
    candidates = []
    for on, off in zip(onsets, offsets):
        candidates.append([on, off])
        
    # --- Constraint Processing (Debouncing) ---
    # 30 ms dalam jumlah sampel
    # ms_limit = 0.03 detik
    min_samples = int(0.03 * fs) 
    
    # Tahap A: Merging (Gabung jika jeda terlalu singkat)
    if not candidates: return []
    
    merged = [candidates[0]]
    
    for i in range(1, len(candidates)):
        curr_start, curr_end = candidates[i]
        last_start, last_end = merged[-1]
        
        gap = curr_start - last_end
        
        if gap < min_samples:
            # Gabung: Update end dari kandidat terakhir
            merged[-1][1] = curr_end
        else:
            # Jeda cukup lama, tambahkan sebagai aktivasi baru
            merged.append([curr_start, curr_end])
            
    # Tahap B: Discarding (Buang jika durasi terlalu singkat)
    final_activations = []
    for start, end in merged:
        duration = end - start
        if duration >= min_samples:
            final_activations.append((start, end))
            
    return final_activations

def apply_threshold(segments):
    """
    Wrapper utama untuk memproses semua segmen.
    """
    if not segments: return []
    
    print(f"[-] Menentukan Onset/Offset (Threshold 1%, Min Durasi/Gap 30ms)...")
    
    processed_segments = []
    
    for seg in segments:
        new_seg = seg.copy()
        fs = seg['fs']
        
        # --- Proses GL ---
        if 'cwt_gl' in seg:
            E_gl = seg['cwt_gl']['E']
            profile_gl = get_envelope(E_gl)
            
            activations_gl = detect_bursts(profile_gl, fs)
            
            # Simpan hasil dalam detik dan indeks
            res_gl = []
            for start, end in activations_gl:
                res_gl.append({
                    'start_idx': start,
                    'end_idx': end,
                    'start_t': seg['time'][start],
                    'end_t': seg['time'][end-1] if end < len(seg['time']) else seg['time'][-1]
                })
            new_seg['activations_gl'] = res_gl
            # Simpan juga profile 1D untuk visualisasi
            new_seg['energy_profile_gl'] = profile_gl

        # --- Proses VL ---
        if 'cwt_vl' in seg:
            E_vl = seg['cwt_vl']['E']
            profile_vl = get_envelope(E_vl)
            
            activations_vl = detect_bursts(profile_vl, fs)
            
            res_vl = []
            for start, end in activations_vl:
                res_vl.append({
                    'start_idx': start,
                    'end_idx': end,
                    'start_t': seg['time'][start],
                    'end_t': seg['time'][end-1] if end < len(seg['time']) else seg['time'][-1]
                })
            new_seg['activations_vl'] = res_vl
            new_seg['energy_profile_vl'] = profile_vl
            
        processed_segments.append(new_seg)
        
    return processed_segments

def plot_threshold_result(segments, cycle_idx=0, muscle='GL'):
    """
    Visualisasi Profil Energi 1D beserta area yang terdeteksi aktif.
    """
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
        print("Data aktivasi belum dihitung.")
        return
        
    time = seg['time']
    # Normalisasi waktu mulai dari 0
    t_plot = time - time[0]
    
    plt.figure(figsize=(10, 5))
    
    # Plot Profil Energi
    plt.plot(t_plot, profile, color='black', linewidth=1, label='Integrated Energy (CWT)')
    
    # Plot Threshold Line
    th_val = 0.01 * np.max(profile)
    plt.axhline(th_val, color='orange', linestyle='--', label='Threshold (1%)')
    
    # Highlight Area Aktif
    if activations:
        for i, act in enumerate(activations):
            t_start = act['start_t'] - time[0]
            t_end = act['end_t'] - time[0]
            
            plt.axvspan(t_start, t_end, color='green', alpha=0.3, 
                        label='Detected Burst' if i == 0 else "")
            
            # Tandai Onset/Offset
            plt.axvline(t_start, color='green', linestyle='-', linewidth=0.5)
            plt.axvline(t_end, color='red', linestyle='-', linewidth=0.5)
            
    plt.title(f"Muscle Activation Detection - {title} (Cycle {seg['cycle_id']})")
    plt.xlabel("Time (s)")
    plt.ylabel("Total Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- Blok Testing Mandiri ---
if __name__ == "__main__":
    # Simulasi Energi Profile
    fs = 2000
    t = np.linspace(0, 1, fs)
    
    # Buat sinyal energi palsu
    # Burst 1: Valid (panjang 100ms)
    # Burst 2: Noise pendek (panjang 10ms) -> Harus dibuang
    # Burst 3: Dekat dengan Burst 1 (jarak 10ms) -> Harus digabung ke Burst 1
    
    energy = np.zeros_like(t)
    
    # Burst Utama (0.2s - 0.3s)
    energy[400:600] = 100 
    
    # Noise (0.31s - 0.35s) -> Jarak 0.01s (10ms) dari burst utama.
    # Karena gap 10ms < 30ms, ini harusnya MERGE dengan burst utama.
    energy[620:700] = 80
    
    # Noise Terpisah Jauh tapi Pendek (0.8s - 0.81s) -> Durasi 10ms
    # Karena durasi 10ms < 30ms, ini harusnya DISCARD.
    energy[1600:1620] = 90
    
    # Normalisasi input biar sesuai struktur
    dummy_seg = [{
        'cycle_id': 1,
        'fs': fs,
        'time': t,
        'cwt_gl': {'E': np.expand_dims(energy, axis=0)}, # Fake 2D matrix
    }]
    
    res = apply_threshold(dummy_seg)
    activations = res[0]['activations_gl']
    
    print(f"Ditemukan {len(activations)} aktivasi.")
    for i, act in enumerate(activations):
        dur = act['end_t'] - act['start_t']
        print(f"Aktivasi {i+1}: {act['start_t']:.3f}s - {act['end_t']:.3f}s (Durasi: {dur*1000:.1f} ms)")
        
    plot_threshold_result(res, muscle='GL')
    
    # Harapan Output:
    # Hanya ada 1 aktivasi besar gabungan (0.2s sampai 0.35s).
    # Burst pendek di 0.8s hilang.