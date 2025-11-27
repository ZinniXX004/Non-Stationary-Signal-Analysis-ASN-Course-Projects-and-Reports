import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ManualCWT:
    """
    Implementasi Continuous Wavelet Transform (CWT) Manual.
    Mendukung dua jenis Mother Wavelet:
    1. Morlet (Analitik)
    2. Daubechies 4 (Pendekatan Numerik via Cascade Algorithm)
    """
    def __init__(self, fs, f_min=20, f_max=450, num_scales=64, wavelet_type='morlet'):
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        self.num_scales = num_scales
        self.wavelet_type = wavelet_type.lower()
        
        # Generate scales berdasarkan frekuensi target
        # Rumus: Scale = (Center_Freq * Fs) / Target_Freq
        if self.wavelet_type == 'morlet':
            self.fc = 0.8125  # Center freq standar Morlet
        else: # db4
            self.fc = 0.7143  # Center freq aproksimasi db4
            
        # Buat array frekuensi secara logaritmik agar resolusi bagus di frekuensi rendah
        self.freqs = np.logspace(np.log10(f_min), np.log10(f_max), num_scales)
        self.scales = (self.fc * fs) / self.freqs
        
        # Cache untuk bentuk dasar db4 (agar tidak hitung ulang terus)
        self.db4_prototype = None
        if self.wavelet_type == 'db4':
            self.db4_prototype = self._generate_db4_prototype()

    def _morlet_wavelet(self, scale):
        """
        Membuat Complex Morlet Wavelet pada skala tertentu.
        Rumus: psi(t) = pi^(-0.25) * exp(i*w*t) * exp(-t^2/2)
        """
        # Tentukan rentang waktu efektif (-6 sigma s.d +6 sigma)
        M = int(scale * 10) 
        if M % 2 == 0: M += 1 # Ganjilkan
        t = np.arange(-(M//2), (M//2)+1) / scale
        
        # Konstanta Normalisasi
        C = np.pi**(-0.25)
        
        # Complex Morlet (w0 = 6 biasanya)
        w0 = 6 
        psi = C * np.exp(1j * w0 * t) * np.exp(-0.5 * t**2)
        
        # Normalisasi energi agar 1/sqrt(a) terpenuhi
        return psi * (1 / np.sqrt(scale))

    def _generate_db4_prototype(self, iterations=6):
        """
        Membangkitkan bentuk gelombang db4 menggunakan Cascade Algorithm.
        """
        # Koefisien Rekonstruksi (Inverse Filter) untuk db4
        h0 = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
        h1 = (3 + np.sqrt(3)) / (4 * np.sqrt(2))
        h2 = (3 - np.sqrt(3)) / (4 * np.sqrt(2))
        h3 = (1 - np.sqrt(3)) / (4 * np.sqrt(2))
        
        # Low Pass Reconstruction (Scaling Filter)
        L_R = np.array([h3, h2, h1, h0, h0, h1, h2, h3]) 
        # High Pass Reconstruction (Wavelet Filter)
        H_R = np.array([h3, -h2, h1, -h0, h0, -h1, h2, -h3]) 
        
        # Algoritma Kaskade (Iterative Upsampling + Convolution)
        psi = H_R 
        
        for _ in range(iterations):
            # 1. Upsample (sisipkan nol)
            psi_up = np.zeros(len(psi) * 2)
            psi_up[::2] = psi
            
            # 2. Konvolusi dengan Scaling Filter (Low Pass)
            psi = np.convolve(psi_up, L_R)
            
        return psi

    def _db4_wavelet(self, scale):
        """
        Membuat db4 wavelet pada skala tertentu dengan cara Resampling (Interpolasi).
        """
        prototype = self.db4_prototype
        len_proto = len(prototype)
        
        # Target panjang window berdasarkan skala
        target_len = int(scale * 7)
        if target_len < 8: target_len = 8 # Minimal length
        
        # Interpolasi Linear Manual (Resampling)
        x_original = np.linspace(0, 1, len_proto)
        x_target = np.linspace(0, 1, target_len)
        
        psi_resampled = np.interp(x_target, x_original, prototype)
        
        # Normalisasi Energi
        return psi_resampled * (1 / np.sqrt(scale))

    def compute(self, signal):
        """
        Menghitung Koefisien CWT dan Scalogram.
        Returns:
            time: array waktu
            freqs: array frekuensi
            scalogram: Matriks Energi (Frekuensi x Waktu)
        """
        n = len(signal)
        cwt_matrix = np.zeros((self.num_scales, n), dtype=complex)
        
        # Loop untuk setiap skala (frekuensi)
        for i, scale in enumerate(self.scales):
            # 1. Generate Mother Wavelet pada skala ini
            if self.wavelet_type == 'morlet':
                psi = self._morlet_wavelet(scale)
            else:
                psi = self._db4_wavelet(scale)
            
            # 2. Konvolusi (Sinyal * Wavelet)
            # mode='same' agar panjang output sama dengan input
            coeffs = np.convolve(signal, psi, mode='same')
            
            cwt_matrix[i, :] = coeffs
            
        # Hitung Scalogram (Densitas Energi)
        energy_density = np.abs(cwt_matrix)**2
        
        return energy_density, self.freqs

def compute_cwt_for_segments(segments, wavelet_type='morlet'):
    """Wrapper untuk list segments."""
    if not segments: return []
    
    # Ambil Fs dari data pertama
    fs = segments[0]['fs']
    
    cwt_solver = ManualCWT(fs, f_min=20, f_max=450, num_scales=50, wavelet_type=wavelet_type)
    
    print(f"[-] Menghitung CWT Manual (Wavelet: {wavelet_type}, Scales: 50)...")
    
    processed_segments = []
    for seg in segments:
        new_seg = seg.copy()
        
        # Prioritas Sinyal: Denoised -> Filtered -> Raw
        gl_sig = seg.get('gl_denoised', seg.get('gl_filtered', seg['gl_segment']))
        vl_sig = seg.get('vl_denoised', seg.get('vl_filtered', seg['vl_segment']))
        
        # Hitung CWT GL
        E_gl, f_gl = cwt_solver.compute(gl_sig)
        new_seg['cwt_gl'] = {'f': f_gl, 't': seg['time'], 'E': E_gl, 'type': wavelet_type}
        
        # Hitung CWT VL
        E_vl, f_vl = cwt_solver.compute(vl_sig)
        new_seg['cwt_vl'] = {'f': f_vl, 't': seg['time'], 'E': E_vl, 'type': wavelet_type}
        
        processed_segments.append(new_seg)
        
    return processed_segments

def plot_cwt(segments, cycle_idx=0, muscle='GL', use_db=False, mode='2D'):
    """
    Visualisasi CWT Scalogram.
    Modifikasi: Sumbu X menggunakan % Gait Cycle (0-100).
    """
    if not segments: return
    
    seg = segments[cycle_idx]
    key = 'cwt_gl' if muscle == 'GL' else 'cwt_vl'
    
    if key not in seg:
        print("Data CWT belum dihitung.")
        return

    data = seg[key]
    f = data['f']
    t = data['t']
    E = data['E'] # Energy Density |W|^2
    
    # --- MODIFIKASI UTAMA: Normalisasi Waktu ke % Gait Cycle ---
    # Normalisasi t agar 0% = awal siklus, 100% = akhir siklus
    if len(t) > 1:
        t_norm = (t - t[0]) / (t[-1] - t[0]) * 100
    else:
        t_norm = t
    
    # Konversi Unit
    if use_db:
        # 10 log10 karena E sudah berupa Power (|W|^2)
        # Tambahkan epsilon untuk menghindari log(0)
        Z_plot = 10 * np.log10(E + 1e-10)
        unit_label = "Power Spectral Density (dB)"
    else:
        Z_plot = E
        unit_label = "Energy Density (Linear)"

    fig = plt.figure(figsize=(10, 6))
    
    if mode == '2D':
        ax = fig.add_subplot(111)
        # Contourf untuk hasil CWT
        levels = 50
        cf = ax.contourf(t_norm, f, Z_plot, levels=levels, cmap='jet')
        cbar = fig.colorbar(cf, ax=ax, label=unit_label)
        
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('% Gait Cycle') # Updated Label
        ax.set_title(f"CWT Scalogram ({data['type']}) - {muscle} Cycle {seg['cycle_id']}")
        ax.set_xlim(0, 100) # Pastikan limit 0-100%
        
    elif mode == '3D':
        ax = fig.add_subplot(111, projection='3d')
        
        # Meshgrid diperlukan untuk plot 3D
        # Gunakan t_norm untuk sumbu X
        T_grid, F_grid = np.meshgrid(t_norm, f)
        
        # Surface plot
        surf = ax.plot_surface(T_grid, F_grid, Z_plot, cmap='viridis', 
                               edgecolor='none', rstride=1, cstride=1, alpha=0.9)
        
        fig.colorbar(surf, ax=ax, label=unit_label, pad=0.1)
        
        ax.set_xlabel('% Gait Cycle') # Updated Label
        ax.set_ylabel('Frequency (Hz)')
        ax.set_zlabel(unit_label)
        ax.set_title(f"CWT 3D Surface ({data['type']}) - {muscle} Cycle {seg['cycle_id']}")
        ax.set_xlim(0, 100) # Pastikan limit 0-100%
        
        ax.view_init(elev=40, azim=-45)

    plt.tight_layout()
    plt.show()

# --- Blok Testing Mandiri ---
if __name__ == "__main__":
    # Buat sinyal dummy Burst (Aktivasi Otot simulasi)
    fs_test = 2000
    t = np.linspace(0, 1, fs_test)
    # Sinyal burst di tengah (0.4s - 0.6s) frekuensi 150Hz
    sig = np.zeros_like(t)
    mask = (t > 0.4) & (t < 0.6)
    sig[mask] = np.sin(2 * np.pi * 150 * t[mask]) * np.hanning(sum(mask))
    
    dummy_segs = [{
        'cycle_id': 1,
        'fs': fs_test,
        'time': t,
        'gl_denoised': sig,
        'vl_denoised': sig
    }]
    
    # 1. Test Morlet
    print("Testing Morlet...")
    res_morlet = compute_cwt_for_segments(dummy_segs, wavelet_type='morlet')
    plot_cwt(res_morlet, mode='2D', use_db=True)
    
    # 2. Test db4 (Approximation)
    print("Testing db4 (Numerical)...")
    res_db4 = compute_cwt_for_segments(dummy_segs, wavelet_type='db4')
    plot_cwt(res_db4, mode='3D', use_db=False)