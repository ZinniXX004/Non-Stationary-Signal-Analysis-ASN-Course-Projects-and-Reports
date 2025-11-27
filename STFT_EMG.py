import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

class ManualWindow:
    """
    Manual implementation of Signal Windowing functions.
    """
    @staticmethod
    def get_window(window_type, N, beta=14.0):
        """
        Generates a window array of length N based on the selected type.
        """
        if N <= 1: return np.ones(N)
        n = np.arange(N)
        
        if window_type == 'Rectangular':
            return np.ones(N)
        elif window_type == 'Hanning':
            return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
        elif window_type == 'Hamming':
            return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
        elif window_type == 'Blackman':
            return (0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 
                    0.08 * np.cos(4 * np.pi * n / (N - 1)))
        elif window_type == 'Triangular':
            return 1 - np.abs((n - (N - 1) / 2) / ((N - 1) / 2))
        elif window_type == 'Kaiser':
            def manual_i0(x):
                sum_val = 1.0
                term = 1.0
                for k in range(1, 25):
                    term *= (x**2) / (4 * (k**2))
                    sum_val += term
                return sum_val
            alpha = (N - 1) / 2.0
            denom = manual_i0(beta)
            window = np.zeros(N)
            for i in range(N):
                term = np.sqrt(1 - ((i - alpha) / alpha)**2) if abs((i - alpha) / alpha) <= 1 else 0
                window[i] = manual_i0(beta * term) / denom
            return window
        else:
            return np.ones(N)

class ManualFFT:
    """
    Manual implementation of Fast Fourier Transform (FFT).
    Uses the Cooley-Tukey Radix-2 Recursive algorithm.
    """
    def __init__(self):
        pass

    def _next_power_of_2(self, x):
        """Finds the next power of 2 greater than or equal to x."""
        return 1 if x == 0 else 2**(x - 1).bit_length()

    def recursive_fft(self, x):
        """
        Cooley-Tukey Recursive FFT Algorithm.
        Input x must have a length equal to a power of 2.
        """
        N = len(x)
        if N <= 1: return x
        
        # Divide
        even = self.recursive_fft(x[0::2])
        odd = self.recursive_fft(x[1::2])
        
        # Conquer (Butterfly Operation)
        T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
        
        left = [even[k] + T[k] for k in range(N // 2)]
        right = [even[k] - T[k] for k in range(N // 2)]
        
        return left + right

    def compute_fft(self, signal):
        """
        Wrapper to pad the signal to the next power of 2 and compute FFT.
        Returns the full spectrum and the padded length.
        """
        n_original = len(signal)
        n_padded = self._next_power_of_2(n_original)
        
        # Zero padding if necessary
        if n_padded != n_original:
            padded_signal = np.pad(signal, (0, n_padded - n_original), 'constant')
        else:
            padded_signal = signal
            
        spectrum = self.recursive_fft(padded_signal)
        return np.array(spectrum), n_padded

class ManualSTFT:
    def __init__(self, window_size=256, overlap=128, window_type='Hanning'):
        self.window_size = window_size
        self.overlap = overlap
        self.window_type = window_type
        self.fft_solver = ManualFFT()

    def compute(self, signal, fs):
        """
        Computes the Short-Time Fourier Transform.
        
        Returns:
            f: Array frekuensi
            t: Array waktu
            Zxx: Magnitudo Spektrogram (Frequency x Time)
        """
        n_signal = len(signal)
        step = self.window_size - self.overlap
        
        # Calculate number of segments
        if n_signal < self.window_size:
            n_segments = 1
        else:
            n_segments = (n_signal - self.window_size) // step + 1
        
        # Generate Window Function
        window = ManualWindow.get_window(self.window_type, self.window_size)
        
        stft_matrix = []
        time_stamps = []
        
        for i in range(n_segments):
            start = i * step
            end = start + self.window_size
            
            # Extract segment
            segment = signal[start:end]
            
            # Handle edge case if segment is shorter than window
            if len(segment) < self.window_size:
                segment = np.pad(segment, (0, self.window_size - len(segment)), 'constant')
            
            # Apply Window
            windowed_segment = segment * window
            
            # Compute Manual FFT
            spectrum, n_fft = self.fft_solver.compute_fft(windowed_segment)
            
            # Take positive frequencies (first half)
            half_n = n_fft // 2
            magnitude = np.abs(spectrum[:half_n]) / n_fft
            
            stft_matrix.append(magnitude)
            
            # Time stamp (center of the window)
            t_mid = (start + self.window_size/2) / fs
            time_stamps.append(t_mid)
            
        # Transpose to get (Frequency x Time) shape
        Zxx = np.array(stft_matrix).T
        
        # Compute Frequency Axis
        n_final_fft = self.fft_solver._next_power_of_2(self.window_size)
        freqs = np.linspace(0, fs/2, n_final_fft // 2)
        
        return freqs, np.array(time_stamps), Zxx

def compute_stft_for_segments(segments, window_type='Hanning'):
    """Fungsi wrapper untuk memproses list segment."""
    if not segments: return []
    
    stft_solver = ManualSTFT(window_size=256, overlap=128, window_type=window_type)
    print(f"[-] Computing STFT (Manual FFT, Window: {window_type})...")
    
    processed_segments = []
    for seg in segments:
        new_seg = seg.copy()
        fs = seg['fs']
        
        # --- FIX: Safer input retrieval using if-else ---
        # Priority: Denoised > Filtered > Segment (Raw)
        if 'gl_denoised' in seg:
            gl_sig = seg['gl_denoised']
        elif 'gl_filtered' in seg:
            gl_sig = seg['gl_filtered']
        else:
            gl_sig = seg['gl_segment']

        if 'vl_denoised' in seg:
            vl_sig = seg['vl_denoised']
        elif 'vl_filtered' in seg:
            vl_sig = seg['vl_filtered']
        else:
            vl_sig = seg['vl_segment']
        
        # Hitung STFT GL
        f_gl, t_gl, Z_gl = stft_solver.compute(gl_sig, fs)
        new_seg['stft_gl'] = {'f': f_gl, 't': t_gl, 'Z': Z_gl}
        
        # Hitung STFT VL
        f_vl, t_vl, Z_vl = stft_solver.compute(vl_sig, fs)
        new_seg['stft_vl'] = {'f': f_vl, 't': t_vl, 'Z': Z_vl}
        
        processed_segments.append(new_seg)
        
    return processed_segments

def plot_stft(segments, cycle_idx=0, muscle='GL', use_db=False, mode='2D'):
    """
    Fungsi Plotting Fleksibel (Dipanggil oleh GUI).
    Modifikasi: Sumbu X menggunakan % Gait Cycle (0-100).
    """
    if not segments: return
    
    seg = segments[cycle_idx]
    key = 'stft_gl' if muscle == 'GL' else 'stft_vl'
    
    if key not in seg:
        print("Data STFT belum dihitung.")
        return

    data = seg[key]
    f = data['f']
    t = data['t']
    Z = data['Z']
    
    # --- MODIFIKASI UTAMA: Normalisasi Waktu ke % Gait Cycle ---
    if len(t) > 1:
        t_norm = (t - t[0]) / (t[-1] - t[0]) * 100
    else:
        t_norm = t 
    
    # Konversi Unit Magnitudo
    if use_db:
        Z_plot = 20 * np.log10(Z + 1e-10)
        unit_label = "Magnitude (dB)"
    else:
        Z_plot = Z
        unit_label = "Magnitude (Linear)"

    # Create Plot
    fig = plt.figure(figsize=(10, 6))
    
    if mode == '2D':
        ax = fig.add_subplot(111)
        # Use contourf for smoother spectrogram look
        cf = ax.contourf(t_norm, f, Z_plot, levels=50, cmap='jet')
        cbar = fig.colorbar(cf, ax=ax, label=unit_label)
        
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('% Gait Cycle') 
        ax.set_title(f"STFT 2D Analysis - {muscle} (Cycle {seg['cycle_id']})")
        ax.set_ylim(0, 500) 
        ax.set_xlim(0, 100) 
        
    elif mode == '3D':
        ax = fig.add_subplot(111, projection='3d')
        
        T_grid, F_grid = np.meshgrid(t_norm, f)
        
        surf = ax.plot_surface(T_grid, F_grid, Z_plot, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, ax=ax, label=unit_label, pad=0.1)
        
        ax.set_xlabel('% Gait Cycle') 
        ax.set_ylabel('Frequency (Hz)')
        ax.set_zlabel(unit_label)
        ax.set_title(f"STFT 3D Analysis - {muscle} (Cycle {seg['cycle_id']})")
        ax.set_ylim(0, 500)
        ax.set_xlim(0, 100) 
        
        ax.view_init(elev=30, azim=120)

    plt.tight_layout()
    plt.show()

# --- Blok Testing Mandiri ---
if __name__ == "__main__":
    # Buat sinyal dummy untuk testing
    fs_test = 2000
    t_dur = np.linspace(0, 1, fs_test)
    sig = np.sin(2*np.pi*100*t_dur[:1000]) 
    sig = np.concatenate([sig, np.sin(2*np.pi*300*t_dur[1000:])]) 
    
    # Complete dummy structure to prevent KeyError
    dummy_segs = [{
        'cycle_id': 1,
        'fs': fs_test,
        'gl_segment': sig,  # Raw
        'vl_segment': sig,  # Raw
        'gl_denoised': sig, # Denoised
        'vl_denoised': sig  # Denoised
    }]
    
    # Hitung STFT
    res = compute_stft_for_segments(dummy_segs, window_type='Hamming')
    
    # Tes Plot 2D dB
    print("Plotting 2D dB...")
    plot_stft(res, use_db=True, mode='2D')
    
    # Tes Plot 3D Linear
    print("Plotting 3D Linear...")
    plot_stft(res, use_db=False, mode='3D')