import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ManualCWT:
    """
    Features:
    1. Supports 'Morlet' (Analytical) and 'db4' (Numerical Approximation) wavelets
    2. Uses direct convolution (time-domain) instead of FFT-based convolution
    """
    def __init__(self, fs, f_min=20, f_max=450, num_scales=64, wavelet_type='morlet'):
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        self.num_scales = num_scales
        self.wavelet_type = wavelet_type.lower()
        
        # 1. Determine Center Frequency (fc)
        # fc is the normalized center frequency of the mother wavelet.
        # Values are approximations based on standard literature.
        if self.wavelet_type == 'morlet':
            self.fc = 0.8125  # Standard center freq for Morlet (w0=6)
        else: # db4
            self.fc = 0.7143  # Approximate center freq for db4
            
        # 2. Generate Scales from Frequencies
        # We use logarithmic spacing for frequencies to capture low-freq details better.
        # Formula: Scale (a) = (fc * fs) / f
        self.freqs = np.logspace(np.log10(f_min), np.log10(f_max), num_scales)
        self.scales = (self.fc * fs) / self.freqs
        
        # 3. Pre-compute db4 prototype if needed (to save computation time)
        self.db4_prototype = None
        if self.wavelet_type == 'db4':
            self.db4_prototype = self._generate_db4_prototype()

    def _morlet_wavelet(self, scale):
        """
        Analytical Formula:
        psi(t) = C * exp(i * w0 * t) * exp(-t^2 / 2)
        
        Where:
        C  = pi^(-0.25) (Normalization constant)
        w0 = 6 (Wavenumber, standard for CWT to satisfy admissibility condition)
        """
        # Define effective time support [-5 sigma, +5 sigma]
        # The wavelet is scaled by dividing t by 'scale'
        M = int(scale * 10) 
        if M % 2 == 0: M += 1 # Ensure odd length for symmetry
        
        t = np.arange(-(M//2), (M//2)+1) / scale
        
        # Normalization Constant
        C = np.pi**(-0.25)
        
        # Wavenumber
        w0 = 6 
        
        # Calculate Wavelet
        psi = C * np.exp(1j * w0 * t) * np.exp(-0.5 * t**2)
        
        # Energy Normalization: 1 / sqrt(scale)
        # Ensures the energy of the wavelet remains constant across scales
        return psi * (1 / np.sqrt(scale))

    def _generate_db4_prototype(self, iterations=6):
        """
        Generates the db4s wavelet shape using the Cascade Algorithm
        Since db4 has no closed-form equation, we iteratively upscale and convolve
        the filter coefficients to approximate the continuous function
        """
        # Daubechies 4 Decomposition Coefficients (derived analytically)
        h0 = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
        h1 = (3 + np.sqrt(3)) / (4 * np.sqrt(2))
        h2 = (3 - np.sqrt(3)) / (4 * np.sqrt(2))
        h3 = (1 - np.sqrt(3)) / (4 * np.sqrt(2))
        
        # Reconstruction Low Pass Filter (Scaling Filter)
        L_R = np.array([h3, h2, h1, h0, h0, h1, h2, h3]) 
        
        # Reconstruction High Pass Filter (Wavelet Filter)
        # Alternating signs for Quadrature Mirror Filter
        H_R = np.array([h3, -h2, h1, -h0, h0, -h1, h2, -h3]) 
        
        # Cascade Algorithm (Iterative Upsampling + Convolution)
        # Start with the high-pass filter (Wavelet branch)
        psi = H_R 
        
        for _ in range(iterations):
            # 1. Upsample (Insert zeros between samples)
            psi_up = np.zeros(len(psi) * 2)
            psi_up[::2] = psi
            
            # 2. Convolve with Scaling Filter
            # This smooths the discrete steps
            psi = np.convolve(psi_up, L_R)
            
        return psi

    def _db4_wavelet(self, scale):
        prototype = self.db4_prototype
        len_proto = len(prototype)
        
        # Determine target length based on scale
        # db4 effective width is approx 7. Effective length ~ scale * 7
        target_len = int(scale * 7)
        if target_len < 8: target_len = 8 # Minimum length constraint
        
        # Manual Linear Interpolation (Resampling)
        x_original = np.linspace(0, 1, len_proto)
        x_target = np.linspace(0, 1, target_len)
        
        psi_resampled = np.interp(x_target, x_original, prototype)
        
        # Energy Normalization: 1 / sqrt(scale)
        return psi_resampled * (1 / np.sqrt(scale))

    def compute(self, signal):
        """
        Method:
        Direct convolution of the signal with the scaled wavelet for each scales
        
        Returns:
            energy_density: |CWT|^2 (Matrix: Scales x Time)
            freqs: Frequency axis corresponding to scales
        """
        n = len(signal)
        # Initialize complex matrix for coefficients
        cwt_matrix = np.zeros((self.num_scales, n), dtype=complex)
        
        # Iterate through each scale (frequency band)
        for i, scale in enumerate(self.scales):
            # 1. Generate Mother Wavelet at current scale
            if self.wavelet_type == 'morlet':
                psi = self._morlet_wavelet(scale)
            else:
                psi = self._db4_wavelet(scale)
            
            # 2. Perform Convolution (Signal * Wavelet)
            # mode='same' ensures output length matches input length
            # This represents the inner product <signal, wavelet> at every time shift
            coeffs = np.convolve(signal, psi, mode='same')
            
            cwt_matrix[i, :] = coeffs
            
        # Calculate Scalogram (Energy Density)
        # Formula: E = |W(a, b)|^2
        energy_density = np.abs(cwt_matrix)**2
        
        return energy_density, self.freqs

def compute_cwt_for_segments(segments, wavelet_type='morlet'):
    if not segments: return []
    
    # Get sampling frequency
    fs = segments[0]['fs']
    
    # Initialize Manual CWT Solver
    cwt_solver = ManualCWT(fs, f_min=20, f_max=450, num_scales=50, wavelet_type=wavelet_type)
    
    print(f"[-] Computing Manual CWT (Wavelet: {wavelet_type}, Scales: 50)...")
    
    processed_segments = []
    for seg in segments:
        new_seg = seg.copy()
        
        # Select best available signal: Denoised > Filtered > Raw
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
        
        # Compute CWT for Gastrocnemius (GL)
        E_gl, f_gl = cwt_solver.compute(gl_sig)
        new_seg['cwt_gl'] = {'f': f_gl, 't': seg['time'], 'E': E_gl, 'type': wavelet_type}
        
        # Compute CWT for Vastus Lateralis (VL)
        E_vl, f_vl = cwt_solver.compute(vl_sig)
        new_seg['cwt_vl'] = {'f': f_vl, 't': seg['time'], 'E': E_vl, 'type': wavelet_type}
        
        processed_segments.append(new_seg)
        
    return processed_segments

def plot_cwt(segments, cycle_idx=0, muscle='GL', use_db=False, mode='2D'):
    if not segments: return
    
    seg = segments[cycle_idx]
    key = 'cwt_gl' if muscle == 'GL' else 'cwt_vl'
    
    if key not in seg:
        print("CWT data not found.")
        return

    data = seg[key]
    f = data['f']
    t = data['t']
    E = data['E'] # Energy Density |W|^2
    
    # Normalize time to % gait cycle
    # Map t[start] to 0% and t[end] to 100%
    if len(t) > 1:
        t_norm = (t - t[0]) / (t[-1] - t[0]) * 100
    else:
        t_norm = t
    
    # Unit Conversion
    if use_db:
        # Convert to decibels: 10 * log10(Power)
        # Add epsilon to avoid log(0) error
        Z_plot = 10 * np.log10(E + 1e-10)
        unit_label = "Power Spectral Density (dB)"
    else:
        Z_plot = E
        unit_label = "Energy Density (Linear)"

    fig = plt.figure(figsize=(10, 6))
    
    if mode == '2D':
        ax = fig.add_subplot(111)
        # Use contourf for smooth energy representation
        levels = 50
        cf = ax.contourf(t_norm, f, Z_plot, levels=levels, cmap='jet')
        cbar = fig.colorbar(cf, ax=ax, label=unit_label)
        
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('% Gait Cycle') 
        ax.set_title(f"CWT Scalogram ({data['type']}) - {muscle} Cycle {seg['cycle_id']}")
        ax.set_xlim(0, 100)
        
    elif mode == '3D':
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid for 3D plotting
        T_grid, F_grid = np.meshgrid(t_norm, f)
        
        # Surface plot
        surf = ax.plot_surface(T_grid, F_grid, Z_plot, cmap='viridis', 
                               edgecolor='none', rstride=1, cstride=1, alpha=0.9)
        
        fig.colorbar(surf, ax=ax, label=unit_label, pad=0.1)
        
        ax.set_xlabel('% Gait Cycle') 
        ax.set_ylabel('Frequency (Hz)')
        ax.set_zlabel(unit_label)
        ax.set_title(f"CWT 3D Surface ({data['type']}) - {muscle} Cycle {seg['cycle_id']}")
        ax.set_xlim(0, 100)
        
        # Initial Camera View
        ax.view_init(elev=40, azim=-45)

    plt.tight_layout()
    plt.show()

# <<<< Standalone Testing Block >>>>
if __name__ == "__main__":
    # Generate Dummy Burst Signal (Simulating Muscle Activation)
    fs_test = 2000
    t = np.linspace(0, 1, fs_test)
    # Burst in the middle (0.4s - 0.6s) at 150Hz
    sig = np.zeros_like(t)
    mask = (t > 0.4) & (t < 0.6)
    sig[mask] = np.sin(2 * np.pi * 150 * t[mask]) * np.hanning(sum(mask))
    
    # Create dummy segment structure
    dummy_segs = [{
        'cycle_id': 1,
        'fs': fs_test,
        'time': t,
        'gl_segment': sig, 'vl_segment': sig,
        'gl_filtered': sig, 'vl_filtered': sig,
        'gl_denoised': sig, 'vl_denoised': sig
    }]
    
    # 1. Test Morlet
    print("Testing Morlet Wavelet...")
    res_morlet = compute_cwt_for_segments(dummy_segs, wavelet_type='morlet')
    plot_cwt(res_morlet, mode='2D', use_db=True)
    
    # 2. Test db4 (Approximation)
    print("Testing db4 Wavelet (Numerical)...")
    res_db4 = compute_cwt_for_segments(dummy_segs, wavelet_type='db4')
    plot_cwt(res_db4, mode='3D', use_db=False)