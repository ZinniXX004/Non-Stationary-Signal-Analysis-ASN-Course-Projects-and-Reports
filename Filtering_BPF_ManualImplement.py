import numpy as np
import matplotlib.pyplot as plt

class ManualButterworthBPF:
    """
    Manual implementation of a Digital 2nd Order Butterworth Bandpass Filter.
    Derivation based on Analog Prototype H(s) and Bilinear Transformation.
    Reference: PDF Material Page 25 (2nd Order BPF) & Delphi Legacy Code.
    """
    def __init__(self, fs):
        self.fs = fs
        self.T = 1.0 / fs

    def compute_coeffs(self, f_low, f_high):
        """
        Calculates coefficients for 2nd Order BPF using Bilinear Transform.
        
        Analog Transfer Function H(s):
        H(s) = ( (w0/Q) * s ) / ( s^2 + (w0/Q)*s + w0^2 )
        
        Where:
        w0 = sqrt(w_low * w_high)  -> Center Frequency (Geometric Mean)
        BW = w_high - w_low        -> Bandwidth
        Q  = w0 / BW               -> Quality Factor
        
        Substitute s = (2/T) * (1 - z^-1) / (1 + z^-1)
        
        Returns:
            b (array): [b0, b1, b2]
            a (array): [a0, a1, a2] (Normalized so a0=1)
        """
        # 1. Analog Parameters
        w_low = 2 * np.pi * f_low
        w_high = 2 * np.pi * f_high
        
        # Center Frequency & Bandwidth
        # Note: For wideband BPF, we often cascade HPF and LPF. 
        # But for a standard 2nd order BPF section (resonant), we use w0 and BW.
        # Let's use the standard BPF design derived from Bilinear Transform.
        
        # Pre-warping (Optional but recommended for accuracy near Nyquist)
        # omega_d = (2/T) * tan(omega_a * T / 2)
        # For this manual implementation, we stick to the direct substitution 
        # logic seen in your Delphi code (lpfandbpf3rdbutter.txt) or standard textbook.
        
        # Using Pre-warped frequencies for better accuracy:
        wd_low = 2 * self.fs * np.tan(w_low / (2 * self.fs))
        wd_high = 2 * self.fs * np.tan(w_high / (2 * self.fs))
        
        w0 = np.sqrt(wd_low * wd_high)
        bw = wd_high - wd_low
        
        # K is a scaling factor from the s-domain term w0/Q (which is BW)
        # H(s) numerator is BW * s
        # H(s) denominator is s^2 + BW*s + w0^2
        
        # Constants for Bilinear Transform substitution:
        # s = C * (1-z)/(1+z), where C = 2/T = 2*fs
        C = 2 * self.fs
        C2 = C * C
        
        # Derivation:
        # H(z) = (BW * C * (1 - z^-2)) / ( (C^2 + BW*C + w0^2) + (2*w0^2 - 2*C^2)z^-1 + (C^2 - BW*C + w0^2)z^-2 )
        # We need to verify this algebra manually.
        
        # Denominator terms (D):
        # D = (2/T)^2 * (1-z^-1)^2 + BW*(2/T)*(1-z^-1)(1+z^-1) + w0^2 * (1+z^-1)^2
        # Expanding D:
        # coeff z^0: C^2 + BW*C + w0^2
        # coeff z^1: 2*w0^2 - 2*C^2
        # coeff z^2: C^2 - BW*C + w0^2
        
        a0_raw = C2 + bw*C + w0**2
        a1_raw = 2*w0**2 - 2*C2
        a2_raw = C2 - bw*C + w0**2
        
        # Numerator terms (N):
        # N = BW * (2/T) * (1-z^-1) * (1+z^-1)
        # N = BW * C * (1 - z^-2)
        
        b0_raw = bw * C
        b1_raw = 0.0
        b2_raw = -(bw * C)
        
        # Normalize by a0_raw so a[0] becomes 1.0
        b = np.zeros(3)
        a = np.zeros(3)
        
        b[0] = b0_raw / a0_raw
        b[1] = b1_raw / a0_raw
        b[2] = b2_raw / a0_raw
        
        a[0] = 1.0
        a[1] = a1_raw / a0_raw
        a[2] = a2_raw / a0_raw
        
        return b, a

    def lfilter_manual(self, b, a, x):
        """
        Implements Direct Form I Difference Equation.
        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        """
        y = np.zeros_like(x)
        
        x_prev1, x_prev2 = 0.0, 0.0
        y_prev1, y_prev2 = 0.0, 0.0
        
        b0, b1, b2 = b[0], b[1], b[2]
        a1, a2 = a[1], a[2]
        
        for n in range(len(x)):
            x_curr = x[n]
            
            y_curr = (b0 * x_curr) + (b1 * x_prev1) + (b2 * x_prev2) - (a1 * y_prev1) - (a2 * y_prev2)
            
            x_prev2 = x_prev1
            x_prev1 = x_curr
            y_prev2 = y_prev1
            y_prev1 = y_curr
            
            y[n] = y_curr
            
        return y

    def filtfilt_manual(self, b, a, x):
        """Zero-phase filtering (Forward-Backward)."""
        y_fwd = self.lfilter_manual(b, a, x)
        y_rev = y_fwd[::-1]
        y_back = self.lfilter_manual(b, a, y_rev)
        return y_back[::-1]

def apply_bpf(segments):
    """
    Applies the Manual 2nd Order Butterworth BPF (20-450 Hz).
    """
    if not segments:
        return []

    fs = segments[0]['fs']
    # Define Cutoff Frequencies
    f_low = 20.0
    f_high = 450.0
    
    # Initialize Filter
    bpf = ManualButterworthBPF(fs)
    b, a = bpf.compute_coeffs(f_low, f_high)
    
    print(f"[-] Applying Manual 2nd Order Butterworth BPF (20-450 Hz) to {len(segments)} segments...")
    print(f"    Coeffs B: {b}")
    print(f"    Coeffs A: {a}")
    
    processed_segments = []
    
    for seg in segments:
        new_seg = seg.copy()
        
        # --- Process GL ---
        gl_raw = np.array(seg['gl_segment'], dtype=float)
        new_seg['gl_filtered'] = bpf.filtfilt_manual(b, a, gl_raw)
        
        # --- Process VL ---
        vl_raw = np.array(seg['vl_segment'], dtype=float)
        new_seg['vl_filtered'] = bpf.filtfilt_manual(b, a, vl_raw)
        
        processed_segments.append(new_seg)
        
    return processed_segments

def plot_filtered_comparison(segments, cycle_idx=0):
    """Plots comparison for validation."""
    if not segments: return
    
    seg = segments[cycle_idx]
    time = seg['time']
    t_plot = time - time[0]
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t_plot, seg['gl_segment'], color='lightgray', label='Raw GL')
    plt.plot(t_plot, seg['gl_filtered'], color='blue', linewidth=1.5, label='BPF GL (20-450Hz)')
    plt.title(f"BPF Result - Cycle {seg['cycle_id']} (Gastrocnemius)")
    plt.legend(loc='upper right')
    
    plt.subplot(2, 1, 2)
    plt.plot(t_plot, seg['vl_segment'], color='lightgray', label='Raw VL')
    plt.plot(t_plot, seg['vl_filtered'], color='red', linewidth=1.5, label='BPF VL (20-450Hz)')
    plt.title(f"BPF Result - Cycle {seg['cycle_id']} (Vastus)")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# --- Standalone Test Block ---
if __name__ == "__main__":
    # Test with dummy signal
    fs_dummy = 2000
    t = np.linspace(0, 1, fs_dummy)
    # 5Hz (Artifact) + 100Hz (Signal) + 1000Hz (Noise)
    raw_signal = 2.0*np.sin(2*np.pi*5*t) + np.sin(2*np.pi*100*t) + 0.5*np.sin(2*np.pi*1000*t)
    
    dummy_seg = [{'cycle_id': 1, 'fs': fs_dummy, 'time': t, 
                  'gl_segment': raw_signal, 'vl_segment': raw_signal}]
    
    res = apply_bpf(dummy_seg)
    
    plt.figure(figsize=(10,4))
    plt.plot(t, raw_signal, label='Raw (5+100+1000 Hz)', alpha=0.5)
    plt.plot(t, res[0]['gl_filtered'], label='Filtered (Should be ~100Hz)', color='green')
    plt.legend()
    plt.title("Manual BPF 2nd Order Test")
    plt.show()