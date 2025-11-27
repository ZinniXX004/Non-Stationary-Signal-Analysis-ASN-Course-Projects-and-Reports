import numpy as np
import matplotlib.pyplot as plt

# Constants for Method Selection
METHOD_STANDARD = "Standard BPF (PDF/Delphi)"
METHOD_RBJ = "RBJ Cascade (HPF + LPF)"

class ManualFilterDesigner:
    """
    Handles coefficient calculation and manual filtering operations
    without external signal processing libraries.
    """
    def __init__(self, fs):
        self.fs = float(fs)
        self.T = 1.0 / self.fs

    # -------------------------------------------------------------------------
    # METHOD 1: Standard BPF (From PDF/Delphi)
    # Derivation: H(s) -> Bilinear Transform -> H(z)
    # -------------------------------------------------------------------------
    def compute_standard_bpf_coeffs(self, f_low, f_high):
        """
        Calculates coefficients for a single 2nd Order Bandpass Filter.
        Based on Analog Prototype H(s) = (BW*s) / (s^2 + BW*s + w0^2).
        """
        # 1. Pre-warping frequencies
        # w_d = (2/T) * tan(w_a * T / 2)
        wd_low = 2 * self.fs * np.tan((2 * np.pi * f_low) / (2 * self.fs))
        wd_high = 2 * self.fs * np.tan((2 * np.pi * f_high) / (2 * self.fs))
        
        # 2. Analog Parameters
        w0_sq = wd_low * wd_high        # Center frequency squared
        bw = wd_high - wd_low           # Bandwidth
        
        # 3. Bilinear Substitution Constants
        # s = C * (1 - z^-1) / (1 + z^-1)
        C = 2 * self.fs
        C2 = C * C
        
        # 4. Coefficient derivation (Algebraic expansion of substitution)
        # Denominator (a)
        a0_raw = C2 + bw*C + w0_sq
        a1_raw = 2*w0_sq - 2*C2
        a2_raw = C2 - bw*C + w0_sq
        
        # Numerator (b)
        b0_raw = bw * C
        b1_raw = 0.0
        b2_raw = -(bw * C)
        
        # Normalize so a0 = 1
        b = np.zeros(3)
        a = np.zeros(3)
        
        b[0] = b0_raw / a0_raw
        b[1] = b1_raw / a0_raw
        b[2] = b2_raw / a0_raw
        
        a[0] = 1.0
        a[1] = a1_raw / a0_raw
        a[2] = a2_raw / a0_raw
        
        return b, a

    # -------------------------------------------------------------------------
    # METHOD 2: RBJ Cascade (Audio EQ Cookbook)
    # Derivation: 2nd Order HPF followed by 2nd Order LPF
    # -------------------------------------------------------------------------
    def compute_rbj_coeffs(self, cutoff, filter_type):
        """
        Calculates coefficients for RBJ Biquad (LPF or HPF).
        """
        w0 = 2 * np.pi * cutoff / self.fs
        alpha = np.sin(w0) / (2 * 0.707) # Q = 0.707 (Butterworth)
        cos_w0 = np.cos(w0)
        
        b = np.zeros(3)
        a = np.zeros(3)
        norm = 1 / (1 + alpha)

        if filter_type == 'lowpass':
            b[0] = ((1 - cos_w0) / 2) * norm
            b[1] = (1 - cos_w0) * norm
            b[2] = ((1 - cos_w0) / 2) * norm
            a[0] = 1.0
            a[1] = (-2 * cos_w0) * norm
            a[2] = (1 - alpha) * norm
            
        elif filter_type == 'highpass':
            b[0] = ((1 + cos_w0) / 2) * norm
            b[1] = -(1 + cos_w0) * norm
            b[2] = ((1 + cos_w0) / 2) * norm
            a[0] = 1.0
            a[1] = (-2 * cos_w0) * norm
            a[2] = (1 - alpha) * norm
            
        return b, a

    # -------------------------------------------------------------------------
    # Core Filtering Logic (Difference Equation)
    # -------------------------------------------------------------------------
    def lfilter_manual(self, b, a, x):
        """Direct Form I implementation."""
        y = np.zeros_like(x)
        x_prev1, x_prev2 = 0.0, 0.0
        y_prev1, y_prev2 = 0.0, 0.0
        
        b0, b1, b2 = b[0], b[1], b[2]
        a1, a2 = a[1], a[2]
        
        for n in range(len(x)):
            x_curr = x[n]
            y_curr = (b0*x_curr) + (b1*x_prev1) + (b2*x_prev2) - (a1*y_prev1) - (a2*y_prev2)
            
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

    # -------------------------------------------------------------------------
    # Analysis Helper: Manual Frequency Response
    # -------------------------------------------------------------------------
    def freqz_manual(self, b, a, n_points=512):
        """
        Calculates Frequency Response H(e^jw) manually.
        H(z) = B(z)/A(z) evaluated at z = e^(j*w)
        """
        w = np.linspace(0, np.pi, n_points) # 0 to Nyquist
        z = np.exp(1j * w)
        
        # Evaluate Polynomials
        # B(z) = b0 + b1*z^-1 + b2*z^-2
        num = b[0] + b[1]*z**(-1) + b[2]*z**(-2)
        den = a[0] + a[1]*z**(-1) + a[2]*z**(-2)
        
        H = num / den
        freqs_hz = w * self.fs / (2 * np.pi)
        
        return freqs_hz, np.abs(H)

def apply_bpf(segments, method=METHOD_STANDARD):
    """
    Main function called by GUI.
    Applies BPF (20-450 Hz) using the selected method.
    """
    if not segments:
        return []

    fs = segments[0]['fs']
    designer = ManualFilterDesigner(fs)
    f_low, f_high = 20.0, 450.0
    
    print(f"[-] Applying BPF ({method}) from {f_low} to {f_high} Hz...")
    
    processed_segments = []
    
    # Pre-calculate coefficients based on method
    if method == METHOD_STANDARD:
        b, a = designer.compute_standard_bpf_coeffs(f_low, f_high)
        # Standard method is a single filter stage
        filters = [(b, a)]
        
    else: # METHOD_RBJ
        b_hp, a_hp = designer.compute_rbj_coeffs(f_low, 'highpass')
        b_lp, a_lp = designer.compute_rbj_coeffs(f_high, 'lowpass')
        # RBJ method is a cascade: Signal -> HPF -> LPF
        filters = [(b_hp, a_hp), (b_lp, a_lp)]

    # Apply filtering
    for seg in segments:
        new_seg = seg.copy()
        
        # Filter GL
        gl_sig = np.array(seg['gl_segment'], dtype=float)
        for b, a in filters:
            gl_sig = designer.filtfilt_manual(b, a, gl_sig)
        new_seg['gl_filtered'] = gl_sig
        
        # Filter VL
        vl_sig = np.array(seg['vl_segment'], dtype=float)
        for b, a in filters:
            vl_sig = designer.filtfilt_manual(b, a, vl_sig)
        new_seg['vl_filtered'] = vl_sig
        
        processed_segments.append(new_seg)
        
    return processed_segments

def get_frequency_response_data(fs, method=METHOD_STANDARD):
    """
    Returns (freqs, magnitude_linear) for the selected method to be plotted in GUI.
    NO dB conversion, as requested.
    """
    designer = ManualFilterDesigner(fs)
    f_low, f_high = 20.0, 450.0
    
    if method == METHOD_STANDARD:
        b, a = designer.compute_standard_bpf_coeffs(f_low, f_high)
        f, h = designer.freqz_manual(b, a)
        mag = h # Single stage response
        
    else: # METHOD_RBJ
        b_hp, a_hp = designer.compute_rbj_coeffs(f_low, 'highpass')
        b_lp, a_lp = designer.compute_rbj_coeffs(f_high, 'lowpass')
        
        f, h_hp = designer.freqz_manual(b_hp, a_hp)
        _, h_lp = designer.freqz_manual(b_lp, a_lp)
        
        # Cascade response = Multiplication in frequency domain
        mag = h_hp * h_lp
        
    # Return Linear Magnitude (0.0 - 1.0 typically)
    return f, mag

def plot_filtered_comparison(segments, cycle_idx=0):
    """Helper for standalone testing"""
    if not segments: return
    seg = segments[cycle_idx]
    t = seg['time'] - seg['time'][0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, seg['gl_segment'], color='lightgray', label='Raw')
    plt.plot(t, seg['gl_filtered'], color='blue', label='Filtered')
    plt.title(f"Filter Check - Cycle {seg['cycle_id']}")
    plt.legend()
    plt.show()

# --- Standalone Testing ---
if __name__ == "__main__":
    fs = 2000
    t = np.linspace(0, 1, fs)
    # 5Hz noise + 100Hz signal
    sig = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*100*t)
    dummy = [{'cycle_id':1, 'fs':fs, 'time':t, 'gl_segment':sig, 'vl_segment':sig}]
    
    print("Testing Standard BPF...")
    res1 = apply_bpf(dummy, METHOD_STANDARD)
    
    print("Testing RBJ Cascade...")
    res2 = apply_bpf(dummy, METHOD_RBJ)
    
    # Compare Freq Responses (Linear)
    f, mag1 = get_frequency_response_data(fs, METHOD_STANDARD)
    _, mag2 = get_frequency_response_data(fs, METHOD_RBJ)
    
    plt.figure()
    plt.plot(f, mag1, label=METHOD_STANDARD)
    plt.plot(f, mag2, label=METHOD_RBJ, linestyle='--')
    plt.title("Frequency Response Comparison (Linear)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 600)
    plt.show()