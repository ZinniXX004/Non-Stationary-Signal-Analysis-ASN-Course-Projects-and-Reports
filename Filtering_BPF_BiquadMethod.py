import numpy as np
import matplotlib.pyplot as plt

class ManualButterworth:
    """
    Manual implementation of a 2nd Order Butterworth Filter (Biquad).
    Uses the Bilinear Transform method for coefficient calculation.
    """
    def __init__(self, fs):
        self.fs = fs

    def compute_coeffs(self, cutoff, type_filter):
        """
        Calculates coefficients 'a' and 'b' for a 2nd order filter.
        Formulas based on 'Audio EQ Cookbook' by Robert Bristow-Johnson.
        
        Args:
            cutoff (float): Cutoff frequency in Hz.
            type_filter (str): 'lowpass' or 'highpass'.
            
        Returns:
            b (array): Numerator coefficients [b0, b1, b2].
            a (array): Denominator coefficients [a0, a1, a2].
        """
        # Convert cutoff frequency to angular frequency
        w0 = 2 * np.pi * cutoff / self.fs
        
        # Calculate Alpha (Bandwidth/Q factor parameter)
        # Q = 0.707 (1/sqrt(2)) is the standard for Butterworth response (maximally flat)
        alpha = np.sin(w0) / (2 * 0.707) 
        cos_w0 = np.cos(w0)

        b = np.zeros(3)
        a = np.zeros(3)

        # Normalization factor (standardize so a0 becomes 1)
        # The raw a0 in the formula is (1 + alpha)
        norm = 1 / (1 + alpha)

        if type_filter == 'lowpass':
            # LPF Coefficients
            b[0] = ((1 - cos_w0) / 2) * norm
            b[1] = (1 - cos_w0) * norm
            b[2] = ((1 - cos_w0) / 2) * norm
            
            a[0] = 1.0 
            a[1] = (-2 * cos_w0) * norm
            a[2] = (1 - alpha) * norm 
            
        elif type_filter == 'highpass':
            # HPF Coefficients
            b[0] = ((1 + cos_w0) / 2) * norm
            b[1] = -(1 + cos_w0) * norm
            b[2] = ((1 + cos_w0) / 2) * norm
            
            a[0] = 1.0
            a[1] = (-2 * cos_w0) * norm
            a[2] = (1 - alpha) * norm
            
        return b, a

    def lfilter_manual(self, b, a, x):
        """
        Implements the Direct Form I Difference Equation manually.
        Equation: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        
        Args:
            b, a: Filter coefficients.
            x: Input signal array.
        """
        y = np.zeros_like(x)
        
        # Initialize filter state (buffer history)
        x_prev1, x_prev2 = 0.0, 0.0
        y_prev1, y_prev2 = 0.0, 0.0
        
        # Extract coefficients for readability
        b0, b1, b2 = b[0], b[1], b[2]
        a1, a2 = a[1], a[2] # a0 is assumed to be 1.0 due to normalization
        
        for n in range(len(x)):
            x_curr = x[n]
            
            # Calculate current output using Difference Equation
            y_curr = (b0 * x_curr) + (b1 * x_prev1) + (b2 * x_prev2) - (a1 * y_prev1) - (a2 * y_prev2)
            
            # Shift history for the next iteration
            x_prev2 = x_prev1
            x_prev1 = x_curr
            y_prev2 = y_prev1
            y_prev1 = y_curr
            
            y[n] = y_curr
            
        return y

    def filtfilt_manual(self, b, a, x):
        """
        Zero-phase filtering: Forward filter -> Reverse -> Backward filter -> Reverse.
        This prevents phase distortion (time delay) in the output signal.
        """
        # 1. Forward Filtering
        y_fwd = self.lfilter_manual(b, a, x)
        
        # 2. Reverse the signal
        y_rev = y_fwd[::-1]
        
        # 3. Backward Filtering (filtering the reversed signal)
        y_back = self.lfilter_manual(b, a, y_rev)
        
        # 4. Reverse back to original orientation
        return y_back[::-1]

def apply_bpf(segments):
    """
    Applies a Cascade Band-Pass Filter (HPF + LPF) to signal segments.
    Target Range: 20 - 450 Hz.
    """
    if not segments:
        return []

    fs = segments[0]['fs']
    butter = ManualButterworth(fs)
    
    # Pre-calculate coefficients
    # Cascade Strategy: Input -> HighPass (20Hz) -> LowPass (450Hz) -> Output
    # This creates a Bandpass effect.
    b_hp, a_hp = butter.compute_coeffs(cutoff=20, type_filter='highpass')
    b_lp, a_lp = butter.compute_coeffs(cutoff=450, type_filter='lowpass')
    
    print(f"[-] Applying Manual BPF Butterworth Order 2 (20-450 Hz) to {len(segments)} segments...")
    
    processed_segments = []
    
    for seg in segments:
        # Create a copy of the segment dictionary to avoid modifying the original
        new_seg = seg.copy()
        
        # --- Process Gastrocnemius (GL) ---
        gl_raw = seg['gl_segment']
        # Ensure input is float for precision
        gl_raw = np.array(gl_raw, dtype=float)
        
        gl_hp = butter.filtfilt_manual(b_hp, a_hp, gl_raw) # Apply HPF first
        gl_bpf = butter.filtfilt_manual(b_lp, a_lp, gl_hp)  # Apply LPF next
        new_seg['gl_filtered'] = gl_bpf
        
        # --- Process Vastus Lateralis (VL) ---
        vl_raw = seg['vl_segment']
        vl_raw = np.array(vl_raw, dtype=float)
        
        vl_hp = butter.filtfilt_manual(b_hp, a_hp, vl_raw)
        vl_bpf = butter.filtfilt_manual(b_lp, a_lp, vl_hp)
        new_seg['vl_filtered'] = vl_bpf
        
        processed_segments.append(new_seg)
        
    return processed_segments

def plot_filtered_comparison(segments, cycle_idx=0):
    """
    Plots a comparison between Raw and Filtered signals for a specific cycle.
    """
    if not segments:
        return
        
    seg = segments[cycle_idx]
    time = seg['time']
    # Normalize time to start at 0 for plotting
    t_plot = time - time[0]
    
    plt.figure(figsize=(10, 8))
    
    # Plot GL
    plt.subplot(2, 1, 1)
    plt.plot(t_plot, seg['gl_segment'], color='lightgray', label='Raw GL')
    plt.plot(t_plot, seg['gl_filtered'], color='blue', linewidth=1.5, label='Filtered GL (20-450Hz)')
    plt.title(f"Filter Result - Cycle {seg['cycle_id']} (Gastrocnemius)")
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Plot VL
    plt.subplot(2, 1, 2)
    plt.plot(t_plot, seg['vl_segment'], color='lightgray', label='Raw VL')
    plt.plot(t_plot, seg['vl_filtered'], color='red', linewidth=1.5, label='Filtered VL (20-450Hz)')
    plt.title(f"Filter Result - Cycle {seg['cycle_id']} (Vastus)")
    plt.xlabel('Time (s) from Heel Strike')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --- Block for standalone testing ---
if __name__ == "__main__":
    # Simulate dummy data (Sinusoid + Noise)
    fs_dummy = 2000
    t = np.linspace(0, 1, fs_dummy)
    
    # Signal: 50Hz (Main) + 5Hz (Artifact/Movement) + 800Hz (High freq Noise)
    # Expected result: 5Hz removed (HPF 20Hz), 800Hz removed (LPF 450Hz), 50Hz remains.
    raw_signal = np.sin(2 * np.pi * 50 * t) + \
                 2.0 * np.sin(2 * np.pi * 5 * t) + \
                 0.5 * np.sin(2 * np.pi * 800 * t)
                 
    dummy_seg = [{
        'cycle_id': 1,
        'fs': fs_dummy,
        'time': t,
        'gl_segment': raw_signal,
        'vl_segment': raw_signal # Use same signal for testing
    }]
    
    # Run Filter
    filtered_res = apply_bpf(dummy_seg)
    
    # Plot Verification
    plt.figure(figsize=(10,4))
    plt.plot(t, raw_signal, label='Raw (5Hz + 50Hz + 800Hz)', alpha=0.5)
    plt.plot(t, filtered_res[0]['gl_filtered'], label='BPF Output (Should be ~50Hz)', color='red')
    plt.legend()
    plt.title("Validation of Manual Butterworth Filter")
    plt.show()