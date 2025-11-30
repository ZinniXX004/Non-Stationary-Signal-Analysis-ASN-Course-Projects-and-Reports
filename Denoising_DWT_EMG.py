import numpy as np
import matplotlib.pyplot as plt

class ManualWindow:
    @staticmethod
    def get_window(window_type, N, beta=14.0):
        """
        Generates a window array of length N
        
        Arguments:
            window_type (str): 'Rectangular', 'Hanning', 'Hamming', 'Blackman', 'Triangular', 'Kaiser'
            N (int): Length of the window
            beta (float): Shape parameter for Kaiser window (default 14.0)
        """
        if N <= 1:
            return np.ones(N)
        
        n = np.arange(N)
        
        if window_type == 'Rectangular':
            return np.ones(N)
            
        elif window_type == 'Hanning':
            # w[n] = 0.5 - 0.5 * cos(2*pi*n / (N-1))
            return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
            
        elif window_type == 'Hamming':
            # w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
            return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
            
        elif window_type == 'Blackman':
            # w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
            return (0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 
                    0.08 * np.cos(4 * np.pi * n / (N - 1)))
                    
        elif window_type == 'Triangular':
            # w[n] = 1 - | (n - (N-1)/2) / ((N-1)/2) |
            return 1 - np.abs((n - (N - 1) / 2) / ((N - 1) / 2))
            
        elif window_type == 'Kaiser':
            # Requires Modified Bessel Function of first kind, order 0 (I0)
            # Formula: w[n] = I0(beta * sqrt(1 - ((2n/(N-1)) - 1)^2)) / I0(beta)
            
            # 1. Define manual I0 function (Taylor Series approximation)
            def manual_i0(x):
                sum_val = 1.0
                term = 1.0
                # 25 iterations is sufficient for convergence
                for k in range(1, 25):
                    term *= (x**2) / (4 * (k**2))
                    sum_val += term
                return sum_val

            # 2. Calculate coefficients
            alpha = (N - 1) / 2.0
            denom = manual_i0(beta)
            window = np.zeros(N)
            
            for i in range(N):
                term = np.sqrt(1 - ((i - alpha) / alpha)**2)
                # Handle potential tiny negative numbers in sqrt due to float precision
                if np.isnan(term): term = 0
                window[i] = manual_i0(beta * term) / denom
                
            return window
            
        else:
            # Default to Rectangular if unknown
            print(f"[!] Unknown window type '{window_type}', using Rectangular.")
            return np.ones(N)

class ManualDWT:
    # Uses the Mallat Algorithm (Filter Bank) with Daubechies 4 (db4) coefficients.
    def __init__(self):
        # Decomposition Low-Pass Coefficients (L_D) for db4
        self.ld_coeffs = np.array([
            0.230377813309, 0.714846570553, 0.630880767930, -0.027983769417,
            -0.187034811719, 0.030841381836, 0.032883011667, -0.010597401785
        ])
        
        # High-Pass Decomposition (H_D): Alternating Flip of L_D
        # H_D[n] = (-1)^n * L_D[N-1-n]
        self.hd_coeffs = np.zeros_like(self.ld_coeffs)
        N = len(self.ld_coeffs)
        for i in range(N):
            self.hd_coeffs[i] = ((-1)**i) * self.ld_coeffs[N-1-i]
            
        # Reconstruction Coefficients (Inverse DWT)
        # L_R (Low Recon) = Reverse L_D
        self.lr_coeffs = self.ld_coeffs[::-1]
        # H_R (High Recon) = Reverse H_D
        self.hr_coeffs = self.hd_coeffs[::-1]

    def downsample(self, arr):
        # Take every even index element (Decimation by 2)
        return arr[1::2]

    def upsample(self, arr):
        # Insert zeros between elements (Interpolation)
        up = np.zeros(len(arr) * 2)
        up[::2] = arr
        return up

    def pad_signal(self, signal):
        # Symmetric padding to handle convolution edge effects
        pad_len = len(self.ld_coeffs) // 2
        return np.pad(signal, (pad_len, pad_len), mode='edge')

    def dwt_decomposition(self, signal, levels=8):
        coeffs = []
        current_appx = signal
        
        for i in range(levels):
            # Convolve using 'valid' mode after padding
            padded = self.pad_signal(current_appx)
            
            cA = np.convolve(padded, self.ld_coeffs, mode='valid')
            cD = np.convolve(padded, self.hd_coeffs, mode='valid')
            
            # Downsampling
            cA_down = self.downsample(cA)
            cD_down = self.downsample(cD)
            
            coeffs.append(cD_down) # Store Detail
            current_appx = cA_down # Approximation continues to next level
            
        coeffs.append(current_appx) # Store final Approximation
        
        # Return in standard order: [cA8, cD8, cD7, ..., cD1]
        return coeffs[::-1]

    def idwt_reconstruction(self, coeffs):
        current_appx = coeffs[0]
        details = coeffs[1:]
        
        for cD in details:
            # Ensure the approximation vector matches the detail vector length
            # before upsampling, as they must align for the next level
            if len(current_appx) != len(cD):
                if len(current_appx) > len(cD):
                    current_appx = current_appx[:len(cD)]
                else:
                    pad_amt = len(cD) - len(current_appx)
                    current_appx = np.pad(current_appx, (0, pad_amt), 'edge')
            
            # 1. Upsampling
            up_appx = self.upsample(current_appx)
            up_detail = self.upsample(cD)
            
            # 2. Convolution with Reconstruction Filters (Full Mode)
            rec_appx = np.convolve(up_appx, self.lr_coeffs, mode='full')
            rec_detail = np.convolve(up_detail, self.hr_coeffs, mode='full')
            
            # 3. Summation and Trimming
            rec_sum = rec_appx + rec_detail
            
            target_len = len(up_appx)
            filter_len = len(self.ld_coeffs)
            
            # Empirical trimming indices for db4 alignment
            start_idx = filter_len - 1
            end_idx = start_idx + target_len
            
            if end_idx > len(rec_sum):
                rec_sum = rec_sum[start_idx:]
            else:
                rec_sum = rec_sum[start_idx:end_idx]
            
            current_appx = rec_sum
            
        return current_appx

def soft_thresholding(x, threshold):
    # y = sign(x) * max(0, |x| - T)
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def denoise_dwt(segments, window_type='Rectangular'):
    """
    Main function for DWT Denoisings
    
    Steps:
    1. Apply selected Window Function
    2. Decompose using DWT (db4, level 8)
    3. Estimate Noise using MAD on Detail Coeffs Level 1
    4. Apply Soft Thresholding
    5. Reconstruct Signal
    """
    if not segments:
        return []
    
    dwt = ManualDWT()
    print(f"[-] Applying Denoising DWT (Window: {window_type}, db4, level 8)...")
    
    processed_segments = []
    
    for seg in segments:
        new_seg = seg.copy()
        
        # Prioritize 'gl_filtered', fallback to 'gl_segment' (raw)
        if 'gl_filtered' in seg:
            gl_input = seg['gl_filtered']
        else:
            gl_input = seg['gl_segment']

        if 'vl_filtered' in seg:
            vl_input = seg['vl_filtered']
        else:
            vl_input = seg['vl_segment']
        
        # Generate Window
        N = len(gl_input)
        window = ManualWindow.get_window(window_type, N)
        
        # Process GL
        try:
            # 1. Apply Window
            gl_windowed = gl_input * window
            
            # 2. Decomposition
            coeffs_gl = dwt.dwt_decomposition(gl_windowed, levels=8)
            
            # 3. Threshold Calculation (Donoho's Method)
            cD1_gl = coeffs_gl[-1]
            # MAD = median(|x - median(x)|), assuming mean 0 -> median(|x|)
            sigma_gl = np.median(np.abs(cD1_gl)) / 0.6745
            threshold_gl = sigma_gl * np.sqrt(2 * np.log(len(gl_windowed)))
            
            # 4. Thresholding
            new_coeffs_gl = [coeffs_gl[0]] # Keep Approximation
            for detail in coeffs_gl[1:]:
                new_coeffs_gl.append(soft_thresholding(detail, threshold_gl))
                
            # 5. Reconstruction
            gl_denoised = dwt.idwt_reconstruction(new_coeffs_gl)
            
            # 6. Final Resizing (Safety check)
            if len(gl_denoised) > len(gl_input):
                gl_denoised = gl_denoised[:len(gl_input)]
            elif len(gl_denoised) < len(gl_input):
                gl_denoised = np.pad(gl_denoised, (0, len(gl_input) - len(gl_denoised)), 'edge')
                
            new_seg['gl_denoised'] = gl_denoised
            
        except Exception as e:
            print(f"[!] Error denoising GL cycle {seg['cycle_id']}: {e}")
            new_seg['gl_denoised'] = gl_input # Fallback

        # Process VL
        try:
            # 1. Apply Window
            vl_windowed = vl_input * window
            
            # 2. Decomposition
            coeffs_vl = dwt.dwt_decomposition(vl_windowed, levels=8)
            
            # 3. Threshold Calculation
            cD1_vl = coeffs_vl[-1]
            sigma_vl = np.median(np.abs(cD1_vl)) / 0.6745
            threshold_vl = sigma_vl * np.sqrt(2 * np.log(len(vl_windowed)))
            
            # 4. Thresholding
            new_coeffs_vl = [coeffs_vl[0]]
            for detail in coeffs_vl[1:]:
                new_coeffs_vl.append(soft_thresholding(detail, threshold_vl))
            
            # 5. Reconstruction
            vl_denoised = dwt.idwt_reconstruction(new_coeffs_vl)
            
            if len(vl_denoised) > len(vl_input):
                vl_denoised = vl_denoised[:len(vl_input)]
            elif len(vl_denoised) < len(vl_input):
                vl_denoised = np.pad(vl_denoised, (0, len(vl_input) - len(vl_denoised)), 'edge')
                
            new_seg['vl_denoised'] = vl_denoised
            
        except Exception as e:
            print(f"[!] Error denoising VL cycle {seg['cycle_id']}: {e}")
            new_seg['vl_denoised'] = vl_input 
        
        processed_segments.append(new_seg)
        
    return processed_segments

def plot_denoising_comparison(segments, cycle_idx=0):
    if not segments: return
    
    seg = segments[cycle_idx]
    t = seg['time']
    t = t - t[0]
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    # Safer check for input signal
    if 'gl_filtered' in seg:
        input_sig = seg['gl_filtered']
        label = 'Filtered Input'
    else:
        input_sig = seg['gl_segment']
        label = 'Raw Input'

    plt.plot(t[:len(input_sig)], input_sig, color='lightgray', label=label)
    plt.plot(t[:len(seg['gl_denoised'])], seg['gl_denoised'], color='blue', linewidth=1.5, label='DWT Denoised')
    plt.title(f"DWT Denoising Result - GL (Cycle {seg['cycle_id']})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    if 'vl_filtered' in seg:
        input_sig_vl = seg['vl_filtered']
        label_vl = 'Filtered Input'
    else:
        input_sig_vl = seg['vl_segment']
        label_vl = 'Raw Input'

    plt.plot(t[:len(input_sig_vl)], input_sig_vl, color='lightgray', label=label_vl)
    plt.plot(t[:len(seg['vl_denoised'])], seg['vl_denoised'], color='red', linewidth=1.5, label='DWT Denoised')
    plt.title(f"DWT Denoising Result - VL (Cycle {seg['cycle_id']})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# <<<< Standalone Testing Block >>>>
if __name__ == "__main__":
    # Dummy Data Generation
    t = np.linspace(0, 1, 2000)
    clean = np.sin(2*np.pi*10*t) * np.exp(-2*t) 
    noise = np.random.normal(0, 0.2, len(t))
    noisy_sig = clean + noise
    
    # Updated Dummy Data Structure (Includes both segment and filtered keys)
    dummy_segs = [{
        'cycle_id': 99,
        'time': t,
        'gl_segment': noisy_sig,   # Raw
        'gl_filtered': noisy_sig,  # Filtered
        'vl_segment': noisy_sig,   # Raw
        'vl_filtered': noisy_sig,  # Filtered
        'fs': 2000
    }]
    
    # Test with Hanning Window
    res = denoise_dwt(dummy_segs, window_type='Hanning')
    
    plt.figure(figsize=(10,4))
    plt.plot(noisy_sig, color='gray', alpha=0.5, label='Noisy Input')
    plt.plot(res[0]['gl_denoised'], color='green', label='Denoised (Hanning + DWT)')
    plt.plot(clean, color='black', linestyle='--', label='Clean Ground Truth')
    plt.legend()
    plt.title("Manual DWT + Hanning Window Test")
    plt.show()