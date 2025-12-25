"""
CWT module with manual implementation of the Continuous Wavelet Transform equations.
Primary method uses direct mathematical formulas (Morlet Wavelet + Convolution).
Secondary backends (Scipy/PyWavelets) are available but optional.

Public API:
    compute_cwt(signal, fs, fmin=20, fmax=500, n_freqs=120, backend='auto', **kwargs)
    compute_threshold_and_cogs(scalogram, freqs, times, s1=0.6, s2=0.1, ...)
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np

# Optional libraries for secondary backends
try:
    import pywt
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False

try:
    import scipy.signal as _scipy_signal  
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

__all__ = ["compute_cwt", "compute_threshold_and_cogs"]

# 1. Manual Math Helper Functions
def _get_morlet_kernel(M, w0=5.0):
    """
    Formula: psi(t) = pi^(-0.25) * exp(j*w0*t) * exp(-t^2 / 2)
    """
    # Create time vector centered at 0
    t = np.arange(-M // 2, M // 2)
    # Standardize time for formulation (assuming standard deviation scaling happens outside or via window length)
    # Treat 't' as normalized time units relative to the window width
    # Commonly, for window width M, cover approx -4 to +4 sigma.
    sigma = M / 8.0 
    t_scaled = t / sigma

    # Constant factor pi^(-1/4)
    c = np.pi ** (-0.25)
    
    # Gaussian envelope: exp(-t^2 / 2)
    envelope = np.exp(-0.5 * t_scaled**2)
    
    # Oscillatory part: exp(j * w0 * t)
    # Using Euler's formula: exp(jx) = cos(x) + j*sin(x)
    oscillation = np.cos(w0 * t_scaled) + 1j * np.sin(w0 * t_scaled)
    
    psi = c * envelope * oscillation
    return psi

def _manual_convolution(signal, kernel):
    """
    Performs 1D convolution manually (equivalent to mathematical sliding sum)
    y[n] = sum_k ( x[k] * h[n-k] )
    """
    return np.convolve(signal, kernel, mode='same')

def _compute_cwt_manual(signal, fs, fmin, fmax, n_freqs, w0=5.0):
    """
    Manual implementation of CWT using loop over scales and convolution.
    
    1. Define Frequencies.
    2. Convert Frequency -> Scale (a).
    3. Generate Wavelet psi(t) for that scale.
    4. Convolve signal with psi.
    5. Compute Power |W|^2.
    """
    signal = np.asarray(signal)
    N = signal.size
    dt = 1.0 / fs
    
    # 1. Generate Frequency Vector (Linear Space)
    freqs = np.linspace(fmin, fmax, n_freqs)
    
    # 2. Convert to Scales
    # Relation for Morlet: f = (w0 * fs) / (2 * pi * a)  (approx)
    # Thus: a = (w0 * fs) / (2 * pi * f)
    # However, to match standard definitions where w0 is usually approx 5-6 (omega0)
    scales = (w0 * fs) / (2.0 * np.pi * freqs)
    
    # Prepare output matrix
    # Scalogram: Rows = Frequencies, Cols = Time
    cwt_matrix = np.zeros((n_freqs, N), dtype=float)
    
    # 3. and 4. Loop over scales (Direct implementation)
    for i, a in enumerate(scales):
        # Determine effective support of the wavelet at this scale
        # We need a window large enough to hold the Gaussian envelope.
        # usually 6 to 8 standard deviations. 
        # width ~ a * base_width. 
        window_size = int(min(10 * a, N)) # Limit window to signal size
        if window_size % 2 == 0: 
            window_size += 1 # Ensure odd length for centering
        
        # Generate Scaled Morlet Kernel manually
        # In CWT formula: psi_a,b(t) = (1/sqrt(a)) * psi((t-b)/a)
        # The 'a' scaling in time is handled by the window size and sampling.
        # The '1/sqrt(a)' is the energy normalization.
        
        # Base kernel (time scaling handles the 'a' in denominator)
        kernel = _get_morlet_kernel(window_size, w0=w0)
        
        # Energy Normalization (1/sqrt(a))
        # Note: In discrete convolution, amplitude scaling requires care
        # Normalize so energy is consistent.
        norm_factor = 1.0 / np.sqrt(a)
        kernel = kernel * norm_factor
        
        # Convolve
        # The output of convolution is the wavelet coefficient W(a,b)
        # Use 'same' to keep time alignment
        conv_res = _manual_convolution(signal, kernel)
        
        # 5. Compute Power (Magnitude Squared)
        # |W(a,b)|^2 = Real^2 + Imag^2
        power = (conv_res.real ** 2) + (conv_res.imag ** 2)
        cwt_matrix[i, :] = power

    times = np.arange(N) * dt
    return cwt_matrix, freqs, times

# 2. Main API Function
def compute_cwt(signal: np.ndarray,
                fs: int,
                fmin: float = 20.0,
                fmax: float = 500.0,
                n_freqs: int = 120,
                backend: str = "auto",
                **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Main entry point. 
    Default backend ('auto') maps to 'manual' implementation of math formulas
    """
    sig = np.asarray(signal).ravel()
    n = sig.size
    if n == 0:
        return np.zeros((0,0)), np.array([]), np.array([]), "empty"

    # Normalize backend selection
    b_end = backend.lower()
    
    # PRIMARY OPTION: MANUAL IMPLEMENTATION
    if b_end == "auto" or b_end == "manual" or b_end == "pascal":
        try:
            # Call the manual math function
            power, freqs, times = _compute_cwt_manual(sig, fs, fmin, fmax, n_freqs)
            
            # Normalize to [0..1] for consistency with GUI visualization
            maxv = np.nanmax(power) if power.size else 0.0
            if maxv > 0:
                power = power / (maxv + 1e-12)
                
            return power, freqs, times, "manual-morlet"
        except Exception as e:
            # Fallback if manual fails (unlikely with basic numpy)
            print(f"Manual CWT failed: {e}. Trying fallback.")
            if _HAS_SCIPY: b_end = "scipy"
            elif _HAS_PYWT: b_end = "pywt"

    # SECONDARY OPTION: PyWavelets
    if b_end == "pywt" and _HAS_PYWT:
        try:
            wavelet = kwargs.get("wavelet", "morl")
            central = pywt.central_frequency(wavelet)
            freqs = np.linspace(fmin, fmax, n_freqs)
            scales = (central * float(fs)) / freqs
            coeffs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1.0/float(fs))
            power = (np.abs(coeffs) ** 2)
            times = np.arange(n) / float(fs)
            maxv = np.nanmax(power) if power.size else 0.0
            if maxv > 0:
                power = power / (maxv + 1e-12)
            return power, freqs, times, "pywt-morl"
        except Exception:
            pass # Fall through

    # SECONDARY OPTION: Scipy CWT
    if b_end == "scipy" and _HAS_SCIPY:
        try:
            from scipy.signal import cwt, morlet2
            # Map frequencies to widths
            # w = 5 assumed for morlet2 default
            w0 = 5
            widths = (w0 * fs) / (2 * np.pi * np.linspace(fmin, fmax, n_freqs))
            
            coeffs = cwt(sig, morlet2, widths, w=w0)
            
            # Re-map freqs explicitly
            freqs_out = np.linspace(fmin, fmax, n_freqs)
            power = (np.abs(coeffs) ** 2)
            times = np.arange(n) / float(fs)
            
            # Flip if needed (cwt usually returns high freq at small width, depends on width order)
            # We computed widths from linspace freqs, so index 0 is fmin (large width)
            
            maxv = np.nanmax(power) if power.size else 0.0
            if maxv > 0:
                power = power / (maxv + 1e-12)
            return power, freqs_out, times, "scipy-cwt"
        except Exception:
            pass

    return np.zeros((0,0)), np.array([]), np.array([]), "error-no-backend"

# 3. Center of Gravity & Thresholding (Manual Math) ------>>> # UNUSED, but similar to the one in Threshold_Plot_CoG.py
def compute_threshold_and_cogs(scalogram: np.ndarray,
                               freqs: np.ndarray,
                               times: np.ndarray,
                               s1: float = 0.6,
                               s2: float = 0.10,
                               min_area: Optional[int] = None,
                               keep_top: int = 3) -> Dict[str, Any]:
    """
    Calculates the Center of Gravity (CoG) for S1 and S2 areas.
    CoG_x = sum(x * mass) / sum(mass)
    """
    
    # 1. Thresholding Helper
    def _manual_threshold(scal, ratio):
        if scal is None or scal.size == 0:
            return np.zeros_like(scal, dtype=bool)
        
        # Find global peak manually
        peak = 0.0
        # Flatten to find max
        flat = scal.ravel()
        for v in flat:
            if v > peak: peak = v
            
        if peak <= 0:
            return np.zeros_like(scal, dtype=bool)
        
        thr = ratio * peak
        # Boolean mask
        mask = (scal >= thr)
        return mask

    # 2. Center of Gravity Helper
    def _manual_cog(scal, freq_axis, time_axis, mask):
        # Computes weighted average of Time and Frequency.
        if scal is None or scal.size == 0:
            return None
            
        # Extract Energy where mask is True
        # E_masked = scal * mask
        
        total_mass = 0.0
        moment_time = 0.0
        moment_freq = 0.0
        
        n_freqs, n_times = scal.shape
        
        # Double loop implementation of formula:
        # T_cg = Sum(t * E(f,t)) / Sum(E(f,t))
        # F_cg = Sum(f * E(f,t)) / Sum(E(f,t))
        
        # Create grid of values
        # E[i, j] is energy at freq i, time j
        
        # Using basic numpy sum (allowed)
        E = scal * mask
        total_mass = np.sum(E)
        
        if total_mass <= 1e-12:
            return None
            
        # Time Moment
        # Sum(E[i,j] * times[j])
        # Sum over freq axis first -> 1D array of energy per time step
        energy_per_time = np.sum(E, axis=0) 
        moment_time = np.sum(energy_per_time * time_axis)
        
        # Freq Moment
        # Sum over time axis first -> 1D array of energy per freq bin
        energy_per_freq = np.sum(E, axis=1)
        moment_freq = np.sum(energy_per_freq * freq_axis)
        
        t_cog = moment_time / total_mass
        f_cog = moment_freq / total_mass
        
        return (float(t_cog), float(f_cog))

    # Execution
    mask1 = _manual_threshold(scalogram, s1)
    mask2 = _manual_threshold(scalogram, s2)
    
    cog1 = _manual_cog(scalogram, freqs, times, mask1)
    cog2 = _manual_cog(scalogram, freqs, times, mask2)
    
    return {'S1_mask': mask1, 'S2_mask': mask2, 'S1_cog': cog1, 'S2_cog': cog2}

# Backward compatibility wrapper for the old manual function if GUI calls it
def compute_cwt_pascal(*args, **kwargs):
    # Redirects to the manual CWT implementation
    sig = args[0]
    fs = args[1]
    # Extract defaults if not present
    fmin = kwargs.get('fmin', 20.0)
    fmax = kwargs.get('fmax', 500.0)
    # Estimate n_freqs from rows if passed, else default
    n_freqs = kwargs.get('row_count', 120) 
    
    return _compute_cwt_manual(sig, fs, fmin=fmin, fmax=fmax, n_freqs=int(n_freqs))