# API mensure compatibility with GUI.py:
#   compute_stft(x, fs=2000, nperseg=None, noverlap=None, nfft=None, window='hann',
#                scaling='density', mode='magnitude')

import numpy as np
# 1. Manual Window Implementations
def _get_manual_window(window_name, N):
    """
    Generates window coefficients using raw mathematical formulas.
    Formula reference: https://en.wikipedia.org/wiki/Window_function
    """
    # Create index array n = 0..N-1
    n = np.arange(N, dtype=float)
    
    if window_name is None:
        return np.ones(N, dtype=float)
    
    if isinstance(window_name, str):
        w_name = window_name.lower()
        
        if 'hann' in w_name:
            # Formula: 0.5 - 0.5 * cos(2*pi*n / (N-1))
            if N == 1: return np.ones(1)
            return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (N - 1))
            
        elif 'hamming' in w_name:
            # Formula: 0.54 - 0.46 * cos(2*pi*n / (N-1))
            if N == 1: return np.ones(1)
            return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (N - 1))
            
        elif 'blackman' in w_name:
            # Formula: 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
            if N == 1: return np.ones(1)
            term1 = 0.42
            term2 = 0.5 * np.cos(2.0 * np.pi * n / (N - 1))
            term3 = 0.08 * np.cos(4.0 * np.pi * n / (N - 1))
            return term1 - term2 + term3
            
        elif w_name in ('rect', 'box', 'ones', 'rectangular'):
            return np.ones(N, dtype=float)
        else:
            # Fallback to hann if unknown
            if N == 1: return np.ones(1)
            return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (N - 1))
            
    # If user passed a custom array
    arr = np.asarray(window_name, dtype=float)
    if arr.size != N:
        raise ValueError(f"Window length {arr.size} does not match nperseg {N}")
    return arr

# 2. Manual FFT Implementations
def _is_power_of_two(n):
    return (n != 0) and ((n & (n - 1)) == 0)

def _manual_fft_recursive(x):
    """
    Implementation of the Cooley-Tukey Radix-2 Decimation-in-Time FFT algorithm
    x: 1D numpy array (complex or float)
    Returns: 1D numpy array (complex)
    """
    N = x.shape[0]
    
    # Base case
    if N <= 1:
        return x
    
    # Recursive divide
    even = _manual_fft_recursive(x[0::2])
    odd  = _manual_fft_recursive(x[1::2])
    
    # Combine (Butterfly operations)
    # Factor = exp(-j * 2 * pi * k / N)
    k = np.arange(N // 2)
    factor = np.exp(-2j * np.pi * k / N)
    
    return np.concatenate([even + factor * odd, even - factor * odd])

def _manual_dft_slow(x):
    """
    X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2*pi * k * n / N)
    Used if N is not a power of 2 (since Radix-2 FFT requires 2^k)
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # Create the exponential basis matrix
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def _manual_rfft(segment, nfft):
    """
    1. Pads segment to length nfft
    2. Computes full complex FFT (using Cooley-Tukey if 2^k, else DFT)
    3. Returns only the positive frequencies (0 to Nyquist)
    """
    # 1. Pad or crop
    if len(segment) < nfft:
        padded = np.zeros(nfft, dtype=float)
        padded[:len(segment)] = segment
        x_in = padded
    else:
        x_in = segment[:nfft]

    # 2. Compute full FFT
    if _is_power_of_two(nfft):
        X_full = _manual_fft_recursive(x_in)
    else:
        # If nfft is not power of 2, recursive FFT fails. Use definition.
        X_full = _manual_dft_slow(x_in)

    # 3. Slice for rfft (N/2 + 1 bins)
    # For real input, the negative frequencies are symmetric conjugates.
    # only need indices 0 to nfft//2.
    return X_full[:nfft//2 + 1]

def _manual_rfftfreq(nfft, d=1.0):
    """
    Manual generation of frequency bins.
    freq = k / (nfft * d) for k = 0 ... nfft/2
    """
    val = 1.0 / (nfft * d)
    N = nfft // 2 + 1
    return np.arange(0, N, dtype=float) * val

# 3. Main STFT Function
def compute_stft(x, fs=2000, nperseg=None, noverlap=None, nfft=None,
                 window='hann', scaling='density', mode='magnitude'):
    """
    Returns:
       Sxx: magnitude/power matrix (n_freqs, n_times)
       freqs: frequency array
       times: time array
       method: string identifier
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size

    if N == 0:
        return np.zeros((0, 0)), np.zeros((0,)), np.zeros((0,)), "STFT (Manual)"

    # Parameter Validation 
    if nperseg is None:
        nperseg = min(256, max(64, max(1, N // 8)))
    
    if nfft is None:
        nfft = max(256, nperseg)
    
    if noverlap is None:
        noverlap = int(nperseg // 2)
    if noverlap >= nperseg:
        noverlap = nperseg // 2

    nperseg = int(max(1, nperseg))
    nfft = int(max(1, nfft))
    noverlap = int(max(0, noverlap))
    hop = nperseg - noverlap
    if hop <= 0:
        hop = 1

    # Generate Window
    # Manual window creation
    try:
        win = _get_manual_window(window, nperseg)
    except Exception as e:
        raise ValueError(f"Invalid window: {e}")

    # Padding Logic
    if N <= nperseg:
        n_frames = 1
    else:
        n_frames = int(np.ceil((N - noverlap) / float(hop)))

    pad_len = (n_frames - 1) * hop + nperseg
    if pad_len > N:
        x_padded = np.concatenate([x, np.zeros(pad_len - N, dtype=float)])
    else:
        x_padded = x[:pad_len]

    # Main STFT Loop
    frames = []
    times = []
    
    # Iterate through signal
    for i in range(0, pad_len - nperseg + 1, hop):
        # 1. Apply Window
        seg = x_padded[i : i + nperseg] * win
        
        # 2. Compute Manual FFT
        # Returns complex numbers
        X_complex = _manual_rfft(seg, nfft)
        
        # 3. Compute Magnitude
        # Manual abs: sqrt(real^2 + imag^2)
        mag = np.sqrt(X_complex.real**2 + X_complex.imag**2)
        
        if mode == 'power':
            mag = mag ** 2
            
        frames.append(mag)
        
        # Center time calculation
        center = (i + (nperseg / 2.0)) / float(fs)
        times.append(center)

    # Formatting Output
    if len(frames) == 0:
        freqs = _manual_rfftfreq(nfft, d=1.0/fs)
        return np.zeros((freqs.size, 0)), freqs, np.array([], dtype=float), "STFT (Manual)"

    # Stack frames (Frequency x Time)
    S = np.column_stack(frames)
    
    # Generate Frequency Axis
    freqs = _manual_rfftfreq(nfft, d=1.0/fs)

    # Scaling
    # Manual scaling implementation
    if scaling == 'density':
        # sum of squared window samples
        w_energy = np.sum(win ** 2)
        if w_energy <= 0:
            w_energy = 1.0
        S = S / np.sqrt(w_energy)
    elif scaling == 'spectrum':
        pass

    return S, freqs, np.asarray(times, dtype=float), "STFT (Manual)"