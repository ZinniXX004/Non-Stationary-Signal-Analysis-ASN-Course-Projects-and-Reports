# Simple STFT implementation using only NumPy (no SciPy dependency).
# Returns a time-frequency magnitude (or power) matrix compatible with downstream code.
#
# API:
#   compute_stft(x, fs=2000, nperseg=None, noverlap=None, nfft=None, window='hann',
#                scaling='density', mode='magnitude')
# Returns:
#   Sxx: ndarray (n_freqs, n_times)  -- magnitude or power spectrogram
#   freqs: ndarray (n_freqs,)        -- frequency bin centers in Hz (ascending)
#   times: ndarray (n_times,)        -- times of frame centers in seconds
#   method: str                      -- textual method description

import numpy as np

def _get_window_array(window, nperseg):
    """Return a 1D window array of length nperseg.
    Supported string names: 'hann', 'hamming', 'rect', 'box', 'blackman'
    If window is array-like and matches length, it's returned as float array.
    """
    if window is None:
        return np.ones(nperseg, dtype=float)
    if isinstance(window, str):
        w = window.lower()
        if 'hann' in w:
            return np.hanning(nperseg).astype(float)
        if 'hamming' in w:
            return np.hamming(nperseg).astype(float)
        if 'blackman' in w:
            return np.blackman(nperseg).astype(float)
        if w in ('rect', 'box', 'ones', 'rectangular'):
            return np.ones(nperseg, dtype=float)
        # fallback to hann
        return np.hanning(nperseg).astype(float)
    # array-like
    arr = np.asarray(window, dtype=float)
    if arr.ndim != 1:
        raise ValueError("window must be 1D array or a supported string name")
    if arr.size != nperseg:
        raise ValueError(f"window length {arr.size} does not match nperseg {nperseg}")
    return arr

def compute_stft(x, fs=2000, nperseg=None, noverlap=None, nfft=None,
                 window='hann', scaling='density', mode='magnitude'):
    """
    Parameters:
    x : array-like
        1D input signal.
    fs : int
        Sampling frequency (Hz).
    nperseg : int or None
        Window length (samples). If None -> min(256, max(64, len(x)//8))
    noverlap : int or None
        Overlap (samples). If None -> nperseg // 2
    nfft : int or None
        FFT length. If None -> max(256, nperseg)
    window : str or array-like
        Window description (supported strings: 'hann','hamming','rect','blackman') or array of length nperseg.
    scaling : str
        'density' or 'spectrum' (kept for API compatibility; only simple normalization applied).
    mode : str
        'magnitude' (default) returns |STFT|, 'power' returns |STFT|^2.
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size

    if N == 0:
        return np.zeros((0, 0)), np.zeros((0,)), np.zeros((0,)), "STFT (numpy)"

    # sensible defaults
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

    # window
    try:
        win = _get_window_array(window, nperseg)
    except Exception as e:
        raise ValueError(f"Invalid window: {e}")

    # number of frames (include last partial frame by padding)
    if N <= nperseg:
        n_frames = 1
    else:
        n_frames = int(np.ceil((N - noverlap) / float(hop)))

    pad_len = (n_frames - 1) * hop + nperseg
    if pad_len > N:
        x = np.concatenate([x, np.zeros(pad_len - N, dtype=float)])
    elif pad_len < N:
        # should not happen, but safe-guard
        x = x[:pad_len]

    frames = []
    times = []
    for i in range(0, pad_len - nperseg + 1, hop):
        seg = x[i:i + nperseg] * win
        X = np.fft.rfft(seg, n=nfft)
        mag = np.abs(X)
        if mode == 'power':
            mag = mag ** 2
        frames.append(mag)
        # center time of the window
        center = (i + (nperseg / 2.0)) / float(fs)
        times.append(center)

    if len(frames) == 0:
        # fallback: empty
        freqs = np.fft.rfftfreq(nfft, d=1.0 / float(fs))
        return np.zeros((freqs.size, 0)), freqs, np.array([], dtype=float), "STFT (numpy)"

    S = np.column_stack(frames)  # shape: (n_freqs, n_times)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(fs))

    # Simple scaling normalization (not full PSD scaling of scipy)
    if scaling == 'density':
        # normalize by window energy and sample rate to get comparable magnitudes for different windows
        w_energy = np.sum(win ** 2)
        if w_energy <= 0:
            w_energy = 1.0
        S = S / np.sqrt(w_energy)
    elif scaling == 'spectrum':
        # no extra scaling beyond magnitude
        pass

    # ensure ascending freqs (rfft already gives ascending non-negative)
    return S, freqs, np.asarray(times, dtype=float), "STFT (numpy)"
