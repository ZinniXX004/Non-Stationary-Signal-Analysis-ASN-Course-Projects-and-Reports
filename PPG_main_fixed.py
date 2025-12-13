"""
-+-+-+ PPG ANALYSIS BACKEND (FIXED & ROBUST) +-+-+-
Author: Jeremia Manalu (Revisions by Assistant)

Updates:
1. Enforced integer types for all index arrays in 'analyze_signal_zero_crossing' 
   to prevent IndexError in GUI plotting.
2. Maintained 'Filter_CustomRef' and 'CubicSplineInterpolate' logic strict to Delphi.
3. Prepared HRV Analyzer to handle both Original and Downsampled inputs safely.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import math

# ==============================================================================
# 1. CORE MATHEMATICS & FILTERS
# ==============================================================================

def dirac_delta(k):
    """
    Fungsi Dirac Delta sederhana: 
    Return 1 jika k=0, selain itu 0.
    """
    return 1 if k == 0 else 0

def filter_custom_ref(signal, fs, low_cutoff, high_cutoff):
    """
    Implementasi Filter IIR (Infinite Impulse Response) manual.
    Struktur: Cascade LPF (Orde 2) -> HPF (Orde 1).
    Menggunakan koefisien analog-ke-digital spesifik.
    """
    signal = np.array(signal, dtype=float)
    N = len(signal)
    
    # Safety check untuk sinyal terlalu pendek
    if N < 3:
        return signal

    # Parameter Dasar
    Tm = 1.0 / fs
    WcLPF = 2 * np.pi * high_cutoff
    WcHPF = 2 * np.pi * low_cutoff

    # --- 1. Hitung Koefisien LPF (Low Pass Filter) ---
    denom_lpf = (4 / (Tm**2)) + (2 * np.sqrt(2) * WcLPF / Tm) + (WcLPF**2)
    
    if denom_lpf == 0:
        return signal # Hindari pembagian dengan nol

    LPFb1 = ((8 / (Tm**2)) - 2 * (WcLPF**2)) / denom_lpf
    LPFb2 = ((4 / (Tm**2)) - (2 * np.sqrt(2) * WcLPF / Tm) + (WcLPF**2)) / denom_lpf
    LPFa0 = (WcLPF**2) / denom_lpf
    LPFa1 = 2 * (WcLPF**2) / denom_lpf
    LPFa2 = LPFa0

    # --- 2. Hitung Koefisien HPF (High Pass Filter) ---
    denom_hpf = WcHPF + (2 / Tm)
    
    if denom_hpf == 0:
        return signal

    HPFa0 = (2 / Tm) / denom_hpf
    HPFa1 = -HPFa0
    HPFb1 = (WcHPF - (2 / Tm)) / denom_hpf

    # --- 3. Eksekusi LPF ---
    sig_lpf = np.zeros(N)
    sig_lpf[0] = signal[0]
    if N > 1:
        sig_lpf[1] = signal[1]
    
    for i in range(2, N):
        sig_lpf[i] = (LPFb1 * sig_lpf[i-1]) - (LPFb2 * sig_lpf[i-2]) + \
                     (LPFa0 * signal[i]) + (LPFa1 * signal[i-1]) + (LPFa2 * signal[i-2])

    # --- 4. Eksekusi HPF (Input diambil dari hasil LPF) ---
    result = np.zeros(N)
    # result[0] tetap 0 (inisialisasi filter)
    
    for i in range(1, N):
        result[i] = (HPFa0 * (sig_lpf[i] - sig_lpf[i-1])) - (HPFb1 * result[i-1])

    return result

def linear_detrend(signal):
    """
    Menghilangkan tren linear (garis lurus) dari sinyal.
    Penting sebelum melakukan analisis frekuensi (FFT/PSD).
    """
    y = np.array(signal)
    x = np.arange(len(y))
    if len(y) < 2:
        return y - np.mean(y)
    
    # Perhitungan Regresi Linear Manual (Least Squares)
    n = len(y)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)
    
    denominator = (n * sum_xx - sum_x * sum_x)
    if denominator == 0:
        return y - np.mean(y)
        
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    trend = slope * x + intercept
    return y - trend

def cubic_spline_interpolate(x_in, y_in, x_out):
    """
    Implementasi Manual Cubic Spline Interpolation.
    Diterjemahkan dari kode Delphi untuk memastikan hasil Frequency Domain identik.
    """
    n = len(x_in)
    if n < 2:
        return np.zeros_like(x_out)
    
    # Step 1: Hitung selisih h
    h = np.diff(x_in)
    
    # Step 2: Hitung alpha
    alpha = np.zeros(n - 1)
    for i in range(1, n - 1):
        if h[i] != 0 and h[i-1] != 0:
            alpha[i] = (3/h[i]) * (y_in[i+1] - y_in[i]) - (3/h[i-1]) * (y_in[i] - y_in[i-1])
            
    # Step 3: Selesaikan matriks Tridiagonal
    l = np.zeros(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    l[0] = 1.0
    
    for i in range(1, n - 1):
        l[i] = 2 * (x_in[i+1] - x_in[i-1]) - h[i-1] * mu[i-1]
        if l[i] != 0:
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
            
    l[n-1] = 1.0
    z[n-1] = 0.0
    
    # Step 4: Hitung Koefisien c, b, d
    c = np.zeros(n)
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    
    for i in range(n - 2, -1, -1):
        c[i] = z[i] - mu[i] * c[i+1]
        
    for i in range(n - 1):
        if h[i] != 0:
            d[i] = (c[i+1] - c[i]) / (3 * h[i])
            b[i] = (y_in[i+1] - y_in[i]) / h[i] - h[i] * (c[i+1] + 2 * c[i]) / 3
            
    # Step 5: Interpolasi nilai output (x_out)
    y_out = np.zeros_like(x_out)
    
    for i, xi in enumerate(x_out):
        # Cari segmen k (Linear search agar aman)
        k = 0
        while k < n - 2 and x_in[k+1] < xi:
            k += 1
            
        dx = xi - x_in[k]
        y_out[i] = y_in[k] + b[k] * dx + c[k] * (dx**2) + d[k] * (dx**3)
        
    return y_out

# ==============================================================================
# 2. SPECTRAL ANALYSIS (WELCH & FFT)
# ==============================================================================

def fft_from_scratch(signal):
    """
    Implementasi Recursive FFT (Radix-2).
    """
    x = np.asarray(signal, dtype=np.complex128)
    N = x.shape[0]
    
    if N <= 1: return x
    
    # Fallback ke numpy jika panjang bukan pangkat 2 (jarang terjadi karena padding)
    if N % 2 > 0: return np.fft.fft(x) 
    
    even = fft_from_scratch(x[::2])
    odd = fft_from_scratch(x[1::2])
    
    factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
    return np.concatenate([even + factor * odd, even - factor * odd])

def welch_from_scratch(signal_data, fs, segment_len=256, overlap_ratio=0.5):
    """
    Implementasi Welch's Method.
    Fitur: Hanning Window, Zero Padding (min 4096 poin) untuk resolusi tinggi.
    """
    x = np.asarray(signal_data, dtype=float)
    if len(x) < 4: return np.array([]), np.array([])
    
    nperseg = min(segment_len, len(x))
    noverlap = int(nperseg * overlap_ratio)
    step = nperseg - noverlap
    if step < 1: step = 1
    
    # Buat Hanning Window
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1)))
    sum_sq_win = np.sum(window**2)
    
    # Logika Zero Padding: Minimal 4096 point (Meniru Delphi)
    min_nfft = 4096
    nfft = 1 << (nperseg - 1).bit_length() # Next power of 2
    if nfft < min_nfft: nfft = min_nfft
    
    num_segments = (len(x) - noverlap) // step
    if num_segments < 1: num_segments = 1
    
    psd_accum = np.zeros(nfft)
    count = 0
    
    for i in range(num_segments):
        start = i * step
        if start + nperseg > len(x): break
        
        segment = x[start : start + nperseg] * window
        
        # Zero Padding (dilakukan otomatis oleh np.fft.fft jika n > len)
        fft_out = np.fft.fft(segment, n=nfft) 
        
        # Hitung Power Spectrum Segmen
        psd_seg = (np.abs(fft_out)**2) / (fs * sum_sq_win)
        psd_accum += psd_seg
        count += 1
        
    if count == 0: return np.array([]), np.array([])
    
    avg_psd = psd_accum / count
    
    # Ambil sisi positif spektrum (One-sided)
    half = nfft // 2
    freqs = np.fft.fftfreq(nfft, 1/fs)[:half]
    psd = avg_psd[:half]
    
    # Konservasi Energi (Kali 2 untuk komponen AC)
    if len(psd) > 1:
        psd[1:] *= 2
    
    return freqs, psd

def extract_rate_from_signal(signal, fs, freq_band, fft_func=None):
    """
    Mengekstrak frekuensi dominan (Rate) menggunakan:
    Detrend -> Welch -> Smoothing -> Peak Detection -> Quinn's Interpolation.
    Output dalam Hertz (Hz).
    """
    if signal is None or len(signal) < 10 or fs <= 0: return 0.0
    
    # 1. Detrending
    sig_detrend = signal - np.mean(signal)
    
    # 2. Welch PSD (High Res)
    # Gunakan seluruh panjang sinyal sebagai satu segmen besar untuk resolusi maksimal pada rate rendah
    freqs, psd = welch_from_scratch(sig_detrend, fs, segment_len=len(signal), overlap_ratio=0.5)
    if len(freqs) == 0: return 0.0
    
    # 3. Smoothing (3-point moving average)
    smoothed_psd = np.copy(psd)
    if len(psd) > 2:
        smoothed_psd[1:-1] = (psd[:-2] + psd[1:-1] + psd[2:]) / 3.0
    
    # 4. Cari Peak dalam Band
    low_cut, high_cut = freq_band
    mask = (freqs >= low_cut) & (freqs <= high_cut)
    
    if not np.any(mask): return 0.0
    
    # Dapatkan indeks
    band_indices = np.where(mask)[0]
    peak_local_idx = np.argmax(smoothed_psd[band_indices])
    peak_global_idx = band_indices[peak_local_idx]
    max_power = smoothed_psd[peak_global_idx]
    
    # 5. Quinn's Interpolation (Refinement untuk akurasi sub-bin)
    refined_freq = freqs[peak_global_idx]
    
    if 0 < peak_global_idx < len(smoothed_psd) - 1 and max_power > 0.0001:
        y1 = smoothed_psd[peak_global_idx - 1]
        y2 = smoothed_psd[peak_global_idx]
        y3 = smoothed_psd[peak_global_idx + 1]
        
        denom = (y1 - 2*y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom
            # Clamp delta (-0.5 s.d 0.5)
            delta = max(-0.5, min(0.5, delta))
            freq_step = freqs[1] - freqs[0]
            refined_freq = freqs[peak_global_idx] + (delta * freq_step)
            
    return refined_freq 

# ==============================================================================
# 3. ZERO CROSSING ANALYZER
# ==============================================================================

def analyze_signal_zero_crossing(signal, time, zero_line_val=0.0):
    """
    Analisis Zero Crossing State Machine.
    Return: Arrays of INDICES (Integers).
    FIX: Memastikan output berupa Integer Array untuk mencegah IndexError di GUI.
    """
    maxima_idx = []
    minima_idx = []
    zero_cross_idx = []
    
    if len(signal) < 2: 
        return np.array(maxima_idx, dtype=int), np.array(minima_idx, dtype=int), np.array(zero_cross_idx, dtype=int), zero_line_val
        
    # State Variables
    is_positive = signal[0] >= zero_line_val
    local_max = -np.inf
    local_min = np.inf
    idx_max = -1
    idx_min = -1
    
    # Refractory Period Logic (300 ms)
    min_refractory_period = 0.30 
    last_peak_time = -999.0
    
    for i in range(len(signal)):
        val = signal[i]
        
        # A. Cari Ekstrim Lokal
        if is_positive:
            if val > local_max:
                local_max = val
                idx_max = i
        else:
            if val < local_min:
                local_min = val
                idx_min = i
                
        # B. Cek Zero Crossing
        if i < len(signal) - 1:
            # Case 1: Crossing DOWN (Positif -> Negatif)
            if is_positive and (signal[i+1] < zero_line_val):
                is_positive = False
                zero_cross_idx.append(i)
                
                # --- Validasi Puncak (Refractory) ---
                if idx_max != -1:
                    current_peak_time = time[idx_max]
                    
                    if len(maxima_idx) == 0:
                        maxima_idx.append(idx_max)
                        last_peak_time = current_peak_time
                    else:
                        time_diff = current_peak_time - last_peak_time
                        
                        if time_diff >= min_refractory_period:
                            # Jarak aman, terima puncak baru
                            maxima_idx.append(idx_max)
                            last_peak_time = current_peak_time
                        else:
                            # Terlalu dekat. Cek amplitudo.
                            last_peak_idx = maxima_idx[-1]
                            if signal[idx_max] > signal[last_peak_idx]:
                                # Puncak baru lebih tinggi, ganti yang lama
                                maxima_idx.pop()
                                maxima_idx.append(idx_max)
                                last_peak_time = current_peak_time
                                
                # Reset Pencarian Minima
                local_min = np.inf
                idx_min = -1
                
            # Case 2: Crossing UP (Negatif -> Positif)
            elif (not is_positive) and (signal[i+1] >= zero_line_val):
                is_positive = True
                zero_cross_idx.append(i)
                
                # Simpan Minima (Diastolik)
                if idx_min != -1:
                    minima_idx.append(idx_min)
                    
                # Reset Pencarian Maxima
                local_max = -np.inf
                idx_max = -1
                
    # CRITICAL FIX: Cast lists to numpy array of INT
    return (np.array(maxima_idx, dtype=int), 
            np.array(minima_idx, dtype=int), 
            np.array(zero_cross_idx, dtype=int), 
            zero_line_val)

# ==============================================================================
# 4. MAIN ANALYZER CLASSES
# ==============================================================================

class PPGStressAnalyzer:
    def __init__(self, sampling_rate=50.0): 
        self.fs = sampling_rate
        self.original_fs = sampling_rate
        self.qj_time_coeffs = self._initialize_qj_time_coeffs()
        
    def _initialize_qj_time_coeffs(self):
        """Inisialisasi Koefisien DWT Hardcoded (Q1-Q8)."""
        qj_filters = {}
        dd = dirac_delta 
        
        for j in range(1, 9):
            filter_dict = {}
            start_k = -(2**j + 2**(j-1) - 2)
            end_k = (1 - 2**(j-1)) + 1
            k_range = range(start_k, end_k + 1)
            
            if j == 1:
                for k in k_range: filter_dict[k] = -2 * (dd(k) - dd(k + 1))
            elif j == 2:
                for k in k_range: filter_dict[k] = -1/4 * (dd(k-1) + 3*dd(k) + 2*dd(k+1) - 2*dd(k+2) - 3*dd(k+3) - dd(k+4))
            elif j == 3:
                for k in k_range: filter_dict[k] = -1/32 * (dd(k-3) + 3*dd(k-2) + 6*dd(k-1) + 10*dd(k) + 11*dd(k+1) + 9*dd(k+2) + 4*dd(k+3) - 4*dd(k+4) - 9*dd(k+5) - 11*dd(k+6) - 10*dd(k+7) - 6*dd(k+8) - 3*dd(k+9) - dd(k+10))
            elif j == 4:
                for k in k_range: filter_dict[k] = -1/256 * (dd(k-7) + 3*dd(k-6) + 6*dd(k-5) + 10*dd(k-4) + 15*dd(k-3) + 21*dd(k-2) + 28*dd(k-1) + 36*dd(k) + 41*dd(k+1) + 43*dd(k+2) + 42*dd(k+3) + 38*dd(k+4) + 31*dd(k+5) + 21*dd(k+6) + 8*dd(k+7) - 8*dd(k+8) - 21*dd(k+9) - 31*dd(k+10) - 38*dd(k+11) - 42*dd(k+12) - 43*dd(k+13) - 41*dd(k+14) - 36*dd(k+15) - 28*dd(k+16) - 21*dd(k+17) - 15*dd(k+18) - 10*dd(k+19) - 6*dd(k+20) - 3*dd(k+21) - dd(k+22))
            elif j == 5:
                 for k in k_range: filter_dict[k] = -1/2048 * (dd(k-15) + 3*dd(k-14) + 6*dd(k-13) + 10*dd(k-12) + 15*dd(k-11) + 21*dd(k-10) + 28*dd(k-9) + 36*dd(k-8) + 45*dd(k-7) + 55*dd(k-6) + 66*dd(k-5) + 78*dd(k-4) + 91*dd(k-3) + 105*dd(k-2) + 120*dd(k-1) + 136*dd(k) + 149*dd(k+1) + 159*dd(k+2) + 166*dd(k+3) + 170*dd(k+4) + 171*dd(k+5) + 169*dd(k+6) + 164*dd(k+7) + 156*dd(k+8) + 145*dd(k+9) + 131*dd(k+10) + 114*dd(k+11) + 94*dd(k+12) + 71*dd(k+13) + 45*dd(k+14) + 16*dd(k+15) - 16*dd(k+16) - 45*dd(k+17) - 71*dd(k+18) - 94*dd(k+19) - 114*dd(k+20) - 131*dd(k+21) - 145*dd(k+22) - 156*dd(k+23) - 164*dd(k+24) - 169*dd(k+25) - 171*dd(k+26) - 170*dd(k+27) - 166*dd(k+28) - 159*dd(k+29) - 149*dd(k+30) - 136*dd(k+31) - 120*dd(k+32) - 105*dd(k+33) - 91*dd(k+34) - 78*dd(k+35) - 66*dd(k+36) - 55*dd(k+37) - 45*dd(k+38) - 36*dd(k+39) - 28*dd(k+40) - 21*dd(k+41) - 15*dd(k+42) - 10*dd(k+43) - 6*dd(k+44) - 3*dd(k+45) - dd(k+46))
            elif j == 6:
                for k in k_range: filter_dict[k] = -1/16384 * (dd(k-31) + 3*dd(k-30) + 6*dd(k-29) + 10*dd(k-28) + 15*dd(k-27) + 21*dd(k-26) + 28*dd(k-25) + 36*dd(k-24) + 45*dd(k-23) + 55*dd(k-22) + 66*dd(k-21) + 78*dd(k-20) + 91*dd(k-19) + 105*dd(k-18) + 120*dd(k-17) + 136*dd(k-16) + 153*dd(k-15) + 171*dd(k-14) + 190*dd(k-13) + 210*dd(k-12) + 231*dd(k-11) + 253*dd(k-10) + 276*dd(k-9) + 300*dd(k-8) + 325*dd(k-7) + 351*dd(k-6) + 378*dd(k-5) + 406*dd(k-4) + 435*dd(k-3) + 465*dd(k-2) + 496*dd(k-1) + 528*dd(k) + 557*dd(k+1) + 583*dd(k+2) + 606*dd(k+3) + 626*dd(k+4) + 643*dd(k+5) + 657*dd(k+6) + 668*dd(k+7) + 676*dd(k+8) + 681*dd(k+9) + 683*dd(k+10) + 682*dd(k+11) + 678*dd(k+12) + 671*dd(k+13) + 661*dd(k+14) + 648*dd(k+15) + 632*dd(k+16) + 613*dd(k+17) + 591*dd(k+18) + 566*dd(k+19) + 538*dd(k+20) + 507*dd(k+21) + 473*dd(k+22) + 436*dd(k+23) + 396*dd(k+24) + 353*dd(k+25) + 307*dd(k+26) + 258*dd(k+27) + 206*dd(k+28) + 151*dd(k+29) + 93*dd(k+30) + 32*dd(k+31) - 32*dd(k+32) - 93*dd(k+33) - 151*dd(k+34) - 206*dd(k+35) - 258*dd(k+36) - 307*dd(k+37) - 353*dd(k+38) - 396*dd(k+39) - 436*dd(k+40) - 473*dd(k+41) - 507*dd(k+42) - 538*dd(k+43) - 566*dd(k+44) - 591*dd(k+45) - 613*dd(k+46) - 632*dd(k+47) - 648*dd(k+48) - 661*dd(k+49) - 671*dd(k+50) - 678*dd(k+51) - 682*dd(k+52) - 683*dd(k+53) - 681*dd(k+54) - 676*dd(k+55) - 668*dd(k+56) - 657*dd(k+57) - 643*dd(k+58) - 626*dd(k+59) - 606*dd(k+60) - 583*dd(k+61) - 557*dd(k+62) - 528*dd(k+63) - 496*dd(k+64) - 465*dd(k+65) - 435*dd(k+66) - 406*dd(k+67) - 378*dd(k+68) - 351*dd(k+69) - 325*dd(k+70) - 300*dd(k+71) - 276*dd(k+72) - 253*dd(k+73) - 231*dd(k+74) - 210*dd(k+75) - 190*dd(k+76) - 171*dd(k+77) - 153*dd(k+78) - 136*dd(k+79) - 120*dd(k+80) - 105*dd(k+81) - 91*dd(k+82) - 78*dd(k+83) - 66*dd(k+84) - 55*dd(k+85) - 45*dd(k+86) - 36*dd(k+87) - 28*dd(k+88) - 21*dd(k+89) - 15*dd(k+90) - 10*dd(k+91) - 6*dd(k+92) - 3*dd(k+93) - dd(k+94))
            elif j == 7:
                for k in k_range: filter_dict[k] = -1/131072 * (dd(k-63) + 3*dd(k-62) + 6*dd(k-61) + 10*dd(k-60) + 15*dd(k-59) + 21*dd(k-58) + 28*dd(k-57) + 36*dd(k-56) + 45*dd(k-55) + 55*dd(k-54) + 66*dd(k-53) + 78*dd(k-52) + 91*dd(k-51) + 105*dd(k-50) + 120*dd(k-49) + 136*dd(k-48) + 153*dd(k-47) + 171*dd(k-46) + 190*dd(k-45) + 210*dd(k-44) + 231*dd(k-43) + 253*dd(k-42) + 276*dd(k-41) + 300*dd(k-40) + 325*dd(k-39) + 351*dd(k-38) + 378*dd(k-37) + 406*dd(k-36) + 435*dd(k-35) + 465*dd(k-34) + 496*dd(k-33) + 528*dd(k-32) + 561*dd(k-31) + 595*dd(k-30) + 630*dd(k-29) + 666*dd(k-28) + 703*dd(k-27) + 741*dd(k-26) + 780*dd(k-25) + 820*dd(k-24) + 861*dd(k-23) + 903*dd(k-22) + 946*dd(k-21) + 990*dd(k-20) + 1035*dd(k-19) + 1081*dd(k-18) + 1128*dd(k-17) + 1176*dd(k-16) + 1225*dd(k-15) + 1275*dd(k-14) + 1326*dd(k-13) + 1378*dd(k-12) + 1431*dd(k-11) + 1485*dd(k-10) + 1540*dd(k-9) + 1596*dd(k-8) + 1653*dd(k-7) + 1711*dd(k-6) + 1770*dd(k-5) + 1830*dd(k-4) + 1891*dd(k-3) + 1953*dd(k-2) + 2016*dd(k-1) + 2080*dd(k) + 2141*dd(k+1) + 2199*dd(k+2) + 2254*dd(k+3) + 2306*dd(k+4) + 2355*dd(k+5) + 2401*dd(k+6) + 2444*dd(k+7) + 2484*dd(k+8) + 2521*dd(k+9) + 2555*dd(k+10) + 2586*dd(k+11) + 2614*dd(k+12) + 2639*dd(k+13) + 2661*dd(k+14) + 2680*dd(k+15) + 2696*dd(k+16) + 2709*dd(k+17) + 2719*dd(k+18) + 2726*dd(k+19) + 2730*dd(k+20) + 2731*dd(k+21) + 2729*dd(k+22) + 2724*dd(k+23) + 2716*dd(k+24) + 2705*dd(k+25) + 2691*dd(k+26) + 2674*dd(k+27) + 2654*dd(k+28) + 2631*dd(k+29) + 2605*dd(k+30) + 2576*dd(k+31) + 2544*dd(k+32) + 2509*dd(k+33) + 2471*dd(k+34) + 2430*dd(k+35) + 2386*dd(k+36) + 2339*dd(k+37) + 2289*dd(k+38) + 2236*dd(k+39) + 2180*dd(k+40) + 2121*dd(k+41) + 2059*dd(k+42) + 1994*dd(k+43) + 1926*dd(k+44) + 1855*dd(k+45) + 1781*dd(k+46) + 1704*dd(k+47) + 1624*dd(k+48) + 1541*dd(k+49) + 1455*dd(k+50) + 1366*dd(k+51) + 1274*dd(k+52) + 1179*dd(k+53) + 1081*dd(k+54) + 980*dd(k+55) + 876*dd(k+56) + 769*dd(k+57) + 659*dd(k+58) + 546*dd(k+59) + 430*dd(k+60) + 311*dd(k+61) + 189*dd(k+62) + 64*dd(k+63) - 64*dd(k+64) - 189*dd(k+65) - 311*dd(k+66) - 430*dd(k+67) - 546*dd(k+68) - 659*dd(k+69) - 769*dd(k+70) - 876*dd(k+71) - 980*dd(k+72) - 1081*dd(k+73) - 1179*dd(k+74) - 1274*dd(k+75) - 1366*dd(k+76) - 1455*dd(k+77) - 1541*dd(k+78) - 1624*dd(k+79) - 1704*dd(k+80) - 1781*dd(k+81) - 1855*dd(k+82) - 1926*dd(k+83) - 1994*dd(k+84) - 2059*dd(k+85) - 2121*dd(k+86) - 2180*dd(k+87) - 2236*dd(k+88) - 2289*dd(k+89) - 2339*dd(k+90) - 2386*dd(k+91) - 2430*dd(k+92) - 2471*dd(k+93) - 2509*dd(k+94) - 2544*dd(k+95) - 2576*dd(k+96) - 2605*dd(k+97) - 2631*dd(k+98) - 2654*dd(k+99) - 2674*dd(k+100) - 2691*dd(k+101) - 2705*dd(k+102) - 2716*dd(k+103) - 2724*dd(k+104) - 2729*dd(k+105) - 2731*dd(k+106) - 2730*dd(k+107) - 2726*dd(k+108) - 2719*dd(k+109) - 2709*dd(k+110) - 2696*dd(k+111) - 2680*dd(k+112) - 2661*dd(k+113) - 2639*dd(k+114) - 2614*dd(k+115) - 2586*dd(k+116) - 2555*dd(k+117) - 2521*dd(k+118) - 2484*dd(k+119) - 2444*dd(k+120) - 2401*dd(k+121) - 2355*dd(k+122) - 2306*dd(k+123) - 2254*dd(k+124) - 2199*dd(k+125) - 2141*dd(k+126) - 2080*dd(k+127) - 2016*dd(k+128) - 1953*dd(k+129) - 1891*dd(k+130) - 1830*dd(k+131) - 1770*dd(k+132) - 1711*dd(k+133) - 1653*dd(k+134) - 1596*dd(k+135) - 1540*dd(k+136) - 1485*dd(k+137) - 1431*dd(k+138) - 1378*dd(k+139) - 1326*dd(k+140) - 1275*dd(k+141) - 1225*dd(k+142) - 1176*dd(k+143) - 1128*dd(k+144) - 1081*dd(k+145) - 1035*dd(k+146) - 990*dd(k+147) - 946*dd(k+148) - 903*dd(k+149) - 861*dd(k+150) - 820*dd(k+151) - 780*dd(k+152) - 741*dd(k+153) - 703*dd(k+154) - 666*dd(k+155) - 630*dd(k+156) - 595*dd(k+157) - 561*dd(k+158) - 528*dd(k+159) - 496*dd(k+160) - 465*dd(k+161) - 435*dd(k+162) - 406*dd(k+163) - 378*dd(k+164) - 351*dd(k+165) - 325*dd(k+166) - 300*dd(k+167) - 276*dd(k+168) - 253*dd(k+169) - 231*dd(k+170) - 210*dd(k+171) - 190*dd(k+172) - 171*dd(k+173) - 153*dd(k+174) - 136*dd(k+175) - 120*dd(k+176) - 105*dd(k+177) - 91*dd(k+178) - 78*dd(k+179) - 66*dd(k+180) - 55*dd(k+181) - 45*dd(k+182) - 36*dd(k+183) - 28*dd(k+184) - 21*dd(k+185) - 15*dd(k+186) - 10*dd(k+187) - 6*dd(k+188) - 3*dd(k+189) - dd(k+190))
            elif j == 8:
                for k in k_range: filter_dict[k] = -1/1048576 * (dd(k-127) + 3*dd(k-126) + 6*dd(k-125) + 10*dd(k-124) + 15*dd(k-123) + 21*dd(k-122) + 28*dd(k-121) + 36*dd(k-120) + 45*dd(k-119) + 55*dd(k-118) + 66*dd(k-117) + 78*dd(k-116) + 91*dd(k-115) + 105*dd(k-114) + 120*dd(k-113) + 136*dd(k-112) + 153*dd(k-111) + 171*dd(k-110) + 190*dd(k-109) + 210*dd(k-108) + 231*dd(k-107) + 253*dd(k-106) + 276*dd(k-105) + 300*dd(k-104) + 325*dd(k-103) + 351*dd(k-102) + 378*dd(k-101) + 406*dd(k-100) + 435*dd(k-99) + 465*dd(k-98) + 496*dd(k-97) + 528*dd(k-96) + 561*dd(k-95) + 595*dd(k-94) + 630*dd(k-93) + 666*dd(k-92) + 703*dd(k-91) + 741*dd(k-90) + 780*dd(k-89) + 820*dd(k-88) + 861*dd(k-87) + 903*dd(k-86) + 946*dd(k-85) + 990*dd(k-84) + 1035*dd(k-83) + 1081*dd(k-82) + 1128*dd(k-81) + 1176*dd(k-80) + 1225*dd(k-79) + 1275*dd(k-78) + 1326*dd(k-77) + 1378*dd(k-76) + 1431*dd(k-75) + 1485*dd(k-74) + 1540*dd(k-73) + 1596*dd(k-72) + 1653*dd(k-71) + 1711*dd(k-70) + 1770*dd(k-69) + 1830*dd(k-68) + 1891*dd(k-67) + 1953*dd(k-66) + 2016*dd(k-65) + 2080*dd(k-64) + 2145*dd(k-63) + 2211*dd(k-62) + 2278*dd(k-61) + 2346*dd(k-60) + 2415*dd(k-59) + 2485*dd(k-58) + 2556*dd(k-57) + 2628*dd(k-56) + 2701*dd(k-55) + 2775*dd(k-54) + 2850*dd(k-53) + 2926*dd(k-52) + 3003*dd(k-51) + 3081*dd(k-50) + 3160*dd(k-49) + 3240*dd(k-48) + 3321*dd(k-47) + 3403*dd(k-46) + 3486*dd(k-45) + 3570*dd(k-44) + 3655*dd(k-43) + 3741*dd(k-42) + 3828*dd(k-41) + 3916*dd(k-40) + 4005*dd(k-39) + 4186*dd(k-37) + 4278*dd(k-36) + 4371*dd(k-35) + 4465*dd(k-34) + 4560*dd(k-33) + 4656*dd(k-32) + 4753*dd(k-31) + 4851*dd(k-30) + 4950*dd(k-29) + 5050*dd(k-28) + 5151*dd(k-27) + 5253*dd(k-26) + 5356*dd(k-25) + 5460*dd(k-24) + 5565*dd(k-23) + 5671*dd(k-22) + 5778*dd(k-21) + 5886*dd(k-20) + 5995*dd(k-19) + 6105*dd(k-18) + 6216*dd(k-17) + 6328*dd(k-16) + 6441*dd(k-15) + 6555*dd(k-14) + 6670*dd(k-13) + 6786*dd(k-12) + 6903*dd(k-11) + 7021*dd(k-10) + 7140*dd(k-9) + 7260*dd(k-8) + 7381*dd(k-7) + 7503*dd(k-6) + 7626*dd(k-5) + 7750*dd(k-4) + 7875*dd(k-3) + 8001*dd(k-2) + 8128*dd(k-1) + 8256*dd(k) + 8381*dd(k+1) + 8503*dd(k+2) + 8622*dd(k+3) + 8738*dd(k+4) + 8851*dd(k+5) + 8961*dd(k+6) + 9068*dd(k+7) + 9172*dd(k+8) + 9273*dd(k+9) + 9371*dd(k+10) + 9466*dd(k+11) + 9558*dd(k+12) + 9647*dd(k+13) + 9733*dd(k+14) + 9816*dd(k+15) + 9896*dd(k+16) + 9973*dd(k+17) + 10047*dd(k+18) + 10118*dd(k+19) + 10186*dd(k+20) + 10251*dd(k+21) + 10313*dd(k+22) + 10372*dd(k+23) + 10428*dd(k+24) + 10481*dd(k+25) + 10531*dd(k+26) + 10578*dd(k+27) + 10622*dd(k+28) + 10663*dd(k+29) + 10701*dd(k+30) + 10736*dd(k+31) + 10768*dd(k+32) + 10797*dd(k+33) + 10823*dd(k+34) + 10846*dd(k+35) + 10866*dd(k+36) + 10883*dd(k+37) + 10897*dd(k+38) + 10908*dd(k+39) + 10916*dd(k+40) + 10921*dd(k+41) + 10923*dd(k+42) + 10922*dd(k+43) + 10918*dd(k+44) + 10911*dd(k+45) + 10901*dd(k+46) + 10888*dd(k+47) + 10872*dd(k+48) + 10853*dd(k+49) + 10831*dd(k+50) + 10806*dd(k+51) + 10778*dd(k+52) + 10747*dd(k+53) + 10713*dd(k+54) + 10676*dd(k+55) + 10636*dd(k+56) + 10593*dd(k+57) + 10547*dd(k+58) + 10498*dd(k+59) + 10446*dd(k+60) + 10391*dd(k+61) + 10333*dd(k+62) + 10272*dd(k+63) + 10208*dd(k+64) + 10141*dd(k+65) + 10071*dd(k+66) + 9998*dd(k+67) + 9922*dd(k+68) + 9843*dd(k+69) + 9761*dd(k+70) + 9676*dd(k+71) + 9588*dd(k+72) + 9497*dd(k+73) + 9403*dd(k+74) + 9306*dd(k+75) + 9206*dd(k+76) + 9103*dd(k+77) + 8997*dd(k+78) + 8888*dd(k+79) + 8776*dd(k+80) + 8661*dd(k+81) + 8543*dd(k+82) + 8422*dd(k+83) + 8298*dd(k+84) + 8171*dd(k+85) + 8041*dd(k+86) + 7908*dd(k+87) + 7772*dd(k+88) + 7633*dd(k+89) + 7491*dd(k+90) + 7346*dd(k+91) + 7198*dd(k+92) + 7047*dd(k+93) + 6893*dd(k+94) + 6736*dd(k+95) + 6576*dd(k+96) + 6413*dd(k+97) + 6247*dd(k+98) + 5906*dd(k+100) + 5731*dd(k+101) + 5553*dd(k+102) + 5372*dd(k+103) + 5188*dd(k+104) + 5001*dd(k+105) + 4811*dd(k+106) + 4618*dd(k+107) + 4422*dd(k+108) + 4223*dd(k+109) + 4021*dd(k+110) + 3816*dd(k+111) + 3608*dd(k+112) + 3397*dd(k+113) + 3183*dd(k+114) + 2966*dd(k+115) + 2746*dd(k+116) + 2523*dd(k+117) + 2297*dd(k+118) + 2068*dd(k+119) + 1836*dd(k+120) + 1601*dd(k+121) + 1363*dd(k+122) + 1122*dd(k+123) + 878*dd(k+124) + 631*dd(k+125) + 381*dd(k+126) + 128*dd(k+127) - 128*dd(k+128) - 381*dd(k+129) - 631*dd(k+130) - 878*dd(k+131) - 1122*dd(k+132) - 1363*dd(k+133) - 1601*dd(k+134) - 1836*dd(k+135) - 2068*dd(k+136) - 2297*dd(k+137) - 2523*dd(k+138) - 2746*dd(k+139) - 2966*dd(k+140) - 3183*dd(k+141) - 3397*dd(k+142) - 3608*dd(k+143) - 3816*dd(k+144) - 4021*dd(k+145) - 4223*dd(k+146) - 4422*dd(k+147) - 4618*dd(k+148) - 4811*dd(k+149) - 5001*dd(k+150) - 5188*dd(k+151) - 5372*dd(k+152) - 5553*dd(k+153) - 5731*dd(k+154) - 5906*dd(k+155) - 6078*dd(k+156) - 6247*dd(k+157) - 6413*dd(k+158) - 6576*dd(k+159) - 6736*dd(k+160) - 6893*dd(k+161) - 7047*dd(k+162) - 7198*dd(k+163) - 7346*dd(k+164) - 7491*dd(k+165) - 7633*dd(k+166) - 7772*dd(k+167) - 7908*dd(k+168) - 8041*dd(k+169) - 8171*dd(k+170) - 8298*dd(k+171) - 8422*dd(k+172) - 8543*dd(k+173) - 8661*dd(k+174) - 8776*dd(k+175) - 8888*dd(k+176) - 8997*dd(k+177) - 9103*dd(k+178) - 9206*dd(k+179) - 9306*dd(k+180) - 9403*dd(k+181) - 9497*dd(k+182) - 9588*dd(k+183) - 9676*dd(k+184) - 9761*dd(k+185) - 9843*dd(k+186) - 9922*dd(k+187) - 9998*dd(k+188) - 10071*dd(k+189) - 10141*dd(k+190) - 10208*dd(k+191) - 10272*dd(k+192) - 10333*dd(k+193) - 10391*dd(k+194) - 10446*dd(k+195) - 10498*dd(k+196) - 10547*dd(k+197) - 10593*dd(k+198) - 10636*dd(k+199) - 10676*dd(k+200) - 10713*dd(k+201) - 10747*dd(k+202) - 10778*dd(k+203) - 10806*dd(k+204) - 10831*dd(k+205) - 10853*dd(k+206) - 10872*dd(k+207) - 10888*dd(k+208) - 10901*dd(k+209) - 10911*dd(k+210) - 10918*dd(k+211) - 10922*dd(k+212) - 10923*dd(k+213) - 10921*dd(k+214) - 10916*dd(k+215) - 10908*dd(k+216) - 10897*dd(k+217) - 10883*dd(k+218) - 10866*dd(k+219) - 10846*dd(k+220) - 10823*dd(k+221) - 10797*dd(k+222) - 10768*dd(k+223) - 10736*dd(k+224) - 10701*dd(k+225) - 10663*dd(k+226) - 10622*dd(k+227) - 10578*dd(k+228) - 10531*dd(k+229) - 10481*dd(k+230) - 10428*dd(k+231) - 10372*dd(k+232) - 10313*dd(k+233) - 10251*dd(k+234) - 10186*dd(k+235) - 10118*dd(k+236) - 10047*dd(k+237) - 9973*dd(k+238) - 9896*dd(k+239) - 9816*dd(k+240) - 9733*dd(k+241) - 9647*dd(k+242) - 9558*dd(k+243) - 9466*dd(k+244) - 9371*dd(k+245) - 9273*dd(k+246) - 9172*dd(k+247) - 9068*dd(k+248) - 8961*dd(k+249) - 8851*dd(k+250) - 8738*dd(k+251) - 8622*dd(k+252) - 8503*dd(k+253) - 8381*dd(k+254) - 8256*dd(k+255) - 8128*dd(k+256) - 8001*dd(k+257) - 7875*dd(k+258) - 7750*dd(k+259) - 7626*dd(k+260) - 7503*dd(k+261) - 7381*dd(k+262) - 7260*dd(k+263) - 7140*dd(k+264) - 7021*dd(k+265) - 6903*dd(k+266) - 6786*dd(k+267) - 6670*dd(k+268) - 6555*dd(k+269) - 6441*dd(k+270) - 6328*dd(k+271) - 6216*dd(k+272) - 6105*dd(k+273) - 5995*dd(k+274) - 5886*dd(k+275) - 5778*dd(k+276) - 5671*dd(k+277) - 5565*dd(k+278) - 5460*dd(k+279) - 5356*dd(k+280) - 5253*dd(k+281) - 5151*dd(k+282) - 5050*dd(k+283) - 4950*dd(k+284) - 4851*dd(k+285) - 4753*dd(k+286) - 4656*dd(k+287) - 4560*dd(k+288) - 4465*dd(k+289) - 4371*dd(k+290) - 4278*dd(k+291) - 4186*dd(k+292) - 4095*dd(k+293) - 4005*dd(k+294) - 3916*dd(k+295) - 3828*dd(k+296) - 3741*dd(k+297) - 3655*dd(k+298) - 3570*dd(k+299) - 3486*dd(k+300) - 3403*dd(k+301) - 3321*dd(k+302) - 3240*dd(k+303) - 3160*dd(k+304) - 3081*dd(k+305) - 3003*dd(k+306) - 2926*dd(k+307) - 2850*dd(k+308) - 2775*dd(k+309) - 2701*dd(k+310) - 2628*dd(k+311) - 2556*dd(k+312) - 2485*dd(k+313) - 2415*dd(k+314) - 2346*dd(k+315) - 2278*dd(k+316) - 2211*dd(k+317) - 2145*dd(k+318) - 2080*dd(k+319) - 2016*dd(k+320) - 1953*dd(k+321) - 1891*dd(k+322) - 1830*dd(k+323) - 1770*dd(k+324) - 1711*dd(k+325) - 1653*dd(k+326) - 1596*dd(k+327) - 1540*dd(k+328) - 1485*dd(k+329) - 1431*dd(k+330) - 1378*dd(k+331) - 1326*dd(k+332) - 1275*dd(k+333) - 1225*dd(k+334) - 1176*dd(k+335) - 1128*dd(k+336) - 1081*dd(k+337) - 1035*dd(k+338) - 990*dd(k+339) - 946*dd(k+340) - 903*dd(k+341) - 861*dd(k+342) - 820*dd(k+343) - 780*dd(k+344) - 741*dd(k+345) - 703*dd(k+346) - 666*dd(k+347) - 630*dd(k+348) - 595*dd(k+349) - 561*dd(k+350) - 528*dd(k+351) - 496*dd(k+352) - 465*dd(k+353) - 435*dd(k+354) - 406*dd(k+355) - 378*dd(k+356) - 351*dd(k+357) - 325*dd(k+358) - 300*dd(k+359) - 276*dd(k+360) - 253*dd(k+361) - 231*dd(k+362) - 210*dd(k+363) - 190*dd(k+364) - 171*dd(k+365) - 153*dd(k+366) - 136*dd(k+367) - 120*dd(k+368) - 105*dd(k+369) - 91*dd(k+370) - 78*dd(k+371) - 66*dd(k+372) - 55*dd(k+373) - 45*dd(k+374) - 36*dd(k+375) - 28*dd(k+376) - 21*dd(k+377) - 15*dd(k+378) - 10*dd(k+379) - 6*dd(k+380) - 3*dd(k+381) - dd(k+382))
            
            qj_filters[j] = filter_dict
        return qj_filters
    
    def calculate_qj_frequency_responses(self, fs):
        """
        Menghitung Respons Frekuensi dengan FFT pada Impulse Response.
        Match dengan visualisasi Delphi: Frequency Axis = 0 s/d Nyquist.
        """
        responses = {}
        nfft = 2048
        half = nfft // 2
        
        for j in range(1, 9):
            if j not in self.qj_time_coeffs: continue
            
            filter_dict = self.qj_time_coeffs[j]
            min_k = min(filter_dict.keys())
            max_k = max(filter_dict.keys())
            
            # Buat buffer impulse response zero-padded
            padded_signal = np.zeros(nfft)
            
            # Masukkan koefisien ke buffer
            for k, val in filter_dict.items():
                idx = k - min_k
                if 0 <= idx < nfft:
                    padded_signal[idx] = val
            
            # FFT
            fft_out = np.fft.fft(padded_signal)
            mags = np.abs(fft_out)[:half]
            freqs = np.linspace(0, fs/2, half)
            
            responses[j] = (freqs, mags)
            
        return responses

    def fft_from_scratch(self, signal):
        return fft_from_scratch(signal)

    def fft_magnitude_and_frequencies(self, signal):
        fft_complex = self.fft_from_scratch(signal)
        N = len(fft_complex)
        if N == 0: return np.array([]), np.array([])
        
        magnitude = np.abs(fft_complex)[:N//2] * 2 / N
        if N > 0: magnitude[0] /= 2
        
        frequencies = np.fft.fftfreq(N, 1.0/self.fs)[:N//2]
        return frequencies, magnitude

    def dwt_convolution_from_scratch(self, signal, max_scale=8):
        dwt_coeffs = {}
        for j in range(1, max_scale + 1):
            if j in self.qj_time_coeffs:
                q_filter_dict = self.qj_time_coeffs[j]
                min_k = min(q_filter_dict.keys())
                max_k = max(q_filter_dict.keys())
                kernel = [q_filter_dict.get(k, 0) for k in range(min_k, max_k + 1)]
                dwt_coeffs[j] = np.convolve(signal, kernel, mode='same')
        return dwt_coeffs
    
    def load_ppg_data(self, csv_file, ppg_column_name):
        try:
            df = pd.read_csv(csv_file)
            df.columns = [c.strip().lower() for c in df.columns]
            target_col = ppg_column_name.strip().lower()
            
            time_col = next((col for col in df.columns if 'time' in col or 'index' in col), None)
            
            if target_col not in df.columns: raise ValueError(f"Column '{ppg_column_name}' not found.")
            
            if time_col:
                time = df[time_col].values
            else:
                time = np.arange(len(df)) / 50.0 # Default if no time
                self.fs = 50.0
                
            ppg_signal = df[target_col].values
            valid_mask = np.isfinite(time) & np.isfinite(ppg_signal)
            time = time[valid_mask]
            ppg_signal = ppg_signal[valid_mask]
            
            if len(time) < 2: raise ValueError("Not enough data.")
            
            # --- CRITICAL FIX: Calculate Real FS ---
            duration = time[-1] - time[0]
            if duration > 0:
                self.original_fs = self.fs = (len(time) - 1) / duration
            else:
                self.fs = 50.0
                
            return time, ppg_signal
        except Exception as e:
            print(f"Error loading data: {e}"); return None, None

    def downsample_signal(self, signal, time, factor):
        if factor <= 1:
            self.fs = self.original_fs
            return signal, time
        
        ds_signal = signal[::factor]
        ds_time = time[::factor]
        
        if len(ds_time) > 1:
            self.fs = (len(ds_time) - 1) / (ds_time[-1] - ds_time[0])
        else:
            self.fs = self.original_fs / factor
            
        return ds_signal, ds_time

class HRV_Analyzer:
    def __init__(self, signal, time_vector, fs, fft_function):
        self.raw_signal = np.array(signal)
        self.time = np.array(time_vector)
        self.fs = fs
        self.fft_func = fft_function
        
        # State
        self.preprocessed_signal = None
        self.peaks = None
        self.minima = None
        self.zero_crossings = None
        self.rr_intervals = None
        self.peak_times = None

    def _preprocess_and_filter(self):
        """
        Pre-processing Pipeline:
        1. Remove Baseline
        2. Filter Custom (0.5 - 8.0 Hz)
        3. Normalize (CRITICAL FIX: signal / std) -> Enables Peak Detection!
        """
        # 1. Baseline Removal
        window_size = int(self.fs)
        if window_size % 2 == 0: window_size += 1
        
        if len(self.raw_signal) > window_size:
            baseline = np.convolve(self.raw_signal, np.ones(window_size)/window_size, mode='same')
            sig_no_base = self.raw_signal - baseline
        else:
            sig_no_base = self.raw_signal
            
        # 2. Filter
        filtered = filter_custom_ref(sig_no_base, self.fs, 0.5, 8.0)
        
        # 3. Normalization (Agar threshold peak detection di 0.3 dsb valid)
        std_val = np.std(filtered)
        if std_val > 0:
            self.preprocessed_signal = filtered / std_val
        else:
            self.preprocessed_signal = filtered

    def _detect_peaks(self):
        if self.preprocessed_signal is None: return
        
        max_idx, min_idx, zero_idx, _ = analyze_signal_zero_crossing(self.preprocessed_signal, self.time, 0.0)
        self.peaks = max_idx
        self.minima = min_idx
        self.zero_crossings = zero_idx
        
        if len(self.peaks) > 1:
            self.peak_times = self.time[self.peaks]
            # Delphi Logic: RR Interval adalah selisih waktu antar puncak berurutan
            self.rr_intervals = np.diff(self.peak_times)

    def _calculate_time_domain_features(self):
        res = defaultdict(lambda: 0)
        res['rr_histogram'] = (np.array([]), np.array([]))
        if self.rr_intervals is None: return res
        
        # Sanitasi Data (0.3s - 2.0s)
        clean_rr = [rr for rr in self.rr_intervals if 0.30 <= rr <= 2.0]
        if len(clean_rr) < 5: return res
        
        rr_ms = np.array(clean_rr) * 1000.0
        
        mean_nn = np.mean(rr_ms)
        sdnn = np.std(rr_ms, ddof=1)
        diffs = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(diffs**2))
        sdsd = np.std(diffs, ddof=1)
        nn50 = np.sum(np.abs(diffs) > 50)
        pnn50 = (nn50 / len(diffs)) * 100 if len(diffs) > 0 else 0
        
        skew = np.sum(((rr_ms - mean_nn) / sdnn)**3) / len(rr_ms) if sdnn > 0 else 0
            
        res.update({
            'mean_hr': 60000 / mean_nn if mean_nn > 0 else 0,
            'sdnn': sdnn,
            'rmssd': rmssd,
            'sdsd': sdsd,
            'nn50': nn50,
            'pnn50': pnn50,
            'cvnn': sdnn / mean_nn if mean_nn > 0 else 0,
            'cvsd': rmssd / mean_nn if mean_nn > 0 else 0,
            'skewness': skew
        })
        
        # Histogram (Bin Width 50ms)
        min_rr, max_rr = np.min(rr_ms), np.max(rr_ms)
        bin_width = 50.0
        bins = np.arange(min_rr, max_rr + bin_width, bin_width)
        counts, bin_edges = np.histogram(rr_ms, bins=bins)
        res['rr_histogram'] = (counts, bin_edges)
        
        if (max_count := np.max(counts)) > 0:
            res['hti'] = len(rr_ms) / max_count
            non_zero = np.where(counts > 0)[0]
            if len(non_zero) > 1:
                res['tinn'] = bin_edges[non_zero[-1]+1] - bin_edges[non_zero[0]]
                
        # SDANN
        total_duration = self.peak_times[-1] - self.peak_times[0]
        segment_win = 300 if total_duration >= 300 else (60 if total_duration >= 60 else total_duration)
        if segment_win > 0:
            s_means, s_stds, curr = [], [], self.peak_times[0]
            while curr + segment_win <= self.peak_times[-1]:
                window_rrs = []
                for i, t in enumerate(self.peak_times[:-1]):
                    if t >= curr and t < curr + segment_win and 0.3 <= self.rr_intervals[i] <= 2.0:
                        window_rrs.append(self.rr_intervals[i] * 1000)
                if len(window_rrs) > 2:
                    s_means.append(np.mean(window_rrs))
                    s_stds.append(np.std(window_rrs, ddof=1))
                curr += segment_win
            if len(s_means) > 1: res['sdann'] = np.std(s_means, ddof=1)
            if len(s_stds) > 0: res['sdnn_index'] = np.mean(s_stds)
                
        return res

    def _calculate_frequency_domain_features(self, interp_fs=4.0):
        defaults = defaultdict(lambda: 0, {'psd_freqs': np.array([]), 'psd_values': np.array([])})
        if self.rr_intervals is None: return defaults
        
        valid_indices = [i for i, rr in enumerate(self.rr_intervals) if 0.3 <= rr <= 2.0]
        if len(valid_indices) < 5: return defaults
        
        clean_rr = self.rr_intervals[valid_indices]
        clean_times = self.peak_times[valid_indices]
        
        # Cubic Spline Interpolation (Manual Implementation matching Delphi)
        t_start, t_end = clean_times[0], clean_times[-1]
        t_interp = np.arange(t_start, t_end, 1.0/interp_fs)
        if len(t_interp) < 2: return defaults
        
        rr_interp = cubic_spline_interpolate(clean_times, clean_rr, t_interp)
        rr_detrend = linear_detrend(rr_interp)
        
        freqs, psd = welch_from_scratch(rr_detrend, interp_fs, segment_len=min(512, len(rr_detrend)))
        if len(freqs) < 2: return defaults
        
        psd_ms2 = psd * 1e6 # s^2 -> ms^2
        freq_step = freqs[1] - freqs[0]
        
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.5)
        
        lf_pow = np.sum(psd_ms2[lf_mask]) * freq_step
        hf_pow = np.sum(psd_ms2[hf_mask]) * freq_step
        total_pow = np.sum(psd_ms2[(freqs >= 0.0033) & (freqs < 0.5)]) * freq_step
        
        res = {
            'psd_freqs': freqs, 'psd_values': psd_ms2,
            'total_power': total_pow, 'lf_power': lf_pow, 'hf_power': hf_pow,
            'peak_lf': freqs[lf_mask][np.argmax(psd_ms2[lf_mask])] if np.any(lf_mask) else 0,
            'peak_hf': freqs[hf_mask][np.argmax(psd_ms2[hf_mask])] if np.any(hf_mask) else 0
        }
        
        if hf_pow > 0: res['lf_hf_ratio'] = lf_pow / hf_pow
        if (denom := lf_pow + hf_pow) > 0:
            res['lf_nu'] = (lf_pow / denom) * 100
            res['hf_nu'] = (hf_pow / denom) * 100
            
        return res

    def _calculate_nonlinear_features(self):
        if self.rr_intervals is None: return {'poincare_x':[],'poincare_y':[],'sd1':0,'sd2':0,'sd1_sd2_ratio':0}
        valid = [rr for rr in self.rr_intervals if 0.3 <= rr <= 2.0]
        if len(valid) < 2: return {'poincare_x':[],'poincare_y':[],'sd1':0,'sd2':0,'sd1_sd2_ratio':0}
            
        rr_n = np.array(valid[:-1]) * 1000
        rr_n1 = np.array(valid[1:]) * 1000
        diff, summ = (rr_n - rr_n1) / np.sqrt(2), (rr_n + rr_n1) / np.sqrt(2)
        sd1, sd2 = np.std(diff, ddof=1), np.std(summ, ddof=1)
        
        return {
            'poincare_x': rr_n, 'poincare_y': rr_n1,
            'sd1': sd1, 'sd2': sd2,
            'sd1_sd2_ratio': sd1 / sd2 if sd2 > 0 else 0
        }

    def run_all_analyses(self):
        self._preprocess_and_filter()
        self._detect_peaks()
        return {
            "peaks": self.peaks, "minima": self.minima,
            "rr_intervals_s": self.rr_intervals,
            "rr_times": self.peak_times if self.peak_times is not None else np.array([]),
            "time_domain": self._calculate_time_domain_features(),
            "frequency_domain": self._calculate_frequency_domain_features(),
            "nonlinear": self._calculate_nonlinear_features()
        }