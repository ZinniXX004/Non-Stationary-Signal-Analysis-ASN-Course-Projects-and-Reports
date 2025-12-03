import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

class PPGProcessor:
    def __init__(self, fs: float = 125.0, transform_method: str = "FFT"):
        self.fs = float(fs)
        self.transform_method = transform_method.upper()
        self.t = None
        self.raw = None
        self.preproc = None

    # I/O and Preprocessing 
    def load_csv(self, path: str, column: str = "PLETH", time_column: Optional[str] = None):
        try:
            df = pd.read_csv(path)
            if time_column is None:
                # Mencari kolom waktu secara otomatis
                possible = [c for c in df.columns if 'time' in c.lower() or 't[' in c.lower() or 'sec' in c.lower()]
                time_column = possible[0] if possible else df.columns[0]
            self.t = df[time_column].values
            self.raw = df[column].values.astype(np.float64)
            return self.t, self.raw
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None, None

    def detrend_poly(self, x: np.ndarray, deg: int = 3) -> np.ndarray:
        if len(x) <= deg:
            return x - np.mean(x)
        p = np.polyfit(np.arange(len(x)), x, deg=deg)
        return x - np.polyval(p, np.arange(len(x)))

    def running_mean(self, x: np.ndarray, window_samples: int) -> np.ndarray:
        if window_samples <= 1:
            return np.zeros_like(x)
        kernel = np.ones(window_samples) / window_samples
        return np.convolve(x, kernel, mode='same')

    def preprocess(self, sig: np.ndarray, normalize: bool = True, remove_baseline: bool = True,
                   baseline_window_s: float = 2.0) -> np.ndarray:
        if sig is None or len(sig) == 0:
            return np.array([])
        x = sig.copy().astype(np.float64)
        x = x - np.mean(x)
        if remove_baseline:
            win = max(3, int(round(baseline_window_s * self.fs)))
            baseline = self.running_mean(x, win)
            x = x - baseline
        if normalize:
            s = np.std(x)
            if s > 0: x = x / s
        self.preproc = x
        return x

    def preproc_std(self) -> float:
        if self.preproc is None: return float('nan')
        return float(np.std(self.preproc))

    # Deteksi Ekstrema (Optimized NumPy)
    def _local_extrema(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = np.diff(x)
        maxima = np.where((np.hstack([dx, 0]) < 0) & (np.hstack([0, dx]) > 0))[0]
        minima = np.where((np.hstack([dx, 0]) > 0) & (np.hstack([0, dx]) < 0))[0]
        
        # Fallback for flat signal
        if len(maxima) == 0: maxima = np.array([np.argmax(x)], dtype=int)
        if len(minima) == 0: minima = np.array([np.argmin(x)], dtype=int)
        return maxima, minima

    # FAST MANUAL CUBIC SPLINE
    def _solve_tridiagonal(self, a, b, c, d):
        n = len(d)
        c_ = np.zeros(n-1)
        d_ = np.zeros(n)
        
        # Forward elimination
        if b[0] == 0: b[0] = 1e-12
        c_[0] = c[0] / b[0]
        d_[0] = d[0] / b[0]
        for i in range(1, n-1):
            temp = b[i] - a[i-1] * c_[i-1]
            if temp == 0: temp = 1e-12
            c_[i] = c[i] / temp
            d_[i] = (d[i] - a[i-1] * d_[i-1]) / temp
        
        denom_last = (b[n-1] - a[n-2] * c_[n-2])
        if denom_last == 0: denom_last = 1e-12
        d_[n-1] = (d[n-1] - a[n-2] * d_[n-2]) / denom_last
        
        # Backward substitution
        x = np.zeros(n)
        x[n-1] = d_[n-1]
        for i in range(n-2, -1, -1):
            x[i] = d_[i] - c_[i] * x[i+1]
        return x

    def _cubic_spline_interpolation(self, x_knots, y_knots, x_eval):
        n = len(x_knots)
        if n < 2:
            return np.zeros_like(x_eval)
            
        h = np.diff(x_knots) # h_i = x_{i+1} - x_i
        
        if np.any(h == 0): # Handle duplikat x
             return np.interp(x_eval, x_knots, y_knots)

        if n == 2:
            return np.interp(x_eval, x_knots, y_knots)

        a_tdma = h[1:-1] 
        b_tdma = 2 * (h[:-1] + h[1:]) 
        c_tdma = h[1:-1] 
        
        # Hitung RHS (d)
        diff_y = np.diff(y_knots)
        d_tdma = 6 * (diff_y[1:]/h[1:] - diff_y[:-1]/h[:-1])
        
        M_inner = self._solve_tridiagonal(a_tdma, b_tdma, c_tdma, d_tdma)
        
        M = np.hstack(([0], M_inner, [0]))
        
        # Spline Evaluation
        idx = np.searchsorted(x_knots, x_eval, side='right') - 1
        idx = np.clip(idx, 0, n - 2)
        
        xi = x_knots[idx]
        xi1 = x_knots[idx+1]
        yi = y_knots[idx]
        yi1 = y_knots[idx+1]
        hi = h[idx]
        Mi = M[idx]
        Mi1 = M[idx+1]
        
        diff_x = x_eval - xi
        diff_x1 = xi1 - x_eval
        
        term1 = (Mi1 * diff_x**3 + Mi * diff_x1**3) / (6 * hi)
        term2 = (yi1/hi - Mi1*hi/6) * diff_x
        term3 = (yi/hi - Mi*hi/6) * diff_x1
        
        return term1 + term2 + term3

    def get_envelopes(self, x: np.ndarray) -> Dict:
        max_idx, min_idx = self._local_extrema(x)
        N = len(x)
        full_range = np.arange(N)

        if len(max_idx) < 2 or len(min_idx) < 2:
            max_idx_ext = np.unique(np.concatenate(([0], max_idx, [N-1])))
            min_idx_ext = np.unique(np.concatenate(([0], min_idx, [N-1])))
            env_max = np.interp(full_range, max_idx_ext, x[max_idx_ext])
            env_min = np.interp(full_range, min_idx_ext, x[min_idx_ext])
        else:
            # Extrapolation (Clamping ends)
            if max_idx[0] != 0: max_idx = np.insert(max_idx, 0, 0)
            if max_idx[-1] != N - 1: max_idx = np.append(max_idx, N - 1)
            if min_idx[0] != 0: min_idx = np.insert(min_idx, 0, 0)
            if min_idx[-1] != N - 1: min_idx = np.append(min_idx, N - 1)

            # Fast Manual Cubic Spline
            try:
                # Pastikan unique agar h tidak nol
                max_idx_u, u_ind = np.unique(max_idx, return_index=True)
                min_idx_u, u_min_ind = np.unique(min_idx, return_index=True)
                
                env_max = self._cubic_spline_interpolation(max_idx_u, x[max_idx_u], full_range)
                env_min = self._cubic_spline_interpolation(min_idx_u, x[min_idx_u], full_range)
            except Exception:
                # Fallback ke linear jika terjadi error numerik
                env_max = np.interp(full_range, max_idx, x[max_idx])
                env_min = np.interp(full_range, min_idx, x[min_idx])

        mean_env = 0.5 * (env_max + env_min)
        return {'max_idx': max_idx, 'min_idx': min_idx, 'env_max': env_max, 'env_min': env_min, 'mean_env': mean_env}

    # IMF Condition
    def _count_zero_crossings(self, x: np.ndarray) -> int:
        s = np.sign(x)
        s[s==0] = 1 
        return len(np.where(np.diff(s))[0])

    def _check_imf_condition(self, x: np.ndarray) -> bool:
        zc = self._count_zero_crossings(x)
        max_idx, min_idx = self._local_extrema(x)
        extrema = len(max_idx) + len(min_idx)
        return abs(zc - extrema) <= 1

    # EMD Algorithm 
    def emd(self, signal: np.ndarray, max_imfs: int = 10, max_siftings: int = 50,
            epsilon: Optional[float] = 0.2, debug: bool = False,
            use_sg_for_extrema: bool = False
           ) -> Tuple[List[np.ndarray], List[Dict]]:
        
        x = signal.copy().astype(np.float64)
        imfs = []
        meta = []
        residual = x.copy()
        N_STABLE_IMF_CHECKS = 3

        for imf_index in range(max_imfs):
            proto = residual.copy()
            max_idx_r, min_idx_r = self._local_extrema(proto)
            if len(max_idx_r) < 2 or len(min_idx_r) < 2:
                break
            
            h = proto.copy()
            sifts_done = 0
            final_sd = 0.0
            stop_reason = 'max_siftings'
            consecutive_imf_meets = 0

            for sift in range(1, max_siftings + 1):
                h_prev = h.copy()
                
                max_idx_h, min_idx_h = self._local_extrema(h_prev)
                if len(max_idx_h) < 1 or len(min_idx_h) < 1:
                    stop_reason = 'monotonic_h'
                    break

                env_info = self.get_envelopes(h_prev)
                m = env_info['mean_env']
                h = h_prev - m
                
                denom = np.sum(h_prev ** 2)
                numer = np.sum((h_prev - h) ** 2)
                sd = float(numer / denom) if denom > 1e-12 else 0.0
                
                sifts_done = sift
                final_sd = sd
                
                is_imf_now = self._check_imf_condition(h)
                if is_imf_now:
                    consecutive_imf_meets += 1
                else:
                    consecutive_imf_meets = 0

                if (epsilon is not None and sd < epsilon) or (consecutive_imf_meets >= N_STABLE_IMF_CHECKS):
                    stop_reason = 'epsilon' if (epsilon is not None and sd < epsilon) else 'stable_imf'
                    break

            imfs.append(h.copy())
            meta.append({'sifts': sifts_done, 'final_sd': final_sd, 'stop_reason': stop_reason})
            
            residual = residual - h
            
            max_idx_r, min_idx_r = self._local_extrema(residual)
            if len(max_idx_r) < 2 or len(min_idx_r) < 2:
                break

        if np.std(residual) > 1e-10:
            imfs.append(residual)
            meta.append({'sifts': 0, 'final_sd': 0, 'stop_reason': 'residual'})
            
        return imfs, meta

    # EEMD // CEEMD // CEEMDAN
    def eemd(self, signal: np.ndarray, trials: int = 50, noise_std_ratio: float = 0.2,
             max_imfs: int = 10, max_siftings: int = 50, epsilon: Optional[float] = 0.2,
             debug: bool = False, rng_seed: Optional[int] = 12345,
             use_sg_for_extrema: bool = False) -> Tuple[List[np.ndarray], List[Dict]]:
        
        N = len(signal)
        rng = np.random.default_rng(rng_seed)
        std_sig = np.std(signal)
        
        acc_imfs = np.zeros((max_imfs, N))
        stored_noises = [] 
        
        for tr in range(trials):
            noise = rng.normal(0, noise_std_ratio * std_sig, size=N)
            stored_noises.append(noise.copy()) 
            
            imfs_tr, _ = self.emd(signal + noise, max_imfs=max_imfs, 
                                  max_siftings=max_siftings, epsilon=epsilon)
            for i, imf in enumerate(imfs_tr):
                if i < max_imfs:
                    acc_imfs[i] += imf

        avg_modes = []
        avg_meta = []
        current_residual = signal.copy()
        
        for i in range(max_imfs):
            if np.all(np.abs(acc_imfs[i]) < 1e-15): break
            imf_avg = acc_imfs[i] / trials
            avg_modes.append(imf_avg)
            current_residual -= imf_avg
            
            # Menambahkan sifts_mean = 0 agar GUI tidak error
            meta_entry = {'method': 'EEMD', 'sifts_mean': 0}
            if i == 0: meta_entry['noise_trials'] = stored_noises
            avg_meta.append(meta_entry)
        
        if np.std(current_residual) > 1e-10:
            avg_modes.append(current_residual)
            avg_meta.append({'method': 'residual', 'sifts_mean': 0})
            
        return avg_modes, avg_meta

    def ceemd(self, signal: np.ndarray, trials: int = 50, noise_std_ratio: float = 0.2,
              max_imfs: int = 10, max_siftings: int = 50, epsilon: Optional[float] = 0.2,
              debug: bool = False, rng_seed: Optional[int] = 54321,
              use_sg_for_extrema: bool = False) -> Tuple[List[np.ndarray], List[Dict]]:
        
        N = len(signal)
        rng = np.random.default_rng(rng_seed)
        std_sig = np.std(signal)
        acc_imfs = np.zeros((max_imfs, N))
        stored_pairs = []
        
        for tr in range(trials):
            noise = rng.normal(0, noise_std_ratio * std_sig, size=N)
            stored_pairs.append((noise.copy(), -noise.copy())) 
            
            imfs_p, _ = self.emd(signal + noise, max_imfs=max_imfs, max_siftings=max_siftings, epsilon=epsilon)
            imfs_m, _ = self.emd(signal - noise, max_imfs=max_imfs, max_siftings=max_siftings, epsilon=epsilon)
            
            for i, imf in enumerate(imfs_p):
                if i < max_imfs: acc_imfs[i] += imf
            for i, imf in enumerate(imfs_m):
                if i < max_imfs: acc_imfs[i] += imf
                
        avg_modes = []
        avg_meta = []
        current_residual = signal.copy()
        total_runs = trials * 2
        
        for i in range(max_imfs):
            if np.all(np.abs(acc_imfs[i]) < 1e-15): break
            imf_avg = acc_imfs[i] / total_runs
            avg_modes.append(imf_avg)
            current_residual -= imf_avg
            
            meta_entry = {'method': 'CEEMD', 'sifts_mean': 0}
            if i == 0: meta_entry['noise_pairs'] = stored_pairs
            avg_meta.append(meta_entry)
            
        if np.std(current_residual) > 1e-10:
            avg_modes.append(current_residual)
            avg_meta.append({'method': 'residual', 'sifts_mean': 0})

        return avg_modes, avg_meta

    def ceemdan(self, signal: np.ndarray, trials: int = 50, noise_std_ratio: float = 0.2,
                max_imfs: int = 10, max_siftings: int = 50, epsilon: Optional[float] = 0.2,
                debug: bool = False, rng_seed: Optional[int] = 999,
                use_sg_for_extrema: bool = False) -> Tuple[List[np.ndarray], List[Dict]]:
        
        N = len(signal)
        rng = np.random.default_rng(rng_seed)
        std_sig = np.std(signal)
        
        modes = []
        metas = []
        
        # IMF 1
        acc_imf1 = np.zeros(N)
        adaptive_noises_stage1 = [] 
        
        for tr in range(trials):
            noise = rng.normal(0, noise_std_ratio * std_sig, size=N)
            adaptive_noises_stage1.append(noise.copy())
            imfs_tr, _ = self.emd(signal + noise, max_imfs=1, max_siftings=max_siftings, epsilon=epsilon)
            if len(imfs_tr) > 0:
                acc_imf1 += imfs_tr[0]
        
        imf1 = acc_imf1 / trials
        modes.append(imf1)
        # Penting: Tambahkan sifts_mean untuk GUI
        metas.append({'stage': 1, 'adaptive_noises': adaptive_noises_stage1, 'sifts_mean': 0})
        residual = signal - imf1
        
        # IMF 2 until K
        for k in range(1, max_imfs):
            max_idx, min_idx = self._local_extrema(residual)
            if len(max_idx) < 2 or len(min_idx) < 2: break
            acc_imf_k = np.zeros(N)
            adaptive_noises_stage_k = [] 
            
            for tr in range(trials):
                noise = rng.normal(0, noise_std_ratio * std_sig, size=N)
                noise_imfs, _ = self.emd(noise, max_imfs=k+1, max_siftings=10, epsilon=0.5)
                
                if len(noise_imfs) > k: noise_comp = noise_imfs[k]
                elif len(noise_imfs) > 0: noise_comp = noise_imfs[-1]
                else: noise_comp = np.zeros(N)

                added_noise = noise_std_ratio * std_sig * noise_comp
                adaptive_noises_stage_k.append(added_noise.copy())
                
                inp = residual + added_noise
                imfs_res, _ = self.emd(inp, max_imfs=1, max_siftings=max_siftings, epsilon=epsilon)
                if len(imfs_res) > 0: acc_imf_k += imfs_res[0]
            
            imf_k = acc_imf_k / trials
            modes.append(imf_k)
            # Penting: Tambahkan sifts_mean untuk GUI
            metas.append({'stage': k+1, 'adaptive_noises': adaptive_noises_stage_k, 'sifts_mean': 0})
            residual = residual - imf_k
            
        modes.append(residual)
        metas.append({'type': 'residual', 'sifts_mean': 0})
        
        return modes, metas

    # FFT and DSP Utilities
    def _next_pow2(self, n: int) -> int:
        p = 1
        while p < n: p <<= 1
        return p
        
    def _bit_reverse_permutation(self, n: int) -> np.ndarray:
        bits = int(np.log2(n)); perm = np.zeros(n, dtype=int)
        for i in range(n):
            rev = 0; x = i
            for _ in range(bits): rev = (rev << 1) | (x & 1); x >>= 1
            perm[i] = rev
        return perm
        
    def fft(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex); n0 = x.shape[0]; N = self._next_pow2(n0)
        if N != n0: x = np.concatenate([x, np.zeros(N - n0, dtype=complex)])
        perm = self._bit_reverse_permutation(N); X = x[perm].copy(); m = 1
        while m < N:
            step = m * 2; ang = -2j * np.pi / step; W_m = np.exp(ang * np.arange(m))
            for k in range(0, N, step):
                t = W_m * X[k + m:k + step]; u = X[k:k + m]; X[k:k + m] = u + t; X[k + m:k + step] = u - t
            m = step
        return X
        
    def ifft(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=complex); x = np.conjugate(self.fft(np.conjugate(X))); x = x / X.shape[0]; return x
        
    def dft(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = len(x)
        if self.transform_method == "FFT":
            Xfull = self.fft(x); freqs = np.arange(0, len(Xfull)) * (self.fs / len(Xfull))
        else:
            n = np.arange(N); k = np.arange(N); exp = np.exp(-2j * np.pi * np.outer(k, n) / N); Xfull = exp.dot(x); freqs = np.arange(0, N) * (self.fs / N)
        return freqs, Xfull
        
    def idft(self, Xfull: np.ndarray) -> np.ndarray:
        N = len(Xfull)
        if self.transform_method == "FFT": x_recon = self.ifft(Xfull)
        else: k = np.arange(N); n = np.arange(N); exp = np.exp(2j * np.pi * np.outer(n, k) / N); x_recon = exp.dot(Xfull) / N
        return x_recon
        
    def psd_from_spectrum(self, Xk: np.ndarray, N: int) -> np.ndarray:
        return (np.abs(Xk) ** 2) / N
        
    def analytic_signal(self, x: np.ndarray) -> np.ndarray:
        N0 = len(x); freqs, Xfull = self.dft(x); N = len(Xfull); H = np.zeros(N, dtype=complex)
        if N % 2 == 0: H[0] = Xfull[0]; H[N//2] = Xfull[N//2]; H[1:N//2] = 2 * Xfull[1:N//2]
        else: H[0] = Xfull[0]; H[1:(N+1)//2] = 2 * Xfull[1:(N+1)//2]
        x_analytic_full = self.idft(H); return x_analytic_full[:N0]
        
    def inst_amplitude_phase_freq(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = self.analytic_signal(x); a = np.abs(z); phase = np.unwrap(np.angle(z)); dphase = np.gradient(phase); inst_freq = (dphase / (2.0 * np.pi)) * self.fs; return a, phase, inst_freq
        
    def _gaussian_kernel_1d(self, sigma: float, truncate: float = 3.0) -> np.ndarray:
        if sigma <= 0: return np.array([1.0])
        radius = int(max(1, np.ceil(truncate * sigma))); x = np.arange(-radius, radius + 1); kernel = np.exp(-0.5 * (x / sigma) ** 2); kernel = kernel / np.sum(kernel); return kernel
        
    def gaussian_blur_2d(self, img: np.ndarray, sigma_time: float = 2.0, sigma_freq: float = 1.0) -> np.ndarray:
        if sigma_time <= 0 and sigma_freq <= 0: return img.copy()
        out = img.copy()
        if sigma_time > 0:
            k_time = self._gaussian_kernel_1d(sigma_time)
            for r in range(out.shape[0]): out[r, :] = np.convolve(out[r, :], k_time, mode='same')
        if sigma_freq > 0:
            k_freq = self._gaussian_kernel_1d(sigma_freq)
            for c in range(out.shape[1]): out[:, c] = np.convolve(out[:, c], k_freq, mode='same')
        return out
        
    def hht_spectrogram(self, imf: np.ndarray, f_min: float = 0.0, f_max: float = None, freq_bins: int = 128, smooth_sigma_time: float = 2.0, smooth_sigma_freq: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if f_max is None: f_max = 0.5 * self.fs
        a, phase, inst_freq = self.inst_amplitude_phase_freq(imf); inst_freq_clipped = np.clip(inst_freq, f_min, f_max); freq_edges = np.linspace(f_min, f_max, freq_bins + 1); freq_centers = 0.5 * (freq_edges[:-1] + freq_edges[1:]); N = len(imf); amp_map = np.zeros((freq_bins, N)); bin_idx = np.searchsorted(freq_edges, inst_freq_clipped, side='right') - 1; bin_idx = np.clip(bin_idx, 0, freq_bins - 1)
        for ti in range(N): amp_map[bin_idx[ti], ti] += a[ti]
        if smooth_sigma_time > 0 or smooth_sigma_freq > 0: amp_map = self.gaussian_blur_2d(amp_map, sigma_time=smooth_sigma_time, sigma_freq=smooth_sigma_freq)
        return np.arange(N) / self.fs, freq_centers, amp_map

    # Extraction
    def extract_resp_vaso(self, imfs: List[np.ndarray], resp_band: Tuple[float, float] = (0.15, 0.4), vaso_band: Tuple[float, float] = (0.04, 0.15)) -> Dict:
        resp_energies = []; vaso_energies = []
        resp_peak_freqs = []; vaso_peak_freqs = [] 
        
        for imf in imfs:
            freqs, Xk = self.dft(imf)
            psd = self.psd_from_spectrum(Xk, len(imf))
            half = len(freqs)//2 + 1
            freqs_half = freqs[:half]; psd_half = psd[:half]
            
            resp_mask = (freqs_half >= resp_band[0]) & (freqs_half <= resp_band[1])
            vaso_mask = (freqs_half >= vaso_band[0]) & (freqs_half <= vaso_band[1])
            
            resp_energy = np.sum(psd_half[resp_mask]) if np.any(resp_mask) else 0.0
            vaso_energy = np.sum(psd_half[vaso_mask]) if np.any(vaso_mask) else 0.0
            
            if np.any(resp_mask) and np.sum(psd_half[resp_mask]) > 0:
                idx_peak = np.argmax(psd_half[resp_mask])
                resp_peak_freqs.append(freqs_half[resp_mask][idx_peak])
            else: resp_peak_freqs.append(0.0)

            if np.any(vaso_mask) and np.sum(psd_half[vaso_mask]) > 0:
                idx_peak_vaso = np.argmax(psd_half[vaso_mask])
                vaso_peak_freqs.append(freqs_half[vaso_mask][idx_peak_vaso])
            else: vaso_peak_freqs.append(0.0)

            resp_energies.append(resp_energy)
            vaso_energies.append(vaso_energy)
            
        if len(resp_energies) == 0: return {}
        
        resp_idx = int(np.argmax(resp_energies))
        vaso_idx = int(np.argmax(vaso_energies))
        resp_freq_hz = resp_peak_freqs[resp_idx]
        resp_bpm = resp_freq_hz * 60.0
        vaso_freq_hz = vaso_peak_freqs[vaso_idx] 
        total_energy = np.sum(resp_energies) + np.sum(vaso_energies) + 1e-12
        
        return {
            'respiratory_rate_bpm': resp_bpm, 
            'respiratory_freq_hz': resp_freq_hz,
            'resp_imf_index': resp_idx, 
            'resp_energy': resp_energies[resp_idx], 
            'vasomotor_freq_hz': vaso_freq_hz,
            'vaso_imf_index': vaso_idx, 
            'vaso_energy': vaso_energies[vaso_idx], 
            'resp_energy_norm': resp_energies[resp_idx] / total_energy, 
            'vaso_energy_norm': vaso_energies[vaso_idx] / total_energy
        }