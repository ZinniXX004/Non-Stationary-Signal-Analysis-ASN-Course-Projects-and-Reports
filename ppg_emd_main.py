# ppg_emd_main.py
# Backend processing for PPG EMD pipelines
# EMD core with debug mode returning proto & envelope info per IMF
# EEMD / CEEMD / CEEMDAN return averaged modes and add debug proto/envelope info
# Dependencies: numpy, pandas

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

    # --------------------------
    # I/O
    # --------------------------
    def load_csv(self, path: str, column: str = "PLETH", time_column: Optional[str] = None):
        df = pd.read_csv(path)
        if time_column is None:
            possible = [c for c in df.columns if 'time' in c.lower() or 't[' in c.lower() or 'sec' in c.lower()]
            time_column = possible[0] if possible else df.columns[0]
        self.t = df[time_column].values
        self.raw = df[column].values.astype(np.float64)
        return self.t, self.raw

    # --------------------------
    # Preprocessing helpers
    # --------------------------
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
        x = sig.copy().astype(np.float64)
        x = x - np.mean(x)
        if remove_baseline:
            win = max(3, int(round(baseline_window_s * self.fs)))
            baseline = self.running_mean(x, win)
            x = x - baseline
        if normalize:
            s = np.std(x)
            if s > 0:
                x = x / s
        self.preproc = x
        return x

    def preproc_std(self) -> float:
        if self.preproc is None:
            return float('nan')
        return float(np.std(self.preproc))

    # --------------------------
    # Savitzky-Golay (numpy)
    # --------------------------
    def savgol_coeffs(self, window_length: int, polyorder: int) -> np.ndarray:
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length")
        half = (window_length - 1) // 2
        A = np.vander(np.arange(-half, half + 1), polyorder + 1, increasing=True)
        ATA_pinv = np.linalg.pinv(A)
        coeffs = ATA_pinv[0, :]
        return coeffs[::-1]

    def savgol_filter(self, x: np.ndarray, window_length: int = 7, polyorder: int = 3) -> np.ndarray:
        n = len(x)
        if window_length <= 1 or window_length >= n:
            return x.copy()
        if window_length % 2 == 0:
            window_length += 1
        coeffs = self.savgol_coeffs(window_length, polyorder)
        half = (window_length - 1) // 2
        # reflect padding
        left = x[1:half+1][::-1] if half >= 1 else np.array([])
        right = x[-half-1:-1][::-1] if half >= 1 else np.array([])
        xp = np.concatenate([left, x, right])
        y = np.convolve(xp, coeffs, mode='valid')
        return y[:n].copy()

    # --------------------------
    # Extrema detection (robust)
    # --------------------------
    def _local_extrema(self, x: np.ndarray, use_sg: bool = False, sg_win: int = 7, sg_poly: int = 3
                     ) -> Tuple[np.ndarray, np.ndarray]:
        if use_sg:
            xs = self.savgol_filter(x, window_length=sg_win, polyorder=sg_poly)
        else:
            xs = x
        N = len(xs)
        maxima = []
        minima = []
        i = 0
        while i < N:
            if i == 0:
                if N >= 2:
                    if xs[0] > xs[1]:
                        maxima.append(0)
                    elif xs[0] < xs[1]:
                        minima.append(0)
                i += 1
                continue
            if i == N - 1:
                if xs[-1] > xs[-2]:
                    maxima.append(N-1)
                elif xs[-1] < xs[-2]:
                    minima.append(N-1)
                i += 1
                continue
            if xs[i] == xs[i+1] or xs[i] == xs[i-1]:
                L = i
                while L - 1 >= 0 and xs[L - 1] == xs[i]:
                    L -= 1
                R = i
                while R + 1 < N and xs[R + 1] == xs[i]:
                    R += 1
                mid = (L + R) // 2
                left_neighbor = xs[L - 1] if L - 1 >= 0 else xs[L]
                right_neighbor = xs[R + 1] if R + 1 < N else xs[R]
                if xs[mid] >= left_neighbor and xs[mid] >= right_neighbor:
                    maxima.append(mid)
                if xs[mid] <= left_neighbor and xs[mid] <= right_neighbor:
                    minima.append(mid)
                i = R + 1
                continue
            if xs[i] > xs[i-1] and xs[i] > xs[i+1]:
                maxima.append(i)
            elif xs[i] < xs[i-1] and xs[i] < xs[i+1]:
                minima.append(i)
            i += 1
        maxima = np.unique(np.array(maxima, dtype=int))
        minima = np.unique(np.array(minima, dtype=int))
        if maxima.size == 0 and minima.size == 0:
            maxima = np.array([int(np.argmax(xs))], dtype=int)
            minima = np.array([int(np.argmin(xs))], dtype=int)
        return maxima, minima

    def get_envelopes(self, x: np.ndarray, use_sg: bool = False, sg_win: int = 7, sg_poly: int = 3) -> Dict:
        max_idx, min_idx = self._local_extrema(x, use_sg=use_sg, sg_win=sg_win, sg_poly=sg_poly)
        N = len(x)
        if max_idx.size == 0:
            max_idx = np.array([0, N-1], dtype=int)
        if min_idx.size == 0:
            min_idx = np.array([0, N-1], dtype=int)
        if max_idx[0] != 0:
            max_idx = np.insert(max_idx, 0, 0)
        if max_idx[-1] != N-1:
            max_idx = np.append(max_idx, N-1)
        if min_idx[0] != 0:
            min_idx = np.insert(min_idx, 0, 0)
        if min_idx[-1] != N-1:
            min_idx = np.append(min_idx, N-1)
        env_max = np.interp(np.arange(N), max_idx, x[max_idx])
        env_min = np.interp(np.arange(N), min_idx, x[min_idx])
        mean_env = 0.5 * (env_max + env_min)
        return {'max_idx': max_idx, 'min_idx': min_idx, 'env_max': env_max, 'env_min': env_min, 'mean_env': mean_env}

    # --------------------------
    # IMF condition and helpers
    # --------------------------
    def _count_zero_crossings(self, x: np.ndarray) -> int:
        s = np.sign(x)
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i-1] if s[i-1] != 0 else 1
        zc = np.sum(np.abs(np.diff(np.sign(s))) > 0)
        return int(zc)

    def _is_imf(self, x: np.ndarray) -> bool:
        zc = self._count_zero_crossings(x)
        max_idx, min_idx = self._local_extrema(x, use_sg=False)
        extrema = int(len(max_idx) + len(min_idx))
        cond1 = abs(zc - extrema) <= 1
        mean_env = self._envelope_mean(x)
        mad = np.mean(np.abs(mean_env))
        s = np.std(x) if np.std(x) > 1e-12 else 1e-12
        cond2 = mad < 0.1 * s
        return cond1 and cond2

    def _envelope_mean(self, x: np.ndarray) -> np.ndarray:
        max_idx, min_idx = self._local_extrema(x, use_sg=False)
        N = len(x)
        if len(max_idx) < 2 or len(min_idx) < 2:
            return np.full(N, np.mean(x))
        env_max = np.interp(np.arange(N), max_idx, x[max_idx])
        env_min = np.interp(np.arange(N), min_idx, x[min_idx])
        return 0.5 * (env_max + env_min)

    # --------------------------
    # EMD algorithm with debug option
    # --------------------------
    def emd(self, signal: np.ndarray, max_imfs: int = 10, max_siftings: int = 50,
            epsilon: Optional[float] = None, debug: bool = False,
            use_sg_for_extrema: bool = True, sg_window: int = 7, sg_poly: int = 3
           ) -> Tuple[List[np.ndarray], List[Dict]]:
        x = signal.copy().astype(np.float64)
        N = len(x)
        imfs = []
        meta = []
        residual = x.copy()

        for imf_index in range(max_imfs):
            # proto is residual prior to extracting this IMF
            proto = residual.copy()
            max_idx_r, min_idx_r = self._local_extrema(proto, use_sg=use_sg_for_extrema, sg_win=sg_window, sg_poly=sg_poly)
            if len(max_idx_r) + len(min_idx_r) < 2:
                break
            h = proto.copy()
            sifts_done = 0
            final_sd = 0.0
            stop_reason = 'max_siftings'
            for sift in range(1, max_siftings + 1):
                h_prev = h.copy()
                env_info = self.get_envelopes(h_prev, use_sg=use_sg_for_extrema, sg_win=sg_window, sg_poly=sg_poly)
                m = env_info['mean_env']
                h = h_prev - m
                denom = np.sum(h_prev ** 2)
                numer = np.sum((h_prev - h) ** 2)
                sd = float(numer / denom) if denom > 0 else float('inf')
                sifts_done = sift
                final_sd = sd
                if self._is_imf(h):
                    stop_reason = 'imf_cond'
                    break
                if (epsilon is not None) and (sd < epsilon):
                    stop_reason = 'epsilon'
                    break
            imfs.append(h.copy())
            meta_entry = {'sifts': sifts_done, 'final_sd': final_sd, 'stop_reason': stop_reason}
            if debug:
                # store proto and envelope info computed on proto
                env_proto = self.get_envelopes(proto, use_sg=use_sg_for_extrema, sg_win=sg_window, sg_poly=sg_poly)
                meta_entry['proto'] = proto.copy()
                meta_entry['env_info'] = {
                    'max_idx': env_proto['max_idx'],
                    'min_idx': env_proto['min_idx'],
                    'env_max': env_proto['env_max'],
                    'env_min': env_proto['env_min'],
                    'mean_env': env_proto['mean_env']
                }
            meta.append(meta_entry)
            residual = residual - h
            max_idx_r, min_idx_r = self._local_extrema(residual, use_sg=use_sg_for_extrema, sg_win=sg_window, sg_poly=sg_poly)
            if len(max_idx_r) + len(min_idx_r) < 2:
                break
        return imfs, meta

    # --------------------------
    # EEMD (ensemble) with debug proto calculation
    # --------------------------
    def eemd(self, signal: np.ndarray, trials: int = 50, noise_std_ratio: float = 0.2,
             max_imfs: int = 10, max_siftings: int = 50, epsilon: Optional[float] = None,
             debug: bool = False, rng_seed: Optional[int] = 12345,
             use_sg_for_extrema: bool = True, sg_window: int = 7, sg_poly: int = 3) -> Tuple[List[np.ndarray], List[Dict]]:
        N = len(signal)
        rng = np.random.default_rng(rng_seed)
        modes_per_trial = []
        meta_per_trial = []
        noises = []
        for tr in range(trials):
            noise = rng.normal(0, noise_std_ratio * np.std(signal), size=N)
            noises.append(noise.copy())
            imfs_tr, meta_tr = self.emd(signal + noise, max_imfs=max_imfs, max_siftings=max_siftings,
                                        epsilon=epsilon, debug=False,
                                        use_sg_for_extrema=use_sg_for_extrema, sg_window=sg_window, sg_poly=sg_poly)
            modes_per_trial.append(imfs_tr)
            meta_per_trial.append(meta_tr)
        max_modes = max(len(m) for m in modes_per_trial)
        avg_modes = []
        avg_meta = []
        for m in range(max_modes):
            acc = np.zeros(N)
            count = 0
            sifts = []
            sds = []
            reasons = []
            for tr in range(trials):
                if len(modes_per_trial[tr]) > m:
                    acc += modes_per_trial[tr][m]
                    md = meta_per_trial[tr][m]
                    sifts.append(md.get('sifts', 0))
                    sds.append(md.get('final_sd', 0.0))
                    reasons.append(md.get('stop_reason', ''))
                    count += 1
            if count == 0:
                break
            avg_mode = acc / count
            avg_modes.append(avg_mode)
            meta_entry = {'sifts_mean': float(np.mean(sifts)) if sifts else 0.0,
                          'final_sd_mean': float(np.mean(sds)) if sds else 0.0,
                          'stop_reasons': reasons}
            # debug: compute proto for averaged mode (pre - sum previous avg modes)
            if debug:
                # reconstruct proto from signal and average modes up to m-1
                proto = signal.copy()
                for k in range(0, m):
                    proto = proto - avg_modes[k]
                env_proto = self.get_envelopes(proto, use_sg=use_sg_for_extrema, sg_win=sg_window, sg_poly=sg_poly)
                meta_entry['proto'] = proto.copy()
                meta_entry['env_info'] = {
                    'max_idx': env_proto['max_idx'],
                    'min_idx': env_proto['min_idx'],
                    'env_max': env_proto['env_max'],
                    'env_min': env_proto['env_min'],
                    'mean_env': env_proto['mean_env']
                }
            meta_entry['noise_trials'] = noises  # include raw noises for visualization if needed
            avg_meta.append(meta_entry)
        return avg_modes, avg_meta

    # --------------------------
    # CEEMD (complementary) with debug
    # --------------------------
    def ceemd(self, signal: np.ndarray, trials: int = 50, noise_std_ratio: float = 0.2,
              max_imfs: int = 10, max_siftings: int = 50, epsilon: Optional[float] = None,
              debug: bool = False, rng_seed: Optional[int] = 54321,
              use_sg_for_extrema: bool = True, sg_window: int = 7, sg_poly: int = 3) -> Tuple[List[np.ndarray], List[Dict]]:
        N = len(signal)
        rng = np.random.default_rng(rng_seed)
        trial_averaged_modes = []
        trial_meta = []
        noise_pairs = []
        for tr in range(trials):
            noise = rng.normal(0, noise_std_ratio * np.std(signal), size=N)
            noise_pairs.append((noise.copy(), (-noise).copy()))
            imfs_plus, meta_plus = self.emd(signal + noise, max_imfs=max_imfs, max_siftings=max_siftings,
                                            epsilon=epsilon, debug=False,
                                            use_sg_for_extrema=use_sg_for_extrema, sg_window=sg_window, sg_poly=sg_poly)
            imfs_minus, meta_minus = self.emd(signal - noise, max_imfs=max_imfs, max_siftings=max_siftings,
                                              epsilon=epsilon, debug=False,
                                              use_sg_for_extrema=use_sg_for_extrema, sg_window=sg_window, sg_poly=sg_poly)
            ml = max(len(imfs_plus), len(imfs_minus))
            averaged_modes = []
            averaged_mmeta = []
            for m in range(ml):
                a = imfs_plus[m] if m < len(imfs_plus) else np.zeros(N)
                b = imfs_minus[m] if m < len(imfs_minus) else np.zeros(N)
                mdp = meta_plus[m] if m < len(meta_plus) else {'sifts':0,'final_sd':0.0,'stop_reason':''}
                mdm = meta_minus[m] if m < len(meta_minus) else {'sifts':0,'final_sd':0.0,'stop_reason':''}
                averaged_modes.append(0.5 * (a + b))
                averaged_mmeta.append({'sifts_mean': 0.5 * (mdp.get('sifts',0) + mdm.get('sifts',0)),
                                       'final_sd_mean': 0.5 * (mdp.get('final_sd',0.0) + mdm.get('final_sd',0.0)),
                                       'stop_reasons': [mdp.get('stop_reason',''), mdm.get('stop_reason','')]})
            trial_averaged_modes.append(averaged_modes)
            trial_meta.append(averaged_mmeta)
        max_modes = max(len(modes) for modes in trial_averaged_modes)
        avg_modes = []
        avg_meta = []
        for m in range(max_modes):
            acc = np.zeros(N)
            sifts_list = []
            sd_list = []
            reasons = []
            count = 0
            for tr in range(trials):
                if len(trial_averaged_modes[tr]) > m:
                    acc += trial_averaged_modes[tr][m]
                    md = trial_meta[tr][m]
                    sifts_list.append(md.get('sifts_mean', 0.0))
                    sd_list.append(md.get('final_sd_mean', 0.0))
                    reasons.extend(md.get('stop_reasons', []))
                    count += 1
            if count == 0:
                break
            avg_mode = acc / count
            avg_modes.append(avg_mode)
            meta_entry = {'sifts_mean': float(np.mean(sifts_list)) if sifts_list else 0.0,
                          'final_sd_mean': float(np.mean(sd_list)) if sd_list else 0.0,
                          'stop_reasons': reasons}
            if debug:
                proto = signal.copy()
                for k in range(0, m):
                    proto = proto - avg_modes[k]
                env_proto = self.get_envelopes(proto, use_sg=use_sg_for_extrema, sg_win=sg_window, sg_poly=sg_poly)
                meta_entry['proto'] = proto.copy()
                meta_entry['env_info'] = {
                    'max_idx': env_proto['max_idx'],
                    'min_idx': env_proto['min_idx'],
                    'env_max': env_proto['env_max'],
                    'env_min': env_proto['env_min'],
                    'mean_env': env_proto['mean_env']
                }
            meta_entry['noise_pairs'] = noise_pairs
            avg_meta.append(meta_entry)
        return avg_modes, avg_meta

    # --------------------------
    # CEEMDAN (adaptive) with debug
    # --------------------------
    def ceemdan(self, signal: np.ndarray, trials: int = 50, noise_std_ratio: float = 0.2,
                max_imfs: int = 10, max_siftings: int = 50, epsilon: Optional[float] = None,
                debug: bool = False, rng_seed: Optional[int] = 999,
                use_sg_for_extrema: bool = True, sg_window: int = 7, sg_poly: int = 3) -> Tuple[List[np.ndarray], List[Dict]]:
        N = len(signal)
        rng = np.random.default_rng(rng_seed)
        residual = signal.copy()
        modes = []
        metas = []
        adaptive_noises = []
        for k in range(max_imfs):
            std_r = np.std(residual) if np.std(residual) > 1e-12 else 1e-12
            acc = np.zeros(N)
            sifts_list = []
            sd_list = []
            stop_reasons = []
            trial_noises = []
            for tr in range(trials):
                noise = rng.normal(0, noise_std_ratio * std_r, size=N)
                trial_noises.append(noise.copy())
                imfs_tr, meta_tr = self.emd(residual + noise, max_imfs=1, max_siftings=max_siftings, epsilon=epsilon,
                                            debug=False, use_sg_for_extrema=use_sg_for_extrema, sg_window=sg_window, sg_poly=sg_poly)
                if len(imfs_tr) > 0:
                    acc += imfs_tr[0]
                    mtr = meta_tr[0]
                    sifts_list.append(mtr.get('sifts', 0))
                    sd_list.append(mtr.get('final_sd', 0.0))
                    stop_reasons.append(mtr.get('stop_reason',''))
            if len(sifts_list) == 0:
                break
            mode_k = acc / len(sifts_list)
            modes.append(mode_k)
            meta_entry = {'sifts_mean': float(np.mean(sifts_list)), 'final_sd_mean': float(np.mean(sd_list)), 'stop_reasons': stop_reasons}
            adaptive_noises.append(trial_noises)
            if debug:
                proto = residual.copy()
                env_proto = self.get_envelopes(proto, use_sg=use_sg_for_extrema, sg_win=sg_window, sg_poly=sg_poly)
                meta_entry['proto'] = proto.copy()
                meta_entry['env_info'] = {
                    'max_idx': env_proto['max_idx'],
                    'min_idx': env_proto['min_idx'],
                    'env_max': env_proto['env_max'],
                    'env_min': env_proto['env_min'],
                    'mean_env': env_proto['mean_env']
                }
            metas.append(meta_entry)
            residual = residual - mode_k
            max_idx_r, min_idx_r = self._local_extrema(residual, use_sg=use_sg_for_extrema, sg_win=sg_window, sg_poly=sg_poly)
            if len(max_idx_r) + len(min_idx_r) < 2:
                break
        for i in range(len(metas)):
            metas[i]['adaptive_noises'] = adaptive_noises[i]
        return modes, metas

    # --------------------------
    # Fourier transforms
    # --------------------------
    def _next_pow2(self, n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p

    def _bit_reverse_permutation(self, n: int) -> np.ndarray:
        bits = int(np.log2(n))
        perm = np.zeros(n, dtype=int)
        for i in range(n):
            rev = 0
            x = i
            for _ in range(bits):
                rev = (rev << 1) | (x & 1)
                x >>= 1
            perm[i] = rev
        return perm

    def fft(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex)
        n0 = x.shape[0]
        N = self._next_pow2(n0)
        if N != n0:
            x = np.concatenate([x, np.zeros(N - n0, dtype=complex)])
        perm = self._bit_reverse_permutation(N)
        X = x[perm].copy()
        m = 1
        while m < N:
            step = m * 2
            ang = -2j * np.pi / step
            W_m = np.exp(ang * np.arange(m))
            for k in range(0, N, step):
                t = W_m * X[k + m:k + step]
                u = X[k:k + m]
                X[k:k + m] = u + t
                X[k + m:k + step] = u - t
            m = step
        return X

    def ifft(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=complex)
        x = np.conjugate(self.fft(np.conjugate(X)))
        x = x / X.shape[0]
        return x

    def dft(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = len(x)
        if self.transform_method == "FFT":
            Xfull = self.fft(x)
            freqs = np.arange(0, len(Xfull)) * (self.fs / len(Xfull))
        else:
            n = np.arange(N)
            k = np.arange(N)
            exp = np.exp(-2j * np.pi * np.outer(k, n) / N)
            Xfull = exp.dot(x)
            freqs = np.arange(0, N) * (self.fs / N)
        return freqs, Xfull

    def idft(self, Xfull: np.ndarray) -> np.ndarray:
        N = len(Xfull)
        if self.transform_method == "FFT":
            x_recon = self.ifft(Xfull)
        else:
            k = np.arange(N)
            n = np.arange(N)
            exp = np.exp(2j * np.pi * np.outer(n, k) / N)
            x_recon = exp.dot(Xfull) / N
        return x_recon

    def psd_from_spectrum(self, Xk: np.ndarray, N: int) -> np.ndarray:
        return (np.abs(Xk) ** 2) / N

    # --------------------------
    # Hilbert / analytic signal & HHT
    # --------------------------
    def analytic_signal(self, x: np.ndarray) -> np.ndarray:
        N0 = len(x)
        freqs, Xfull = self.dft(x)
        N = len(Xfull)
        H = np.zeros(N, dtype=complex)
        if N % 2 == 0:
            H[0] = Xfull[0]
            H[N//2] = Xfull[N//2]
            H[1:N//2] = 2 * Xfull[1:N//2]
        else:
            H[0] = Xfull[0]
            H[1:(N+1)//2] = 2 * Xfull[1:(N+1)//2]
        x_analytic_full = self.idft(H)
        return x_analytic_full[:N0]

    def inst_amplitude_phase_freq(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = self.analytic_signal(x)
        a = np.abs(z)
        phase = np.unwrap(np.angle(z))
        dphase = np.gradient(phase)
        inst_freq = (dphase / (2.0 * np.pi)) * self.fs
        return a, phase, inst_freq

    # --------------------------
    # Gaussian blur for HHT smoothing
    # --------------------------
    def _gaussian_kernel_1d(self, sigma: float, truncate: float = 3.0) -> np.ndarray:
        if sigma <= 0:
            return np.array([1.0])
        radius = int(max(1, np.ceil(truncate * sigma)))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / np.sum(kernel)
        return kernel

    def gaussian_blur_2d(self, img: np.ndarray, sigma_time: float = 2.0, sigma_freq: float = 1.0) -> np.ndarray:
        if sigma_time <= 0 and sigma_freq <= 0:
            return img.copy()
        out = img.copy()
        if sigma_time > 0:
            k_time = self._gaussian_kernel_1d(sigma_time)
            for r in range(out.shape[0]):
                out[r, :] = np.convolve(out[r, :], k_time, mode='same')
        if sigma_freq > 0:
            k_freq = self._gaussian_kernel_1d(sigma_freq)
            for c in range(out.shape[1]):
                out[:, c] = np.convolve(out[:, c], k_freq, mode='same')
        return out

    def hht_spectrogram(self, imf: np.ndarray, f_min: float = 0.0, f_max: float = None,
                        freq_bins: int = 128, smooth_sigma_time: float = 2.0, smooth_sigma_freq: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if f_max is None:
            f_max = 0.5 * self.fs
        a, phase, inst_freq = self.inst_amplitude_phase_freq(imf)
        inst_freq_clipped = np.clip(inst_freq, f_min, f_max)
        freq_edges = np.linspace(f_min, f_max, freq_bins + 1)
        freq_centers = 0.5 * (freq_edges[:-1] + freq_edges[1:])
        N = len(imf)
        amp_map = np.zeros((freq_bins, N))
        bin_idx = np.searchsorted(freq_edges, inst_freq_clipped, side='right') - 1
        bin_idx = np.clip(bin_idx, 0, freq_bins - 1)
        for ti in range(N):
            amp_map[bin_idx[ti], ti] += a[ti]
        if smooth_sigma_time > 0 or smooth_sigma_freq > 0:
            amp_map = self.gaussian_blur_2d(amp_map, sigma_time=smooth_sigma_time, sigma_freq=smooth_sigma_freq)
        return np.arange(N) / self.fs, freq_centers, amp_map

    # --------------------------
    # Respiration / vasomotor extraction
    # --------------------------
    def extract_resp_vaso(self, imfs: List[np.ndarray],
                          resp_band: Tuple[float, float] = (0.15, 0.4),
                          vaso_band: Tuple[float, float] = (0.04, 0.15)) -> Dict:
        resp_energies = []
        vaso_energies = []
        resp_peak_freqs = []
        for imf in imfs:
            freqs, Xk = self.dft(imf)
            psd = self.psd_from_spectrum(Xk, len(imf))
            half = len(freqs)//2 + 1
            freqs_half = freqs[:half]
            psd_half = psd[:half]
            resp_mask = (freqs_half >= resp_band[0]) & (freqs_half <= resp_band[1])
            vaso_mask = (freqs_half >= vaso_band[0]) & (freqs_half <= vaso_band[1])
            resp_energy = np.sum(psd_half[resp_mask]) if np.any(resp_mask) else 0.0
            vaso_energy = np.sum(psd_half[vaso_mask]) if np.any(vaso_mask) else 0.0
            if np.any(resp_mask):
                idx_peak = np.argmax(psd_half[resp_mask])
                resp_peak_freqs.append(freqs_half[resp_mask][idx_peak])
            else:
                resp_peak_freqs.append(0.0)
            resp_energies.append(resp_energy)
            vaso_energies.append(vaso_energy)
        if len(resp_energies) == 0:
            return {}
        resp_idx = int(np.argmax(resp_energies))
        vaso_idx = int(np.argmax(vaso_energies))
        resp_freq_hz = resp_peak_freqs[resp_idx]
        resp_bpm = resp_freq_hz * 60.0
        total_energy = np.sum(resp_energies) + np.sum(vaso_energies) + 1e-12
        return {
            'respiratory_rate_bpm': resp_bpm,
            'resp_imf_index': resp_idx,
            'resp_energy': resp_energies[resp_idx],
            'vaso_imf_index': vaso_idx,
            'vaso_energy': vaso_energies[vaso_idx],
            'resp_energy_norm': resp_energies[resp_idx] / total_energy,
            'vaso_energy_norm': vaso_energies[vaso_idx] / total_energy
        }


# quick smoke test if run as script
if __name__ == "__main__":
    pp = PPGProcessor(fs=125.0)
    t = np.arange(0, 10, 1/125.0)
    sig = 0.2*np.sin(2*np.pi*0.25*t) + 0.6*np.sin(2*np.pi*2.0*t) + 0.05*np.random.randn(len(t)) + 0.2*np.polyval([0.0002, -0.01, 0.3], np.linspace(0,1,len(t)))
    pre = pp.preprocess(sig, normalize=True, remove_baseline=True)
    imfs, meta = pp.emd(pre, max_imfs=6, max_siftings=80, epsilon=0.01, debug=True)
    print("IMFs extracted:", len(imfs))
    for i,m in enumerate(meta):
        print(f"IMF {i+1}: sifts={m.get('sifts')}, final_sd={m.get('final_sd')}, stop={m.get('stop_reason')}, proto_len={len(m.get('proto',[]))}")
