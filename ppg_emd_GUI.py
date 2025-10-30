# ppg_emd_GUI.py
# GUI for PPG EMD Analyzer that uses backend debug protos/envelopes for plotting
# Dependencies: PyQt6, matplotlib, numpy, pandas
# Make sure ppg_emd_main.py from above is in same folder

import sys, os, math
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QComboBox, QVBoxLayout,
    QHBoxLayout, QTextEdit, QScrollArea, QTabWidget, QSpinBox, QSizePolicy,
    QLineEdit, QGroupBox, QGridLayout, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from ppg_emd_main import PPGProcessor

# Color definitions (raw vs preproc distinct)
RAW_COLOR = "#ffd966"     # warm yellow for raw PPG
PREP_COLOR = "#66e0ff"    # cyan/light-blue for preprocessed (distinct)
DETREND_COLOR = "#ffb84d" # poly detrend
FFT_COLOR = "#79c2ff"
METHOD_COLORS = {'EMD':"#ffd966",'EEMD':"#6ec06e",'CEEMD':"#6790ff",'CEEMDAN':"#ff8da1"}
RESIDUAL_COLOR = "#9aa0a6"
# HHT reversed colormap (red -> yellow)
HHT_CMAP = mcolors.LinearSegmentedColormap.from_list("RY", ["#ff3333", "#ff9933", "#ffcc66", "#ffff99"])


class WorkerThread(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    def __init__(self, proc: PPGProcessor, signal: np.ndarray, methods: list, max_imfs: int = 10,
                 max_siftings: int = 50, epsilon: Optional[float] = None,
                 eemd_trials=30, ce_trials=30, ceemdan_trials=30,
                 hht_sigma_time=2.0, hht_sigma_freq=1.0,
                 normalize_preproc: bool = True):
        super().__init__()
        self.proc = proc
        self.signal = signal
        self.methods = methods
        self.max_imfs = max_imfs
        self.max_siftings = max_siftings
        self.epsilon = epsilon
        self.eemd_trials = eemd_trials
        self.ce_trials = ce_trials
        self.ceemdan_trials = ceemdan_trials
        self.hht_sigma_time = hht_sigma_time
        self.hht_sigma_freq = hht_sigma_freq
        self.normalize_preproc = normalize_preproc

    def run(self):
        out = {}
        sig = self.signal
        self.progress.emit("Preprocessing signal...")
        pre = self.proc.preprocess(sig, normalize=self.normalize_preproc, remove_baseline=True, baseline_window_s=2.0)
        out['preproc'] = pre
        out['preproc_std'] = self.proc.preproc_std()
        imfs_all = {}
        meta_all = {}
        for method in self.methods:
            self.progress.emit(f"Computing {method} ...")
            if method == 'EMD':
                imfs, meta = self.proc.emd(pre, max_imfs=self.max_imfs, max_siftings=self.max_siftings, epsilon=self.epsilon, debug=True)
            elif method == 'EEMD':
                imfs, meta = self.proc.eemd(pre, trials=self.eemd_trials, noise_std_ratio=0.1,
                                            max_imfs=self.max_imfs, max_siftings=self.max_siftings, epsilon=self.epsilon, debug=True)
            elif method == 'CEEMD':
                imfs, meta = self.proc.ceemd(pre, trials=self.ce_trials, noise_std_ratio=0.1,
                                             max_imfs=self.max_imfs, max_siftings=self.max_siftings, epsilon=self.epsilon, debug=True)
            elif method == 'CEEMDAN':
                imfs, meta = self.proc.ceemdan(pre, trials=self.ceemdan_trials, noise_std_ratio=0.1,
                                               max_imfs=self.max_imfs, max_siftings=self.max_siftings, epsilon=self.epsilon, debug=True)
            else:
                imfs, meta = [], []
            N = len(pre)
            while len(imfs) < self.max_imfs:
                imfs.append(np.zeros(N))
                meta.append({'sifts':0,'final_sd':0.0,'stop_reason':'padded'})
            imfs_all[method] = imfs[:self.max_imfs]
            meta_all[method] = meta[:self.max_imfs]
            self.progress.emit(f"{method} done ({len(imfs)} modes)")
        out['imfs_all'] = imfs_all
        out['meta_all'] = meta_all

        # transforms & per-imf data
        self.progress.emit("Computing transforms and HHTs...")
        freqs_full, Xfull = self.proc.dft(pre)
        psd_full = self.proc.psd_from_spectrum(Xfull, len(pre))
        out['orig_freqs'] = freqs_full
        out['orig_Xk'] = Xfull
        out['orig_psd'] = psd_full

        imf_info = {}
        for method in self.methods:
            imf_info[method] = []
            for idx in range(self.max_imfs):
                imf = imfs_all[method][idx]
                freqs, Xk = self.proc.dft(imf)
                psd = self.proc.psd_from_spectrum(Xk, len(imf))
                t_h, freq_centers, amp_map = self.proc.hht_spectrogram(imf,
                                                                       f_min=0.0,
                                                                       f_max=0.5*self.proc.fs,
                                                                       freq_bins=96,
                                                                       smooth_sigma_time=self.hht_sigma_time,
                                                                       smooth_sigma_freq=self.hht_sigma_freq)
                # Acquire proto & env info from meta if present (debug)
                meta = meta_all.get(method, [])
                proto = None
                env_info = None
                if len(meta) > idx:
                    proto = meta[idx].get('proto', None)
                    env_info = meta[idx].get('env_info', None)
                # If proto/env not available, fallback to computing proto locally (pre - sum previous imfs)
                if proto is None:
                    pre_sig = pre
                    sum_prev = np.zeros_like(pre_sig)
                    for k in range(0, idx):
                        sum_prev += imfs_all[method][k]
                    proto = pre_sig - sum_prev
                    env_info = self.proc.get_envelopes(proto)
                imf_info[method].append({
                    'imf': imf,
                    'freqs': freqs,
                    'Xk': Xk,
                    'psd': psd,
                    'hht_t': t_h,
                    'hht_freqs': freq_centers,
                    'hht_map': amp_map,
                    'proto': proto,
                    'env_info': env_info
                })
        out['imf_info'] = imf_info

        preferred = None
        for cand in ['CEEMDAN','CEEMD','EEMD','EMD']:
            if cand in imfs_all:
                preferred = cand
                break
        try:
            metrics = self.proc.extract_resp_vaso(imfs_all[preferred])
        except Exception as e:
            metrics = {'error': str(e)}
        out['metrics'] = metrics
        out['params'] = {
            'epsilon': self.epsilon,
            'max_siftings': self.max_siftings,
            'max_imfs': self.max_imfs,
            'normalize_preproc': self.normalize_preproc
        }
        self.progress.emit("Processing complete.")
        self.finished.emit(out)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPG EMD Analyzer (Debug proto/envelopes)")
        self.resize(1400, 920)
        self.processor = PPGProcessor(fs=125.0, transform_method="FFT")
        self.data = None
        self.time = None
        self.signal = None
        self.time_col = None

        self.page_indices = {'IMF_COMP':0, 'IMF_FFT':0, 'IMF_PSD':0, 'IMF_HHT':0}
        self.page_size = 4

        self.results = None
        self.current_tab_key = None

        self._build_ui()
        self._apply_theme()

    def _apply_theme(self):
        self.setStyleSheet("""
            QWidget { background-color: #1f1b17; color: #f3e9b5; font-family: Arial, Helvetica, sans-serif; }
            QPushButton { background-color: #c79a1a; color: #111; border-radius:6px; padding:6px; }
            QPushButton:hover { background-color:#ffcf66; }
            QComboBox, QSpinBox, QLineEdit, QCheckBox { background:#2a2622; color:#f3e9b5; border:1px solid #8f6a08; padding:4px; }
            QTextEdit { background:#182018; color:#f3e9b5; border:1px solid #8f6a08; }
            QTabWidget::pane { border:1px solid #8f6a08; }
            QTabBar::tab { background:#2a2622; color:#f3e9b5; padding:8px; border:1px solid #8f6a08; }
            QTabBar::tab:selected { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #c79a1a, stop:1 #8f6a08); color:#111; }
        """)
        plt.rcParams.update({
            'figure.facecolor': '#2a2622',
            'axes.facecolor': '#2a2622',
            'axes.edgecolor': '#8f6a08',
            'axes.labelcolor': '#f3e9b5',
            'xtick.color': '#f3e9b5',
            'ytick.color': '#f3e9b5',
            'text.color': '#f3e9b5',
            'lines.linewidth': 0.9
        })

    def _build_ui(self):
        main = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self.load_btn = QPushButton("Load CSV"); self.load_btn.clicked.connect(self.load_csv)
        ctrl.addWidget(self.load_btn)
        self.col_combo = QComboBox(); ctrl.addWidget(QLabel(" Signal column:")); ctrl.addWidget(self.col_combo)
        self.transform_combo = QComboBox(); self.transform_combo.addItems(["FFT","DFT"]); ctrl.addWidget(QLabel(" Transform:")); ctrl.addWidget(self.transform_combo)
        self.method_combo = QComboBox(); self.method_combo.addItems(["EMD","EEMD","CEEMD","CEEMDAN"]); ctrl.addWidget(QLabel(" Method:")); ctrl.addWidget(self.method_combo)
        self.max_imf_spin = QSpinBox(); self.max_imf_spin.setRange(1, 40); self.max_imf_spin.setValue(10); ctrl.addWidget(QLabel(" Max IMFs:")); ctrl.addWidget(self.max_imf_spin)

        params_box = QGroupBox("Sifting controls (optional)")
        params_layout = QHBoxLayout(params_box)
        self.epsilon_input = QLineEdit(); self.epsilon_input.setPlaceholderText("epsilon (e.g., 0.2) or blank")
        params_layout.addWidget(QLabel("Epsilon:")); params_layout.addWidget(self.epsilon_input)
        self.max_sift_spin = QSpinBox(); self.max_sift_spin.setRange(1,1000); self.max_sift_spin.setValue(50)
        params_layout.addWidget(QLabel("Max sifts:")); params_layout.addWidget(self.max_sift_spin)
        params_layout.addWidget(QLabel("HHT σ_time:"))
        self.hht_sigma_time = QLineEdit("2.0"); self.hht_sigma_freq = QLineEdit("1.0")
        params_layout.addWidget(self.hht_sigma_time); params_layout.addWidget(QLabel("σ_freq:")); params_layout.addWidget(self.hht_sigma_freq)
        ctrl.addWidget(params_box)

        prep_box = QGroupBox("Preprocessing")
        prep_layout = QHBoxLayout(prep_box)
        self.normalize_checkbox = QCheckBox("Normalize preprocessing"); self.normalize_checkbox.setChecked(True)
        prep_layout.addWidget(self.normalize_checkbox)
        ctrl.addWidget(prep_box)

        self.imfs_per_page_spin = QSpinBox(); self.imfs_per_page_spin.setRange(1, 12); self.imfs_per_page_spin.setValue(4)
        self.imfs_per_page_spin.valueChanged.connect(self.on_imfs_per_page_changed)
        ctrl.addWidget(QLabel(" IMFs per page:")); ctrl.addWidget(self.imfs_per_page_spin)
        self.run_btn = QPushButton("Run Analysis"); self.run_btn.clicked.connect(self.run_analysis); ctrl.addWidget(self.run_btn)
        main.addLayout(ctrl)

        self.progress = QTextEdit(); self.progress.setReadOnly(True); self.progress.setFixedHeight(120)
        main.addWidget(self.progress)

        self.tabs = QTabWidget()
        tab_defs = [
            ("1. Original & FFT", "ORIG"),
            ("2. Preprocessing", "PRE"),
            ("3. IMF comparisons", "IMF_COMP"),
            ("4. IMF FFTs", "IMF_FFT"),
            ("5. IMF PSDs", "IMF_PSD"),
            ("6. IMF HHTs", "IMF_HHT"),
            ("7. Respiration & Vasomotor", "METRICS"),
            ("8. Noise Trials", "NOISE")
        ]
        self.tab_containers = {}
        self.tab_key_by_index = {}
        for idx, (label, key) in enumerate(tab_defs):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            scroll = QScrollArea(); scroll.setWidgetResizable(True)
            container = QWidget(); container.setLayout(QVBoxLayout()); scroll.setWidget(container)
            layout.addWidget(scroll)
            self.tab_containers[key] = container
            self.tab_key_by_index[idx] = key
            self.tabs.addTab(widget, label)
        main.addWidget(self.tabs)

        pager_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Prev"); self.prev_btn.clicked.connect(lambda: self.change_page(-1))
        self.next_btn = QPushButton("Next"); self.next_btn.clicked.connect(lambda: self.change_page(1))
        self.page_label = QLabel("Page 1/1")
        pager_layout.addWidget(self.prev_btn); pager_layout.addWidget(self.page_label); pager_layout.addWidget(self.next_btn)
        main.addLayout(pager_layout)

        self.tabs.currentChanged.connect(self.on_tab_changed)

    def append_progress(self, s: str):
        self.progress.append(s)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", os.getcwd(), "CSV Files (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            self.append_progress(f"Failed reading CSV: {e}")
            return
        self.data = df
        cols = list(df.columns)
        self.col_combo.clear(); self.col_combo.addItems(cols)
        if 'PLETH' in cols:
            self.col_combo.setCurrentIndex(cols.index('PLETH'))
        time_cols = [c for c in cols if 'time' in c.lower() or 't[' in c.lower() or 'sec' in c.lower()]
        self.time_col = time_cols[0] if len(time_cols)>0 else cols[0]
        self.append_progress(f"Loaded {os.path.basename(path)} (time col: {self.time_col})")

    def run_analysis(self):
        if self.data is None:
            self.append_progress("Please load CSV first.")
            return
        col = self.col_combo.currentText()
        if col == "":
            self.append_progress("Select signal column.")
            return
        t = self.data[self.time_col].values
        sig = self.data[col].values.astype(float)
        self.time = t; self.signal = sig

        eps_text = self.epsilon_input.text().strip()
        epsilon = None
        if eps_text != "":
            try:
                epsilon = float(eps_text)
            except:
                self.append_progress("Invalid epsilon value. Leave blank or enter numeric.")
                return
        max_sifts = int(self.max_sift_spin.value())
        try:
            hht_sigma_time = float(self.hht_sigma_time.text()); hht_sigma_freq = float(self.hht_sigma_freq.text())
        except:
            hht_sigma_time = 2.0; hht_sigma_freq = 1.0

        self.processor.transform_method = self.transform_combo.currentText()
        methods = ["EMD","EEMD","CEEMD","CEEMDAN"]
        max_imfs = int(self.max_imf_spin.value())
        normalize_preproc = bool(self.normalize_checkbox.isChecked())
        self.append_progress(f"Starting analysis (transform={self.processor.transform_method}, max_imfs={max_imfs}, epsilon={epsilon}, max_sifts={max_sifts}, normalize={normalize_preproc}) ...")
        self.worker = WorkerThread(self.processor, sig, methods, max_imfs=max_imfs,
                                   max_siftings=max_sifts, epsilon=epsilon,
                                   hht_sigma_time=hht_sigma_time, hht_sigma_freq=hht_sigma_freq,
                                   normalize_preproc=normalize_preproc)
        self.worker.progress.connect(self.append_progress)
        self.worker.finished.connect(self.on_finished)
        self.run_btn.setEnabled(False)
        self.worker.start()

    def on_finished(self, out: dict):
        self.run_btn.setEnabled(True)
        self.append_progress("Analysis finished. Preparing visualizations...")
        self.results = out
        self.results['max_imfs'] = int(self.max_imf_spin.value())
        for key in ['IMF_COMP','IMF_FFT','IMF_PSD','IMF_HHT']:
            if key not in self.page_indices:
                self.page_indices[key] = 0
            else:
                self.page_indices[key] = min(self.page_indices[key], math.ceil(self.results['max_imfs'] / max(1, self.page_size)) - 1)
        self.render_tab_orig()
        self.render_tab_preproc()
        self.render_tab_metrics()
        self.prepare_noise_tab_controls()
        current_index = self.tabs.currentIndex()
        current_key = self.tab_key_by_index.get(current_index)
        if current_key in ('IMF_COMP','IMF_FFT','IMF_PSD','IMF_HHT'):
            self.render_active_imf_tab()
        else:
            self.update_page_label_for_active_tab()
        self.append_progress("Visualizations ready.")

    # ----------------------
    # Plot helpers
    # ----------------------
    def clear_tab(self, key):
        container = self.tab_containers[key]
        layout = container.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

    def _add_canvas_with_toolbar(self, container, fig, height=360):
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        canvas.setMinimumHeight(height)
        toolbar = NavigationToolbar(canvas, self)
        vbox = container.layout()
        vbox.addWidget(toolbar)
        vbox.addWidget(canvas)
        canvas.draw()
        return canvas, toolbar

    def render_tab_orig(self):
        if self.results is None: return
        self.clear_tab('ORIG')
        t = self.time; orig = self.signal; pre = self.results['preproc']
        freqs = self.results['orig_freqs']; Xk = self.results['orig_Xk']
        fig, axs = plt.subplots(2,1, figsize=(10,5), constrained_layout=True)
        axs[0].plot(t, orig, color=RAW_COLOR, label='Raw PLETH')
        axs[0].plot(t, pre, color=PREP_COLOR, label='Preprocessed')
        axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Amplitude"); axs[0].legend()
        half = len(freqs)//2
        axs[1].plot(freqs[:half], np.abs(Xk)[:half], color=FFT_COLOR, label='Transform Magnitude')
        axs[1].set_xlim(0,5); axs[1].set_xlabel("Frequency (Hz)"); axs[1].set_ylabel("Magnitude"); axs[1].legend()
        self._add_canvas_with_toolbar(self.tab_containers['ORIG'], fig, height=420)

    def render_tab_preproc(self):
        if self.results is None: return
        self.clear_tab('PRE')
        t = self.time; orig = self.signal; pre = self.results['preproc']
        baseline = self.processor.running_mean(orig - np.mean(orig), int(round(2.0 * self.processor.fs)))
        detr = self.processor.detrend_poly(orig, deg=3)
        fig, axs = plt.subplots(2,1, figsize=(10,5), constrained_layout=True)
        axs[0].plot(t, orig, color="#79c2ff", label="Raw")
        axs[0].plot(t, baseline + np.mean(orig), color="#ffb84d", label="Estimated baseline")
        axs[0].legend(); axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Amplitude")
        axs[1].plot(t, detr, color=DETREND_COLOR, label="Poly-detrended")
        axs[1].plot(t, pre, color=PREP_COLOR, label="Final preproc")
        axs[1].legend(); axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Amplitude")
        self._add_canvas_with_toolbar(self.tab_containers['PRE'], fig, height=420)
        stdev = self.results.get('preproc_std', float('nan'))
        params = self.results.get('params', {})
        normalize_flag = params.get('normalize_preproc', True)
        sifts_info = self.max_sift_spin.value()
        info_label = QLabel(f"Preprocessed STDEV: {stdev:.4f}    |    Normalize: {normalize_flag}    |    Max sifts (UI): {sifts_info}    |    Epsilon: {params.get('epsilon')}    |    Max IMFs: {params.get('max_imfs')}")
        self.tab_containers['PRE'].layout().addWidget(info_label)

    def render_tab_metrics(self):
        if self.results is None: return
        self.clear_tab('METRICS')
        metrics = self.results.get('metrics', {})
        text = f"<b>Respiration rate (bpm):</b> {metrics.get('respiratory_rate_bpm','N/A')}<br>"
        text += f"<b>Resp IMF index:</b> {metrics.get('resp_imf_index','N/A')}<br>"
        text += f"<b>Resp energy (norm):</b> {metrics.get('resp_energy_norm','N/A')}<br>"
        text += f"<b>Vasomotor IMF index:</b> {metrics.get('vaso_imf_index','N/A')}<br>"
        text += f"<b>Vasomotor energy (norm):</b> {metrics.get('vaso_energy_norm','N/A')}<br>"
        lbl = QLabel(text); lbl.setWordWrap(True)
        self.tab_containers['METRICS'].layout().addWidget(lbl)

    # Pagination helpers
    def change_page(self, delta: int):
        current_idx = self.tabs.currentIndex()
        key = self.tab_key_by_index.get(current_idx)
        if key not in ('IMF_COMP','IMF_FFT','IMF_PSD','IMF_HHT'):
            return
        max_imfs = self.results['max_imfs'] if self.results is not None else int(self.max_imf_spin.value())
        pages = math.ceil(max_imfs / max(1, self.page_size))
        cur = self.page_indices.get(key, 0)
        new = max(0, min(pages - 1, cur + delta))
        if new != cur:
            self.page_indices[key] = new
            self.render_active_imf_tab()

    def update_page_label_for_active_tab(self):
        if self.results is None:
            self.page_label.setText("Page 0/0")
            return
        max_imfs = self.results['max_imfs']
        pages = math.ceil(max_imfs / max(1, self.page_size))
        current_idx = self.tabs.currentIndex()
        key = self.tab_key_by_index.get(current_idx)
        if key in ('IMF_COMP','IMF_FFT','IMF_PSD','IMF_HHT'):
            cur = self.page_indices.get(key, 0)
            self.page_label.setText(f"Page {cur+1}/{max(1,pages)}")
        else:
            self.page_label.setText("Page -")

    def on_imfs_per_page_changed(self, val):
        self.page_size = int(val)
        if self.results:
            max_imfs = self.results['max_imfs']
        else:
            max_imfs = int(self.max_imf_spin.value())
        pages = math.ceil(max_imfs / max(1, self.page_size))
        for k in self.page_indices:
            self.page_indices[k] = min(self.page_indices[k], max(0, pages - 1))
        self.render_active_imf_tab()

    def on_tab_changed(self, idx):
        key = self.tab_key_by_index.get(idx)
        if key in ('IMF_COMP','IMF_FFT','IMF_PSD','IMF_HHT'):
            self.current_tab_key = key
            self.render_active_imf_tab()
        else:
            self.current_tab_key = None
            if key == 'ORIG':
                self.render_tab_orig()
            elif key == 'PRE':
                self.render_tab_preproc()
            elif key == 'METRICS':
                self.render_tab_metrics()
            elif key == 'NOISE':
                self.prepare_noise_tab_controls()
        self.update_page_label_for_active_tab()

    # IMF rendering uses backend proto/envelope if available
    def render_active_imf_tab(self):
        if self.results is None:
            self.update_page_label_for_active_tab()
            return
        key = self.current_tab_key
        if key is None:
            self.update_page_label_for_active_tab()
            return
        page = self.page_indices.get(key, 0)
        max_imfs = self.results['max_imfs']
        pages = math.ceil(max_imfs / max(1, self.page_size))
        page = min(max(0, page), max(0, pages - 1))
        self.page_indices[key] = page
        start = page * self.page_size
        end = min(start + self.page_size, max_imfs)

        preferred = None
        for cand in ['CEEMDAN','CEEMD','EEMD','EMD']:
            if cand in self.results['imf_info']:
                preferred = cand
                break

        if key == 'IMF_COMP':
            container_key = 'IMF_COMP'
            self.clear_tab(container_key)

            try:
                imfs_pref = self.results['imf_info'][preferred]
                figc, axc = plt.subplots(1,1, figsize=(14,3), constrained_layout=True)
                cmap = plt.get_cmap('tab10')
                for i, entry in enumerate(imfs_pref):
                    imf = entry['imf']
                    axc.plot(self.time, imf, color=cmap(i % 10), label=f"IMF{i+1}")
                axc.set_xlabel("Time (s)"); axc.set_ylabel("Amplitude")
                axc.set_title(f"All IMFs overlay ({preferred})")
                axc.legend(ncol=4, fontsize='small')
                self._add_canvas_with_toolbar(self.tab_containers[container_key], figc, height=280)
            except Exception as e:
                self.append_progress(f"Could not render combined IMF overlay: {e}")

            for idx in range(start, end):
                fig, axs = plt.subplots(1,4, figsize=(16,2.8), constrained_layout=True)
                fig.suptitle(f"IMF #{idx+1}")
                t = self.time
                pre = self.results['preproc']
                for i, method in enumerate(['EMD','EEMD','CEEMD','CEEMDAN']):
                    entry = self.results['imf_info'][method][idx]
                    imf = entry['imf']
                    proto = entry.get('proto', None)
                    env_info = entry.get('env_info', None)
                    axs[i].plot(t, imf, color=METHOD_COLORS[method], label=f"{method} IMF (result)")
                    if proto is None:
                        # fallback
                        sum_prev = np.zeros_like(pre)
                        for k in range(0, idx):
                            sum_prev += self.results['imf_info'][method][k]['imf']
                        proto = pre - sum_prev
                        env_info = self.processor.get_envelopes(proto)
                    # plot proto envelopes and extrema from backend env_info
                    axs[i].plot(t, env_info['env_max'], linestyle='--', color="#ffb84d", linewidth=0.9, label='Upper envelope (proto)')
                    axs[i].plot(t, env_info['env_min'], linestyle='--', color="#79c2ff", linewidth=0.9, label='Lower envelope (proto)')
                    axs[i].plot(t, env_info['mean_env'], linestyle=':', color='#ffffff', linewidth=0.9, label='Mean envelope (proto)')
                    if env_info.get('max_idx') is not None and env_info['max_idx'].size > 0:
                        axs[i].scatter(t[env_info['max_idx']], proto[env_info['max_idx']], marker='^', color='#ff4444', s=10, label='Maxima (proto)' if i==0 else "")
                    if env_info.get('min_idx') is not None and env_info['min_idx'].size > 0:
                        axs[i].scatter(t[env_info['min_idx']], proto[env_info['min_idx']], marker='v', color='#33ff77', s=10, label='Minima (proto)' if i==0 else "")
                    imfs_upto = [self.results['imf_info'][method][k]['imf'] for k in range(0, idx+1)]
                    residual_after = pre - np.sum(imfs_upto, axis=0)
                    axs[i].plot(t, residual_after, linestyle='--', color=RESIDUAL_COLOR, linewidth=0.8, alpha=0.8, label='Residual after extraction')
                    axs[i].set_xlabel("Time (s)"); axs[i].set_ylabel("Amplitude")
                    axs[i].legend(fontsize='small', loc='upper right')
                self._add_canvas_with_toolbar(self.tab_containers[container_key], fig, height=320)
                meta_text_lines = []
                for method in ['EMD','EEMD','CEEMD','CEEMDAN']:
                    meta = self.results.get('meta_all', {}).get(method, [])
                    if len(meta) > idx:
                        m = meta[idx]
                        if 'sifts' in m:
                            meta_text_lines.append(f"{method}: sifts={m.get('sifts')}, sd={m.get('final_sd'):.4g}, stop={m.get('stop_reason')}")
                        else:
                            meta_text_lines.append(f"{method}: sifts_mean={m.get('sifts_mean','N/A'):.2f}, sd_mean={m.get('final_sd_mean','N/A'):.4g}")
                if meta_text_lines:
                    meta_label = QLabel("   |   ".join(meta_text_lines) + f"   | UI max sifts: {self.max_sift_spin.value()}")
                    self.tab_containers[container_key].layout().addWidget(meta_label)

        elif key == 'IMF_FFT':
            container_key = 'IMF_FFT'
            self.clear_tab(container_key)
            for idx in range(start, end):
                fig, axs = plt.subplots(1,4, figsize=(16,2.6), constrained_layout=True)
                fig.suptitle(f"IMF #{idx+1} - Transform Magnitude")
                for i, method in enumerate(['EMD','EEMD','CEEMD','CEEMDAN']):
                    freqs = self.results['imf_info'][method][idx]['freqs']
                    Xk = self.results['imf_info'][method][idx]['Xk']
                    half = len(freqs)//2
                    axs[i].plot(freqs[:half], np.abs(Xk)[:half], color=METHOD_COLORS[method])
                    axs[i].set_xlim(0,5); axs[i].set_xlabel("Frequency (Hz)"); axs[i].set_ylabel("Magnitude")
                    axs[i].legend([method], fontsize='small')
                self._add_canvas_with_toolbar(self.tab_containers[container_key], fig, height=260)

        elif key == 'IMF_PSD':
            container_key = 'IMF_PSD'
            self.clear_tab(container_key)
            for idx in range(start, end):
                fig, axs = plt.subplots(1,4, figsize=(16,2.6), constrained_layout=True)
                fig.suptitle(f"IMF #{idx+1} - PSD")
                for i, method in enumerate(['EMD','EEMD','CEEMD','CEEMDAN']):
                    freqs = self.results['imf_info'][method][idx]['freqs']
                    psd = self.results['imf_info'][method][idx]['psd']
                    half = len(freqs)//2
                    axs[i].plot(freqs[:half], psd[:half], color=METHOD_COLORS[method])
                    axs[i].set_xlim(0,5); axs[i].set_xlabel("Frequency (Hz)"); axs[i].set_ylabel("Power")
                    axs[i].legend([method], fontsize='small')
                self._add_canvas_with_toolbar(self.tab_containers[container_key], fig, height=260)

        elif key == 'IMF_HHT':
            container_key = 'IMF_HHT'
            self.clear_tab(container_key)
            for idx in range(start, end):
                fig, axs = plt.subplots(2,2, figsize=(14,5), constrained_layout=True)
                fig.suptitle(f"IMF #{idx+1} - HHT (smoothed, red=high amp)")
                for i, method in enumerate(['EMD','EEMD','CEEMD','CEEMDAN']):
                    t_h = self.results['imf_info'][method][idx]['hht_t']
                    f_cent = self.results['imf_info'][method][idx]['hht_freqs']
                    amp_map = self.results['imf_info'][method][idx]['hht_map']
                    ax = axs.flatten()[i]
                    im = ax.pcolormesh(t_h, f_cent, amp_map, shading='auto', cmap=HHT_CMAP)
                    ax.set_ylim(0, 2.0)
                    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)"); ax.set_title(method)
                    fig.colorbar(im, ax=ax, format='%.3f')
                self._add_canvas_with_toolbar(self.tab_containers[container_key], fig, height=380)

        self.update_page_label_for_active_tab()

    # ----------------------
    # Noise tab: reuse earlier implementations (kept minimal for brevity)
    # ----------------------
    def prepare_noise_tab_controls(self):
        container = self.tab_containers['NOISE']
        layout = container.layout()
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w: w.setParent(None)
        if self.results is None:
            layout.addWidget(QLabel("Run analysis to see noise trials."))
            return
        meta_all = self.results.get('meta_all', {})
        pre = self.results['preproc']
        t = self.time
        row = QHBoxLayout()
        self.noise_method_combo = QComboBox()
        avail = [m for m in ['EEMD','CEEMD','CEEMDAN'] if m in meta_all]
        if len(avail) == 0:
            layout.addWidget(QLabel("No ensemble methods were run or no noise metadata available."))
            return
        self.noise_method_combo.addItems(avail)
        self.noise_method_combo.currentTextChanged.connect(self.on_noise_method_changed)
        row.addWidget(QLabel("Method:")); row.addWidget(self.noise_method_combo)
        self.noise_trial_spin = QSpinBox(); self.noise_trial_spin.setRange(0, 0); self.noise_trial_spin.setValue(0)
        row.addWidget(QLabel("Trial index:")); row.addWidget(self.noise_trial_spin)
        self.noise_imf_spin = QSpinBox(); self.noise_imf_spin.setRange(1, max(1, self.results['max_imfs'])); self.noise_imf_spin.setValue(1)
        row.addWidget(QLabel("IMF index (CEEMDAN):")); row.addWidget(self.noise_imf_spin)
        self.btn_plot_signal_plus_noise = QPushButton("Plot signal + noise"); self.btn_plot_signal_plus_noise.clicked.connect(self.plot_noise_trial)
        row.addWidget(self.btn_plot_signal_plus_noise)
        self.btn_plot_pair = QPushButton("Plot complementary pair"); self.btn_plot_pair.clicked.connect(self.plot_noise_pair)
        row.addWidget(self.btn_plot_pair)
        self.btn_plot_adaptive = QPushButton("Plot CEEMDAN adaptive noise (IMF)"); self.btn_plot_adaptive.clicked.connect(self.plot_adaptive_noise_for_imf)
        row.addWidget(self.btn_plot_adaptive)
        layout.addLayout(row)
        self.on_noise_method_changed(self.noise_method_combo.currentText())
        help_lbl = QLabel("Use the controls to inspect injected noises and complementary pairs (EEMD/CEEMD) or adaptive noises (CEEMDAN).")
        help_lbl.setWordWrap(True)
        layout.addWidget(help_lbl)

    def on_noise_method_changed(self, method: str):
        meta_all = self.results.get('meta_all', {}) if self.results else {}
        if method == 'EEMD':
            try:
                noise_trials = meta_all[method][0].get('noise_trials', [])
                ntrials = len(noise_trials)
                if ntrials == 0:
                    self.noise_trial_spin.setRange(0, 0); self.noise_trial_spin.setValue(0)
                    self.append_progress("No noise_trials found in EEMD metadata.")
                else:
                    self.noise_trial_spin.setRange(0, max(0, ntrials-1)); self.noise_trial_spin.setValue(0)
            except Exception:
                self.noise_trial_spin.setRange(0, 0); self.noise_trial_spin.setValue(0)
        elif method == 'CEEMD':
            try:
                pairs = meta_all[method][0].get('noise_pairs', [])
                ntrials = len(pairs)
                if ntrials == 0:
                    self.noise_trial_spin.setRange(0, 0); self.noise_trial_spin.setValue(0)
                else:
                    self.noise_trial_spin.setRange(0, max(0, ntrials-1)); self.noise_trial_spin.setValue(0)
            except Exception:
                self.noise_trial_spin.setRange(0, 0); self.noise_trial_spin.setValue(0)
        elif method == 'CEEMDAN':
            try:
                meta_list = meta_all.get('CEEMDAN', [])
                if len(meta_list) == 0:
                    self.noise_trial_spin.setRange(0, 0); self.noise_trial_spin.setValue(0)
                    self.append_progress("No CEEMDAN metadata found.")
                    return
                first_adaptive = meta_list[0].get('adaptive_noises', [])
                ntrials = len(first_adaptive)
                if ntrials == 0:
                    self.noise_trial_spin.setRange(0,0); self.noise_trial_spin.setValue(0)
                    self.append_progress("No adaptive noises found in CEEMDAN metadata.")
                else:
                    self.noise_trial_spin.setRange(0, max(0, ntrials-1)); self.noise_trial_spin.setValue(0)
                self.noise_imf_spin.setRange(1, max(1, len(meta_list)))
                self.noise_imf_spin.setValue(1)
            except Exception:
                self.noise_trial_spin.setRange(0,0); self.noise_trial_spin.setValue(0)

    def plot_noise_trial(self):
        method = self.noise_method_combo.currentText()
        trial = int(self.noise_trial_spin.value())
        container = self.tab_containers['NOISE']
        layout = container.layout()
        self.prepare_noise_tab_controls()
        pre = self.results['preproc']
        t = self.time
        meta_all = self.results.get('meta_all', {})
        if method == 'EEMD':
            try:
                noise_trials = meta_all['EEMD'][0].get('noise_trials', [])
                if not noise_trials:
                    self.append_progress("No noise trials available to plot for EEMD.")
                    return
                n = noise_trials[trial]
                sig_plus = pre + n
                fig, ax = plt.subplots(2,1, figsize=(12,6), constrained_layout=True)
                ax[0].plot(t, pre, color=PREP_COLOR, label='Preprocessed signal')
                ax[0].plot(t, sig_plus, color='#ffd1a9', alpha=0.9, label=f'signal + noise (trial {trial})')
                ax[0].set_xlabel("Time (s)"); ax[0].set_ylabel("Amplitude"); ax[0].legend()
                ax[1].plot(t, n, color='#ff6666', label='noise (trial)')
                ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("Noise amplitude"); ax[1].legend()
                self._add_canvas_with_toolbar(layout, fig, height=420)
            except Exception as e:
                self.append_progress(f"Error plotting EEMD trial: {e}")
        elif method == 'CEEMD':
            try:
                pairs = meta_all['CEEMD'][0].get('noise_pairs', [])
                if not pairs:
                    self.append_progress("No complementary noise pairs available to plot for CEEMD.")
                    return
                noise_plus, noise_minus = pairs[trial]
                sig_plus = pre + noise_plus
                sig_minus = pre + noise_minus
                fig, ax = plt.subplots(3,1, figsize=(12,8), constrained_layout=True)
                ax[0].plot(t, pre, color=PREP_COLOR, label='Preprocessed')
                ax[0].plot(t, sig_plus, color='#ffd1a9', label=f'signal + noise (trial {trial})')
                ax[0].plot(t, sig_minus, color='#c0e8ff', label=f'signal - noise (trial {trial})')
                ax[0].legend(); ax[0].set_ylabel("Amplitude"); ax[0].set_xlabel("Time (s)")
                ax[1].plot(t, noise_plus, color='#ff6666', label='noise (+)')
                ax[1].set_xlabel("Time (s)"); ax[1].set_ylabel("noise +")
                ax[2].plot(t, noise_minus, color='#6699ff', label='noise (-)')
                ax[2].set_xlabel("Time (s)"); ax[2].set_ylabel("noise -")
                self._add_canvas_with_toolbar(layout, fig, height=480)
            except Exception as e:
                self.append_progress(f"Error plotting CEEMD pair: {e}")
        elif method == 'CEEMDAN':
            self.plot_adaptive_noise_for_imf()
        else:
            self.append_progress("Unknown method for noise plot.")

    def plot_noise_pair(self):
        method = self.noise_method_combo.currentText()
        trial = int(self.noise_trial_spin.value())
        container = self.tab_containers['NOISE']
        layout = container.layout()
        self.prepare_noise_tab_controls()
        if method != 'CEEMD':
            self.append_progress("Complementary pairs are only available for CEEMD.")
            return
        meta_all = self.results.get('meta_all', {})
        pre = self.results['preproc']
        t = self.time
        try:
            pairs = meta_all['CEEMD'][0].get('noise_pairs', [])
            if not pairs:
                self.append_progress("No noise_pairs found in CEEMD metadata.")
                return
            plus, minus = pairs[trial]
            fig, axs = plt.subplots(2,1, figsize=(12,6), constrained_layout=True)
            axs[0].plot(t, pre, color=PREP_COLOR, label='Preprocessed')
            axs[0].plot(t, pre + plus, color='#ffd1a9', label=f'signal + noise (trial {trial})')
            axs[0].plot(t, pre + minus, color='#c0e8ff', label=f'signal - noise (trial {trial})')
            axs[0].legend(); axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Amplitude")
            axs[1].plot(t, plus, color='#ff6666', label='noise (+)'); axs[1].plot(t, minus, color='#6699ff', label='noise (-)')
            axs[1].legend(); axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Noise amplitude")
            self._add_canvas_with_toolbar(layout, fig, height=420)
        except Exception as e:
            self.append_progress(f"Error plotting CEEMD pair: {e}")

    def plot_adaptive_noise_for_imf(self):
        method = self.noise_method_combo.currentText()
        if method != 'CEEMDAN':
            self.append_progress("Adaptive noises visualization only applies to CEEMDAN.")
            return
        imf_idx = int(self.noise_imf_spin.value()) - 1
        trial_idx = int(self.noise_trial_spin.value())
        container = self.tab_containers['NOISE']
        self.prepare_noise_tab_controls()
        layout = container.layout()
        meta_all = self.results.get('meta_all', {})
        try:
            meta_list = meta_all['CEEMDAN']
            if len(meta_list) == 0:
                self.append_progress("No CEEMDAN meta available.")
                return
            if imf_idx < 0 or imf_idx >= len(meta_list):
                self.append_progress("IMF index out of range for CEEMDAN.")
                return
            adaptive = meta_list[imf_idx].get('adaptive_noises', [])
            if not adaptive:
                self.append_progress("No adaptive_noise arrays stored for this IMF.")
                return
            try:
                n = adaptive[trial_idx]
            except Exception:
                fig, ax = plt.subplots(1,1, figsize=(12,4), constrained_layout=True)
                for j, arr in enumerate(adaptive):
                    ax.plot(self.time, arr, alpha=0.6, label=f"trial {j}")
                ax.set_xlabel("Time (s)"); ax.set_ylabel("Noise amplitude"); ax.set_title(f"CEEMDAN adaptive noises (IMF {imf_idx+1})")
                ax.legend(fontsize='small')
                self._add_canvas_with_toolbar(layout, fig, height=360)
                return
            fig, axs = plt.subplots(2,1, figsize=(12,6), constrained_layout=True)
            axs[0].plot(self.time, self.results['preproc'], color=PREP_COLOR, label='Preprocessed')
            axs[0].plot(self.time, self.results['preproc'] + n, color='#ffd1a9', label=f'preproc + adaptive noise (trial {trial_idx})')
            axs[0].legend(); axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Amplitude")
            axs[1].plot(self.time, n, color='#ff6666', label='adaptive noise'); axs[1].set_xlabel("Time (s)"); axs[1].set_ylabel("Noise amplitude"); axs[1].legend()
            self._add_canvas_with_toolbar(layout, fig, height=420)
        except Exception as e:
            self.append_progress(f"Error plotting CEEMDAN adaptive noises: {e}")

    # ----------------------
    # Misc
    # ----------------------
    def update_page_label_for_active_tab(self):
        if self.results is None:
            self.page_label.setText("Page 0/0")
            return
        max_imfs = self.results['max_imfs']
        pages = math.ceil(max_imfs / max(1, self.page_size))
        current_idx = self.tabs.currentIndex()
        key = self.tab_key_by_index.get(current_idx)
        if key in ('IMF_COMP','IMF_FFT','IMF_PSD','IMF_HHT'):
            cur = self.page_indices.get(key, 0)
            self.page_label.setText(f"Page {cur+1}/{max(1,pages)}")
        else:
            self.page_label.setText("Page -")

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
