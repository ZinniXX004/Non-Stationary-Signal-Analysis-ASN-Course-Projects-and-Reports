# main_GUI.py
# Dependencies: PyQt6, matplotlib, numpy, pandas

import sys, os, math, bisect
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
from PPG_main_fixed import PPGProcessor

# USER CUSTOM FUNCTIONS (SPLINE & EXTREMA)
def tambah_titik_ujung(indices, values):
    if len(indices) < 2:
        return indices, values
    h_awal = indices[1] - indices[0]
    idx_awal = indices[0] - h_awal
    val_awal = values[0] - (values[1] - values[0])
    h_akhir = indices[-1] - indices[-2]
    idx_akhir = indices[-1] + h_akhir
    val_akhir = values[-1] + (values[-1] - values[-2])
    new_indices = np.concatenate(([idx_awal], indices, [idx_akhir]))
    new_values = np.concatenate(([val_awal], values, [val_akhir]))
    return new_indices, new_values

def hitung_koefisien_spline(x, y):
    nx = len(x)
    h = np.diff(x)
    a = np.array([iy for iy in y])
    A = np.zeros((nx, nx))
    A[0, 0] = 1.0
    for i in range(nx - 1):
        if i != (nx - 2):
            A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
        A[i + 1, i] = h[i]
        A[i, i + 1] = h[i]
    A[0, 1] = 0.0
    A[nx - 1, nx - 2] = 0.0
    A[nx - 1, nx - 1] = 1.0
    B = np.zeros(nx)
    for i in range(nx - 2):
        B[i + 1] = 3.0 * (a[i + 2] - a[i + 1]) / h[i + 1] - 3.0 * (a[i + 1] - a[i]) / h[i]
    try:
        c = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        c = np.zeros(nx) 

    d, b = [], []
    for i in range(nx - 1):
        d.append((c[i + 1] - c[i]) / (3.0 * h[i]))
        tb = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        b.append(tb)
    return a, b, c, d

def spline_envelope(x_points, y_points, x_full_range):
    if len(x_points) < 2:
        return np.interp(x_full_range, x_points, y_points) if len(x_points) > 0 else np.zeros_like(x_full_range)

    a, b, c, d = hitung_koefisien_spline(x_points, y_points)
    y_envelope = []
    for t in x_full_range:
        if t < x_points[0] or t > x_points[-1]:
            if t < x_points[0]: y_envelope.append(y_points[0])
            else: y_envelope.append(y_points[-1])
            continue
        
        i = bisect.bisect_left(x_points, t) - 1
        if i < 0: i = 0
        if i >= len(b): i = len(b) - 1
        
        dx = t - x_points[i]
        result = a[i] + b[i] * dx + c[i] * dx ** 2.0 + d[i] * dx ** 3.0
        y_envelope.append(result)
    return np.array(y_envelope)

# Color definitions
RAW_COLOR = "#ffd966"     # warm yellow
PREP_COLOR = "#66e0ff"    # cyan
DETREND_COLOR = "#ffb84d" # orange
FFT_COLOR = "#79c2ff"
METHOD_COLORS = {'EMD':"#ffd966",'EEMD':"#6ec06e",'CEEMD':"#6790ff",'CEEMDAN':"#ff8da1"}
RESIDUAL_COLOR = "#9aa0a6"
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
            
            # Pad if necessary
            N = len(pre)
            while len(imfs) < self.max_imfs:
                imfs.append(np.zeros(N))
                meta.append({'sifts':0,'final_sd':0.0,'stop_reason':'padded'})
            
            imfs_all[method] = imfs[:self.max_imfs]
            meta_all[method] = meta[:self.max_imfs]
            self.progress.emit(f"{method} done ({len(imfs)} modes)")
        
        out['imfs_all'] = imfs_all
        out['meta_all'] = meta_all

        # transforms and per-imf data
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
                
               
                imf_info[method].append({
                    'imf': imf,
                    'freqs': freqs,
                    'Xk': Xk,
                    'psd': psd,
                    'hht_t': t_h,
                    'hht_freqs': freq_centers,
                    'hht_map': amp_map
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
        self.setWindowTitle("PPG EMD Analyzer")
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
        
        self.epsilon_input = QLineEdit()
        self.epsilon_input.setPlaceholderText("epsilon (e.g., 0.2) or blank")
        self.epsilon_input.setText("0.2") 
        
        params_layout.addWidget(QLabel("Epsilon:")); params_layout.addWidget(self.epsilon_input)
        
        self.max_sift_spin = QSpinBox(); self.max_sift_spin.setRange(1,1000)
        self.max_sift_spin.setValue(20) 

        params_layout.addWidget(QLabel("Max sifts:")); params_layout.addWidget(self.max_sift_spin)
        params_layout.addWidget(QLabel("HHT σ_time:"))
        self.hht_sigma_time = QLineEdit("2.0"); self.hht_sigma_freq = QLineEdit("1.0")
        params_layout.addWidget(self.hht_sigma_time); params_layout.addWidget(QLabel("σ_freq:")); params_layout.addWidget(self.hht_sigma_freq)
        ctrl.addWidget(params_box)

        prep_box = QGroupBox("Preprocessing")
        prep_layout = QHBoxLayout(prep_box)
        
        self.normalize_checkbox = QCheckBox("Normalize preprocessing")
        self.normalize_checkbox.setChecked(False)
        
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

        selected_method = self.method_combo.currentText()
        methods = [selected_method] 

        max_imfs = int(self.max_imf_spin.value())
        normalize_preproc = bool(self.normalize_checkbox.isChecked())
        self.append_progress(f"Starting analysis (method={selected_method}, transform={self.processor.transform_method}, max_imfs={max_imfs}, epsilon={epsilon}, max_sifts={max_sifts}, normalize={normalize_preproc}) ...")
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
        # update page indices safe
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

    # Plot helpers
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

    # Interactive Legend Helper
    def make_legend_interactive(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return

        leg = ax.legend(handles, labels, fontsize='x-small', loc='best')
        
        artist_map = {}
        
        for leg_text, handle in zip(leg.get_texts(), handles):
            leg_text.set_picker(True)  # Aktifkan bisa diklik
            artist_map[leg_text] = handle

        def on_pick(event):
            if event.artist in artist_map:
                text = event.artist
                handle = artist_map[text]
                
                is_visible = not handle.get_visible()
                handle.set_visible(is_visible)
                
                # Update teks legenda: [x] -> [ ] atau sebaliknya
                current_label = text.get_text()
                if is_visible:
                    # Jika jadi visible, ubah [ ] jadi [x] dan tebalkan
                    if "[ ]" in current_label:
                        new_label = current_label.replace("[ ]", "[x]")
                        text.set_text(new_label)
                        text.set_alpha(1.0)
                else:
                    # Jika jadi hidden, ubah [x] jadi [ ] dan pudarkan
                    if "[x]" in current_label:
                        new_label = current_label.replace("[x]", "[ ]")
                        text.set_text(new_label)
                        text.set_alpha(0.5)
                
                # Redraw canvas
                ax.figure.canvas.draw()

        ax.figure.canvas.mpl_connect('pick_event', on_pick)

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
        
        text = "<h3>Analysis Results</h3>"
        
        text += f"<b>Respiratory Rate (BPM):</b> {metrics.get('respiratory_rate_bpm','N/A'):.2f}<br>"
        text += f"<b>Respiratory Freq (Hz):</b> {metrics.get('respiratory_freq_hz','N/A'):.4f}<br>"
        text += f"<b>Resp IMF Index:</b> {metrics.get('resp_imf_index','N/A')}<br>"
        text += f"<b>Resp Energy (Norm):</b> {metrics.get('resp_energy_norm','N/A'):.4f}<br><br>"
        
        text += f"<b>Vasomotor Freq (Hz):</b> {metrics.get('vasomotor_freq_hz','N/A'):.4f}<br>" 
        text += f"<b>Vasomotor IMF Index:</b> {metrics.get('vaso_imf_index','N/A')}<br>"
        text += f"<b>Vasomotor Energy (Norm):</b> {metrics.get('vaso_energy_norm','N/A'):.4f}<br>"
        
        if 'error' in metrics:
            text += f"<br><font color='red'>Error: {metrics['error']}</font>"

        lbl = QLabel(text); lbl.setWordWrap(True)
        font = lbl.font()
        font.setPointSize(11)
        lbl.setFont(font)
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

        methods_available = list(self.results.get('imf_info', {}).keys())
        if len(methods_available) == 0:
            self.append_progress("No IMF info available to render.")
            return

        preferred = None
        for cand in ['CEEMDAN','CEEMD','EEMD','EMD']:
            if cand in methods_available:
                preferred = cand
                break
        if preferred is None:
            preferred = methods_available[0]

        if key == 'IMF_COMP':
            container_key = 'IMF_COMP'
            self.clear_tab(container_key)
            
            # Tambahkan label petunjuk di atas
            hint_label = QLabel("<b>Tip:</b> Click items in the legend to toggle visibility (Checkboxes simulation)")
            hint_label.setStyleSheet("color: #aaa; font-size: 11px;")
            hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tab_containers[container_key].layout().addWidget(hint_label)

            # Overlay
            try:
                if preferred in self.results['imf_info']:
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

            # Per-IMF panels
            n_methods = len(methods_available)
            for idx in range(start, end):
                fig, axs = plt.subplots(1, n_methods, figsize=(4*n_methods,2.8), constrained_layout=True)
                if n_methods == 1:
                    axs = [axs]
                fig.suptitle(f"IMF #{idx+1}")
                t = self.time
                pre = self.results['preproc']
                
                for i, method in enumerate(methods_available):
                    entry = self.results['imf_info'][method][idx]
                    imf = entry['imf']
                    ax = axs[i]
                    
                    # Menampilkan Sinyal Input (Residu Sebelumnya)
                    input_signal_label = ""
                    if idx == 0:
                        input_signal = pre
                        input_signal_label = "[x] Raw Signal (Input)"
                    else:
                        all_method_imfs = self.results['imfs_all'][method]
                        prev_imfs_sum = np.sum(all_method_imfs[:idx], axis=0)
                        input_signal = pre - prev_imfs_sum
                        input_signal_label = f"[x] Residual Input (Signal - IMFs 1..{idx})"

                    ax.plot(t, input_signal, color='#707070', linewidth=0.8, alpha=0.5, label=input_signal_label, zorder=1)

                    # Plot IMF Saat Ini (Kuning)
                    ax.plot(t, imf, color='#ffd966', linewidth=1.2, label=f"[x] {method} IMF", zorder=2)
                    
                    if method in ['EMD', 'EEMD']:
                        max_idx, min_idx = self.processor._local_extrema(imf)
                        
                        if len(max_idx) > 0:
                            ax.scatter(t[max_idx], imf[max_idx], color='green', s=20, zorder=5, label='[x] Maxima')
                        if len(min_idx) > 0:
                            ax.scatter(t[min_idx], imf[min_idx], color='red', s=20, zorder=5, label='[x] Minima')
                        
                        if len(max_idx) > 2 and len(min_idx) > 2:
                            mx_idx_sorted = np.unique(max_idx)
                            mx_t_ext, mx_val_ext = tambah_titik_ujung(t[mx_idx_sorted], imf[mx_idx_sorted])
                            upper_env = spline_envelope(mx_t_ext, mx_val_ext, t)
                            
                            mn_idx_sorted = np.unique(min_idx)
                            mn_t_ext, mn_val_ext = tambah_titik_ujung(t[mn_idx_sorted], imf[mn_idx_sorted])
                            lower_env = spline_envelope(mn_t_ext, mn_val_ext, t)
                            
                            mean_env = (upper_env + lower_env) / 2.0
                            
                            ax.plot(t, upper_env, 'b-', linewidth=0.8, label='[x] Upper Env', zorder=3)
                            ax.plot(t, lower_env, 'orange', linewidth=0.8, label='[x] Lower Env', zorder=3)
                            ax.plot(t, mean_env, 'k--', linewidth=0.8, alpha=0.7, label='[x] Mean Env', zorder=3)

                        imfs_upto = [self.results['imf_info'][method][k]['imf'] for k in range(0, idx+1)]
                        residual_after = pre - np.sum(imfs_upto, axis=0)
                        ax.plot(t, residual_after, linestyle='--', color=RESIDUAL_COLOR, 
                                linewidth=0.8, alpha=0.8, label='[x] Residual after', zorder=3)

                    else:
                        proto = entry.get('proto', None)
                        env_info = entry.get('env_info', None)
                        
                        if proto is None:
                            sum_prev = np.zeros_like(pre)
                            for k in range(0, idx):
                                sum_prev += self.results['imf_info'][method][k]['imf']
                            proto = pre - sum_prev
                            env_info = self.processor.get_envelopes(proto)
                        
                        ax.plot(t, env_info['env_max'], linestyle='--', color="#ffb84d", linewidth=0.9, label='[x] Upper env (proto)', zorder=3)
                        ax.plot(t, env_info['env_min'], linestyle='--', color="#79c2ff", linewidth=0.9, label='[x] Lower env (proto)', zorder=3)
                        ax.plot(t, env_info['mean_env'], linestyle=':', color='#ffffff', linewidth=0.9, label='[x] Mean env (proto)', zorder=3)
                        
                        imfs_upto = [self.results['imf_info'][method][k]['imf'] for k in range(0, idx+1)]
                        residual_after = pre - np.sum(imfs_upto, axis=0)
                        ax.plot(t, residual_after, linestyle='--', color=RESIDUAL_COLOR, linewidth=0.8, alpha=0.8, label='[x] Residual after', zorder=3)

                    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
                    
                    self.make_legend_interactive(ax)

                self._add_canvas_with_toolbar(self.tab_containers[container_key], fig, height=320)
                
                meta_text_lines = []
                for method in methods_available:
                    meta = self.results.get('meta_all', {}).get(method, [])
                    if len(meta) > idx:
                        m = meta[idx]
                        if 'sifts' in m:
                            meta_text_lines.append(f"{method}: sifts={m.get('sifts')}, sd={m.get('final_sd'):.4g}, stop={m.get('stop_reason')}")
                        else:
                            meta_text_lines.append(f"{method}: sifts_mean={m.get('sifts_mean','N/A'):.2f}")
                if meta_text_lines:
                    meta_label = QLabel("   |   ".join(meta_text_lines) + f"   | UI max sifts: {self.max_sift_spin.value()}")
                    self.tab_containers[container_key].layout().addWidget(meta_label)

        elif key == 'IMF_FFT':
            container_key = 'IMF_FFT'
            self.clear_tab(container_key)
            n_methods = len(methods_available)
            for idx in range(start, end):
                fig, axs = plt.subplots(1, n_methods, figsize=(4*n_methods,2.6), constrained_layout=True)
                if n_methods == 1:
                    axs = [axs]
                fig.suptitle(f"IMF #{idx+1} - Transform Magnitude")
                for i, method in enumerate(methods_available):
                    freqs = self.results['imf_info'][method][idx]['freqs']
                    Xk = self.results['imf_info'][method][idx]['Xk']
                    half = len(freqs)//2
                    axs[i].plot(freqs[:half], np.abs(Xk)[:half], color=METHOD_COLORS.get(method,'#cccccc'))
                    axs[i].set_xlim(0,5); axs[i].set_xlabel("Frequency (Hz)"); axs[i].set_ylabel("Magnitude")
                    axs[i].legend([method], fontsize='small')
                self._add_canvas_with_toolbar(self.tab_containers[container_key], fig, height=260)

        elif key == 'IMF_PSD':
            container_key = 'IMF_PSD'
            self.clear_tab(container_key)
            n_methods = len(methods_available)
            for idx in range(start, end):
                fig, axs = plt.subplots(1, n_methods, figsize=(4*n_methods,2.6), constrained_layout=True)
                if n_methods == 1:
                    axs = [axs]
                fig.suptitle(f"IMF #{idx+1} - PSD")
                for i, method in enumerate(methods_available):
                    freqs = self.results['imf_info'][method][idx]['freqs']
                    psd = self.results['imf_info'][method][idx]['psd']
                    half = len(freqs)//2
                    axs[i].plot(freqs[:half], psd[:half], color=METHOD_COLORS.get(method,'#cccccc'))
                    axs[i].set_xlim(0,5); axs[i].set_xlabel("Frequency (Hz)"); axs[i].set_ylabel("Power")
                    axs[i].legend([method], fontsize='small')
                self._add_canvas_with_toolbar(self.tab_containers[container_key], fig, height=260)

        elif key == 'IMF_HHT':
            container_key = 'IMF_HHT'
            self.clear_tab(container_key)
            n_methods = len(methods_available)
            cols = 2 if n_methods > 1 else 1
            rows = math.ceil(n_methods / cols)
            for idx in range(start, end):
                fig, axs = plt.subplots(rows, cols, figsize=(7*cols, 3.5*rows), constrained_layout=True)
                ax_list = np.array(axs).flatten() if isinstance(axs, (list, np.ndarray)) else [axs]
                fig.suptitle(f"IMF #{idx+1} - HHT")
                for i, method in enumerate(methods_available):
                    t_h = self.results['imf_info'][method][idx]['hht_t']
                    f_cent = self.results['imf_info'][method][idx]['hht_freqs']
                    amp_map = self.results['imf_info'][method][idx]['hht_map']
                    ax = ax_list[i]
                    im = ax.pcolormesh(t_h, f_cent, amp_map, shading='auto', cmap=HHT_CMAP)
                    ax.set_ylim(0, 2.0)
                    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)"); ax.set_title(method)
                    fig.colorbar(im, ax=ax, format='%.3f')
                self._add_canvas_with_toolbar(self.tab_containers[container_key], fig, height=380)

        self.update_page_label_for_active_tab()

    # Noise tab
    def prepare_noise_tab_controls(self):
        container = self.tab_containers['NOISE']
        layout = container.layout()
        
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w: w.setParent(None)
        
        if self.results is None:
            layout.addWidget(QLabel("Run analysis to see noise trials."))
            layout.addStretch()
            return
            
        meta_all = self.results.get('meta_all', {})
        avail = [m for m in ['EEMD','CEEMD','CEEMDAN'] if m in meta_all]
        
        if len(avail) == 0:
            layout.addWidget(QLabel("No ensemble methods were run or no noise metadata available."))
            layout.addStretch()
            return

        input_group = QGroupBox("Selection Controls")
        input_layout = QHBoxLayout(input_group)
        
        self.noise_method_combo = QComboBox()
        self.noise_method_combo.addItems(avail)
        self.noise_method_combo.currentTextChanged.connect(self.on_noise_method_changed)
        input_layout.addWidget(QLabel("Method:"))
        input_layout.addWidget(self.noise_method_combo)
        input_layout.addSpacing(20) 

        self.noise_trial_spin = QSpinBox()
        self.noise_trial_spin.setRange(0, 0)
        self.noise_trial_spin.setValue(0)
        self.noise_trial_spin.setMinimumWidth(80) 
        input_layout.addWidget(QLabel("Trial index:"))
        input_layout.addWidget(self.noise_trial_spin)
        input_layout.addSpacing(20)

        self.noise_imf_spin = QSpinBox()
        self.noise_imf_spin.setRange(1, max(1, self.results['max_imfs']))
        self.noise_imf_spin.setValue(1)
        self.noise_imf_spin.setMinimumWidth(80)
        input_layout.addWidget(QLabel("IMF index (CEEMDAN):"))
        input_layout.addWidget(self.noise_imf_spin)
        
        input_layout.addStretch()
        
        layout.addWidget(input_group)

        btn_layout = QHBoxLayout()
        
        self.btn_plot_signal_plus_noise = QPushButton("Plot Signal + Noise")
        self.btn_plot_signal_plus_noise.clicked.connect(self.plot_noise_trial)
        self.btn_plot_signal_plus_noise.setMinimumHeight(35) 
        
        self.btn_plot_pair = QPushButton("Plot Complementary Pair (CEEMD)")
        self.btn_plot_pair.clicked.connect(self.plot_noise_pair)
        self.btn_plot_pair.setMinimumHeight(35)
        
        self.btn_plot_adaptive = QPushButton("Plot Adaptive Noise (CEEMDAN)")
        self.btn_plot_adaptive.clicked.connect(self.plot_adaptive_noise_for_imf)
        self.btn_plot_adaptive.setMinimumHeight(35)

        btn_layout.addWidget(self.btn_plot_signal_plus_noise)
        btn_layout.addWidget(self.btn_plot_pair)
        btn_layout.addWidget(self.btn_plot_adaptive)
        
        layout.addLayout(btn_layout)

        self.on_noise_method_changed(self.noise_method_combo.currentText())
        
        help_lbl = QLabel("<i>Select a method and trial index above, then click a button to visualize the injected noise.</i>")
        help_lbl.setStyleSheet("color: #aaa; margin-top: 10px;")
        help_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(help_lbl)

    def on_noise_method_changed(self, method: str):
        meta_all = self.results.get('meta_all', {}) if self.results else {}
        if method == 'EEMD':
            try:
                noise_trials = meta_all[method][0].get('noise_trials', [])
                ntrials = len(noise_trials)
                self.noise_trial_spin.setRange(0, max(0, ntrials-1))
            except Exception:
                self.noise_trial_spin.setRange(0, 0)
        elif method == 'CEEMD':
            try:
                pairs = meta_all[method][0].get('noise_pairs', [])
                ntrials = len(pairs)
                self.noise_trial_spin.setRange(0, max(0, ntrials-1))
            except Exception:
                self.noise_trial_spin.setRange(0, 0)
        elif method == 'CEEMDAN':
            try:
                meta_list = meta_all.get('CEEMDAN', [])
                if len(meta_list) == 0:
                    self.noise_trial_spin.setRange(0, 0)
                    return
                first_adaptive = meta_list[0].get('adaptive_noises', [])
                ntrials = len(first_adaptive)
                self.noise_trial_spin.setRange(0, max(0, ntrials-1))
                self.noise_imf_spin.setRange(1, max(1, len(meta_list)))
            except Exception:
                self.noise_trial_spin.setRange(0,0)

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
                if not noise_trials: return
                n = noise_trials[trial]
                sig_plus = pre + n
                fig, ax = plt.subplots(2,1, figsize=(12,6), constrained_layout=True)
                ax[0].plot(t, pre, color=PREP_COLOR, label='Preprocessed signal')
                ax[0].plot(t, sig_plus, color='#ffd1a9', alpha=0.9, label=f'signal + noise (trial {trial})')
                ax[0].set_xlabel("Time (s)"); ax[0].legend()
                ax[1].plot(t, n, color='#ff6666', label='noise (trial)')
                ax[1].set_xlabel("Time (s)"); ax[1].legend()
                self._add_canvas_with_toolbar(layout, fig, height=420)
            except Exception as e:
                self.append_progress(f"Error: {e}")
        elif method == 'CEEMD':
            try:
                pairs = meta_all['CEEMD'][0].get('noise_pairs', [])
                if not pairs: return
                noise_plus, noise_minus = pairs[trial]
                sig_plus = pre + noise_plus
                sig_minus = pre + noise_minus
                fig, ax = plt.subplots(3,1, figsize=(12,8), constrained_layout=True)
                ax[0].plot(t, pre, color=PREP_COLOR, label='Preprocessed')
                ax[0].plot(t, sig_plus, color='#ffd1a9', label=f'signal + noise (trial {trial})')
                ax[0].plot(t, sig_minus, color='#c0e8ff', label=f'signal - noise (trial {trial})')
                ax[0].legend(); ax[0].set_xlabel("Time (s)")
                ax[1].plot(t, noise_plus, label='noise +', color='#ff6666')
                ax[2].plot(t, noise_minus, label='noise -', color='#66aaff')
                self._add_canvas_with_toolbar(layout, fig, height=480)
            except Exception as e:
                self.append_progress(f"Error: {e}")
        elif method == 'CEEMDAN':
            try:
                meta_list = meta_all.get('CEEMDAN', [])
                imf_idx = max(1, int(self.noise_imf_spin.value()))
                adaptive = meta_list[imf_idx-1].get('adaptive_noises', [])
                noise = adaptive[trial]
                sig_plus = pre + noise
                fig, ax = plt.subplots(2,1, figsize=(12,6), constrained_layout=True)
                ax[0].plot(t, pre, color=PREP_COLOR, label='Preprocessed')
                ax[0].plot(t, sig_plus, color='#ffd1a9', label=f'signal + adaptive noise (trial {trial})')
                ax[0].legend(); ax[1].plot(t, noise, label='adaptive noise')
                ax[1].set_xlabel("Time (s)")
                self._add_canvas_with_toolbar(layout, fig, height=420)
            except Exception as e:
                self.append_progress(f"Error: {e}")

    def plot_noise_pair(self):
        method = self.noise_method_combo.currentText()
        trial = int(self.noise_trial_spin.value())
        container = self.tab_containers['NOISE']
        layout = container.layout()
        self.prepare_noise_tab_controls()
        pre = self.results['preproc']
        t = self.time
        meta_all = self.results.get('meta_all', {})
        if method == 'CEEMD':
            try:
                pairs = meta_all['CEEMD'][0].get('noise_pairs', [])
                noise_plus, noise_minus = pairs[trial]
                fig, ax = plt.subplots(2,1, figsize=(12,6), constrained_layout=True)
                ax[0].plot(t, pre, color=PREP_COLOR, label='pre')
                ax[0].plot(t, pre + noise_plus, label='pre + noise+')
                ax[0].plot(t, pre + noise_minus, label='pre + noise-')
                ax[0].legend()
                ax[1].plot(t, noise_plus, label='noise+'); ax[1].plot(t, noise_minus, label='noise-')
                ax[1].legend()
                self._add_canvas_with_toolbar(layout, fig, height=420)
            except Exception as e:
                self.append_progress(f"Error: {e}")
        else:
            self.append_progress("Available for CEEMD only.")

    def plot_adaptive_noise_for_imf(self):
        trial = int(self.noise_trial_spin.value())
        imf_idx = max(1, int(self.noise_imf_spin.value()))
        container = self.tab_containers['NOISE']
        layout = container.layout()
        self.prepare_noise_tab_controls()
        pre = self.results['preproc']
        t = self.time
        meta_all = self.results.get('meta_all', {})
        try:
            meta_list = meta_all.get('CEEMDAN', [])
            adaptive = meta_list[imf_idx-1].get('adaptive_noises', [])
            noise = adaptive[trial]
            fig, ax = plt.subplots(2,1, figsize=(12,6), constrained_layout=True)
            ax[0].plot(t, pre, color=PREP_COLOR, label='pre')
            ax[0].plot(t, pre + noise, label=f'pre + adaptive noise (trial {trial})')
            ax[0].legend()
            ax[1].plot(t, noise, label='adaptive noise')
            ax[1].legend()
            self._add_canvas_with_toolbar(layout, fig, height=420)
        except Exception as e:
            self.append_progress(f"Error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())