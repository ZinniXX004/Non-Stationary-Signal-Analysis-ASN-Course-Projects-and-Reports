"""
-+-+-+ HRV ANALYSIS GUI APPLICATION (FIXED) +-+-+-
Frontend for the revised PPG analysis logic.
Author: Jeremia Manalu (Revisions by Assistant)

Changes:
- Added 'Original Signal (Preprocessed)' option to ComboBoxes.
- Fixed 'IndexError' in plotting DWT by using correct time vector (ds_time).
- Fixed 'IndexError' in Peak plotting by explicit int casting.
- All plots include proper Titles and Axis Labels.
"""

import sys
import os
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QLabel, QLineEdit,
                             QSpinBox, QFileDialog, QTextEdit, QGroupBox, QGridLayout,
                             QStatusBar, QProgressBar, QComboBox, QCheckBox, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

# Import backend yang sudah diperbaiki
from PPG_main_fixed import (PPGStressAnalyzer, HRV_Analyzer, welch_from_scratch, 
                              extract_rate_from_signal, filter_custom_ref)

class MplCanvasWithToolbar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = PlotCanvas(self)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.toolbar.setStyleSheet("background-color: #4a5568;")

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#2d3748')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.apply_dark_theme()

    def apply_dark_theme(self):
        self.axes.set_facecolor('#1a202c')
        for spine in self.axes.spines.values(): spine.set_color('#a0aec0')
        self.axes.xaxis.label.set_color('#a0aec0')
        self.axes.yaxis.label.set_color('#a0aec0')
        self.axes.title.set_color('white')
        self.axes.tick_params(axis='x', colors='#a0aec0')
        self.axes.tick_params(axis='y', colors='#a0aec0')
        self.fig.tight_layout()

    def plot_data(self, plot_instructions):
        self.axes.clear()
        
        # Handle special plot types (Autonomic Balance Background)
        if any(p.get('type') == 'rect' for p in plot_instructions):
             for instruction in plot_instructions:
                 if instruction.get('type') == 'rect':
                     rect = patches.Rectangle(
                         (instruction['x'], instruction['y']), 
                         instruction['w'], instruction['h'], 
                         linewidth=0, facecolor=instruction['color'], alpha=instruction.get('alpha', 1.0)
                     )
                     self.axes.add_patch(rect)
                     if 'text' in instruction:
                         self.axes.text(instruction['x'] + instruction['w']/2, instruction['y'] + instruction['h']/2, 
                                        instruction['text'], ha='center', va='center', 
                                        color='black', fontsize=12, fontweight='bold')

        # Standard Plots
        for instruction in plot_instructions:
            plot_type = instruction.get('type', 'plot')
            style = instruction.get('style', {})
            x_data = instruction.get('x', [])
            y_data = instruction.get('y', [])
            
            if plot_type == 'rect': continue # Handled above

            if plot_type == 'axhline': 
                self.axes.axhline(y=instruction.get('y', 0), label=instruction.get('label'), **style)
            elif plot_type == 'axvline': 
                self.axes.axvline(x=instruction.get('x', 0), label=instruction.get('label'), **style)
            elif plot_type == 'fill_between': 
                self.axes.fill_between(x_data, instruction.get('y1'), instruction.get('y2', 0), label=instruction.get('label'), **style)
            elif len(x_data) > 0 and len(y_data) > 0:
                if plot_type == 'scatter': 
                    self.axes.scatter(x_data, y_data, label=instruction.get('label'), **style)
                elif plot_type == 'bar': 
                    self.axes.bar(x_data, y_data, label=instruction.get('label'), **style)
                else: 
                    self.axes.plot(x_data, y_data, label=instruction.get('label'), **style)
                    
        self.axes.set_title(plot_instructions[0].get('title', ''))
        self.axes.set_xlabel(plot_instructions[0].get('xlabel', ''))
        self.axes.set_ylabel(plot_instructions[0].get('ylabel', ''))
        
        if xlim := plot_instructions[0].get('xlim'): self.axes.set_xlim(xlim)
        if ylim := plot_instructions[0].get('ylim'): self.axes.set_ylim(ylim)
        
        if any('label' in p for p in plot_instructions):
            legend = self.axes.legend()
            if legend:
                legend.get_frame().set_facecolor('#2d3748')
                plt.setp(legend.get_texts(), color='#a0aec0')
                
        self.axes.grid(True, linestyle='--', alpha=0.2)
        self.apply_dark_theme()
        self.draw()

class AnalysisThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, analyzer_instance, file_path, column_name, downsample, ds_factor, hrv_source_str, rr_source_str, vma_source_str):
        super().__init__()
        self.analyzer = analyzer_instance
        self.file_path, self.column_name = file_path, column_name
        self.downsample, self.ds_factor = downsample, ds_factor if downsample else 1
        self.hrv_source_str, self.rr_source_str, self.vma_source_str = hrv_source_str, rr_source_str, vma_source_str

    def _get_signal_from_source(self, source_str, orig_processed_signal, downsampled_signal, dwt_coeffs):
        """
        Mengambil sinyal berdasarkan pilihan user di ComboBox.
        """
        if source_str == 'Original Signal (Preprocessed)':
            return orig_processed_signal
        elif source_str == 'Downsampled Signal': 
            return downsampled_signal
        
        try:
            # Parse 'DWT Qx'
            level = int(source_str.split(' ')[-1].replace('Q', ''))
            return dwt_coeffs.get(level)
        except: return None

    def run(self):
        try:
            self.progress.emit(5, f"Loading '{self.column_name}'...")
            time, signal = self.analyzer.load_ppg_data(self.file_path, self.column_name)
            if signal is None: self.error.emit(f"Failed to load data."); return
            
            orig_fs = self.analyzer.original_fs
            
            # --- 1. Compute Original Preprocessed Signal (For "Original" Option) ---
            # Kita lakukan preprocessing di sini agar tersedia sebagai source
            # Gunakan HRV_Analyzer sementara untuk meminjam logika preprocessing-nya
            temp_analyzer = HRV_Analyzer(signal, time, orig_fs, self.analyzer.fft_from_scratch)
            temp_analyzer._preprocess_and_filter()
            orig_processed_signal = temp_analyzer.preprocessed_signal
            
            self.progress.emit(10, f"FFT Original (FS={orig_fs:.1f} Hz)...")
            fft_orig = self.analyzer.fft_magnitude_and_frequencies(signal)

            self.progress.emit(15, "Downsampling...")
            ds_signal, ds_time = self.analyzer.downsample_signal(signal, time, self.ds_factor)
            ds_fs = self.analyzer.fs
            
            self.progress.emit(20, f"FFT Downsampled (FS={ds_fs:.1f} Hz)...")
            fft_ds = self.analyzer.fft_magnitude_and_frequencies(ds_signal)
            
            self.progress.emit(30, "DWT Decomposition...")
            dwt_coeffs = self.analyzer.dwt_convolution_from_scratch(ds_signal)

            # --- 2. HRV Analysis ---
            self.progress.emit(45, f"HRV Analysis on '{self.hrv_source_str}'...")
            
            # Determine appropriate FS and Time vector based on source
            hrv_input_sig = self._get_signal_from_source(self.hrv_source_str, orig_processed_signal, ds_signal, dwt_coeffs)
            
            if hrv_input_sig is None: 
                self.error.emit("Invalid HRV Source Selected"); return
            
            # Jika source original, pakai orig_fs & time. Jika tidak, pakai ds_fs & ds_time
            if self.hrv_source_str == 'Original Signal (Preprocessed)':
                current_fs = orig_fs
                current_time = time
            else:
                current_fs = ds_fs
                current_time = ds_time
            
            # Run HRV Logic
            hrv_eng = HRV_Analyzer(hrv_input_sig, current_time, current_fs, self.analyzer.fft_from_scratch)
            hrv_res = hrv_eng.run_all_analyses()
            
            fft_proc = self.analyzer.fft_magnitude_and_frequencies(hrv_eng.preprocessed_signal)

            # --- 3. Respiratory Rate ---
            self.progress.emit(70, "Estimating Respiratory Rate...")
            rr_sig = self._get_signal_from_source(self.rr_source_str, orig_processed_signal, ds_signal, dwt_coeffs)
            
            # Determine correct FS for RR calculation
            if self.rr_source_str == 'Original Signal (Preprocessed)':
                rr_fs = orig_fs
            else:
                rr_fs = ds_fs
            
            rr_clean = filter_custom_ref(rr_sig - np.mean(rr_sig), rr_fs, 0.01, 8.0)
            resp_rate_hz = extract_rate_from_signal(rr_clean, rr_fs, (0.15, 0.4))
            resp_rate_brpm = resp_rate_hz * 60.0

            # --- 4. Vasomotor Activity ---
            self.progress.emit(80, "Estimating Vasomotor Activity...")
            vma_sig = self._get_signal_from_source(self.vma_source_str, orig_processed_signal, ds_signal, dwt_coeffs)
            
            # Determine correct FS for VMA calculation
            if self.vma_source_str == 'Original Signal (Preprocessed)':
                vma_fs = orig_fs
            else:
                vma_fs = ds_fs
            
            vma_clean = filter_custom_ref(vma_sig - np.mean(vma_sig), vma_fs, 0.01, 8.0)
            vaso_rate_hz = extract_rate_from_signal(vma_clean, vma_fs, (0.04, 0.15))
            vma_psd = welch_from_scratch(vma_clean, vma_fs)

            # --- 5. DWT Responses ---
            self.progress.emit(90, "DWT Responses...")
            orig_resp = self.analyzer.calculate_qj_frequency_responses(orig_fs)
            ds_resp = self.analyzer.calculate_qj_frequency_responses(ds_fs)
            dwt_resps = {i: {'orig': orig_resp.get(i), 'ds': ds_resp.get(i)} for i in range(1, 9)}
            
            self.progress.emit(100, "Finalizing...")
            self.finished.emit({
                'raw_signal': signal, 'raw_time': time, 'fft_original': fft_orig,
                'ds_signal': ds_signal, 'ds_time': ds_time, 'fft_downsampled': fft_ds,
                'proc_signal': hrv_eng.preprocessed_signal, 'proc_time': current_time, 'fft_proc': fft_proc,
                'hrv': hrv_res, 'dwt': dwt_coeffs, 'dwt_resp': dwt_resps,
                'vma_psd': vma_psd, 'orig_fs': orig_fs, 'ds_fs': ds_fs,
                'resp_brpm': resp_rate_brpm, 'vaso_hz': vaso_rate_hz,
                'hrv_src': self.hrv_source_str
            })
            
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = PPGStressAnalyzer()
        self.results = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PPG HRV Analysis (Delphi Port) - Jeremia Manalu (5023231017)")
        self.setGeometry(50, 50, 1600, 900)
        self.setStyleSheet("""QWidget { background-color: #2d3748; color: #e2e8f0; font-size: 14px; } QMainWindow { background-color: #1a202c; } QTabWidget::pane { border: none; } QTabBar::tab { background-color: #4a5568; padding: 12px 20px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; font-weight: bold; } QTabBar::tab:selected { background-color: #2d3748; border-bottom: 3px solid #63b3ed; } QGroupBox { border: 1px solid #4a5568; border-radius: 8px; margin-top: 1ex; font-weight: bold; font-size: 16px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; } QPushButton { background-color: #2b6cb0; border-radius: 5px; padding: 8px; font-weight: bold; border: 1px solid #2c5282; } QPushButton:hover { background-color: #3182ce; } QPushButton:disabled { background-color: #4a5568; color: #718096; } QLabel { margin-top: 5px; } QLineEdit, QSpinBox, QComboBox { background-color: #1a202c; border: 1px solid #4a5568; padding: 5px; border-radius: 5px; } QTextEdit { background-color: #1a202c; border: 1px solid #4a5568; font-family: "Consolas", "Courier New", monospace; } QProgressBar { border-radius: 5px; text-align: center; color: black; font-weight: bold; } QProgressBar::chunk { background-color: #68d391; border-radius: 5px;} QStatusBar { font-weight: bold; } QScrollArea { border: none; }""")
        main_widget = QWidget(); self.setCentralWidget(main_widget); main_layout = QVBoxLayout(main_widget)
        top_layout = QHBoxLayout(); top_layout.addWidget(self._create_control_panel()); self.tabs = QTabWidget(); self._create_tabs(); top_layout.addWidget(self.tabs); main_layout.addLayout(top_layout)
        bottom_bar = QHBoxLayout(); self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.progress_bar = QProgressBar(); self.status_bar.addPermanentWidget(self.progress_bar, 1); self.progress_bar.hide()
        clear_btn = QPushButton("Clear All"); exit_btn = QPushButton("Exit Application"); clear_btn.setFixedWidth(120); exit_btn.setFixedWidth(150)
        bottom_bar.addStretch(); bottom_bar.addWidget(clear_btn); bottom_bar.addWidget(exit_btn); main_layout.addLayout(bottom_bar)
        clear_btn.clicked.connect(self.clear_all_plots); exit_btn.clicked.connect(self.close)

    def _create_control_panel(self):
        panel = QGroupBox("Controls & Parameters"); layout = QGridLayout(panel)
        self.file_path_le = QLineEdit("Select CSV..."); self.file_path_le.setReadOnly(True); browse_btn = QPushButton("Browse..."); self.column_combo = QComboBox()
        layout.addWidget(QLabel("Data File:"), 0, 0, 1, 2); layout.addWidget(self.file_path_le, 1, 0, 1, 2); layout.addWidget(browse_btn, 1, 2)
        layout.addWidget(QLabel("Signal Column:"), 2, 0, 1, 3); layout.addWidget(self.column_combo, 3, 0, 1, 3)
        self.downsample_cb = QCheckBox("Enable Downsampling"); self.downsample_cb.setChecked(True); self.downsample_sb = QSpinBox(); self.downsample_sb.setRange(1, 20); self.downsample_sb.setValue(7)
        layout.addWidget(self.downsample_cb, 4, 0, 1, 2); layout.addWidget(self.downsample_sb, 4, 2)
        
        # Updated Options
        source_options = ['Downsampled Signal', 'Original Signal (Preprocessed)'] + [f'DWT Q{i}' for i in range(1, 9)]
        
        self.hrv_source_combo = QComboBox(); self.hrv_source_combo.addItems(source_options)
        self.rr_source_combo = QComboBox(); self.rr_source_combo.addItems(source_options); self.rr_source_combo.setCurrentText("DWT Q5")
        self.vma_source_combo = QComboBox(); self.vma_source_combo.addItems(source_options); self.vma_source_combo.setCurrentText("DWT Q6")
        
        layout.addWidget(QLabel("HRV Metrics Source:"), 5, 0, 1, 3); layout.addWidget(self.hrv_source_combo, 6, 0, 1, 3)
        layout.addWidget(QLabel("Resp. Rate Source:"), 7, 0, 1, 3); layout.addWidget(self.rr_source_combo, 8, 0, 1, 3)
        layout.addWidget(QLabel("Vasomotor Source:"), 9, 0, 1, 3); layout.addWidget(self.vma_source_combo, 10, 0, 1, 3)
        self.run_analysis_btn = QPushButton("RUN ANALYSIS PIPELINE"); layout.addWidget(self.run_analysis_btn, 11, 0, 1, 3)
        self.results_te = QTextEdit(); self.results_te.setReadOnly(True); layout.addWidget(QLabel("Analysis Results:"), 12, 0, 1, 3); layout.addWidget(self.results_te, 13, 0, 1, 3)
        panel.setFixedWidth(400)
        browse_btn.clicked.connect(self.browse_file); self.run_analysis_btn.clicked.connect(self.run_full_analysis); self.downsample_cb.stateChanged.connect(self.downsample_sb.setEnabled)
        return panel

    def _create_scrollable_tab(self, layout):
        widget = QWidget(); widget.setLayout(layout); scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(widget); return scroll

    def _create_tabs(self):
        # 1. Signal Processing
        l1 = QGridLayout()
        self.pl_raw, self.pl_fft_raw = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self)
        self.pl_ds, self.pl_fft_ds = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self)
        self.pl_proc, self.pl_fft_proc = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self)
        l1.addWidget(self.pl_raw, 0, 0); l1.addWidget(self.pl_fft_raw, 0, 1)
        l1.addWidget(self.pl_ds, 1, 0); l1.addWidget(self.pl_fft_ds, 1, 1)
        l1.addWidget(self.pl_proc, 2, 0); l1.addWidget(self.pl_fft_proc, 2, 1)
        self.tabs.addTab(self._create_scrollable_tab(l1), "1. Signal Processing")
        
        # 2. Peak Detection
        l2 = QHBoxLayout(); self.pl_peaks = MplCanvasWithToolbar(self); l2.addWidget(self.pl_peaks)
        self.tabs.addTab(self._create_scrollable_tab(l2), "2. Peak Detection")
        
        # 3. HRV Time
        l3 = QHBoxLayout(); self.pl_tach, self.pl_hist = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self)
        l3.addWidget(self.pl_tach); l3.addWidget(self.pl_hist)
        self.tabs.addTab(self._create_scrollable_tab(l3), "3. HRV - Time")
        
        # 4. HRV Freq
        l4 = QHBoxLayout(); self.pl_hrv_psd, self.pl_vaso = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self)
        l4.addWidget(self.pl_hrv_psd); l4.addWidget(self.pl_vaso)
        self.tabs.addTab(self._create_scrollable_tab(l4), "4. HRV - Freq.")
        
        # 5. Non-Linear
        l5 = QHBoxLayout(); self.pl_poincare = MplCanvasWithToolbar(self); l5.addWidget(self.pl_poincare)
        self.tabs.addTab(self._create_scrollable_tab(l5), "5. HRV - Non-linear")
        
        # 6. Autonomic Balance
        l6 = QHBoxLayout(); self.pl_balance = MplCanvasWithToolbar(self); l6.addWidget(self.pl_balance)
        self.tabs.addTab(self._create_scrollable_tab(l6), "6. Autonomic Balance")
        
        # 7. DWT Coeffs
        self.dwt_tabs = QTabWidget(); self.dwt_plots = {}
        for i in range(1, 9):
            tl = QGridLayout()
            tp, tf, tpsd, tfr = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self), MplCanvasWithToolbar(self), MplCanvasWithToolbar(self)
            tl.addWidget(tp, 0, 0); tl.addWidget(tf, 0, 1); tl.addWidget(tpsd, 1, 0); tl.addWidget(tfr, 1, 1)
            self.dwt_plots[i] = {'time': tp, 'fft': tf, 'psd': tpsd, 'resp': tfr}
            self.dwt_tabs.addTab(self._create_scrollable_tab(tl), f"Q{i}")
        self.tabs.addTab(self.dwt_tabs, "7. DWT Coefficients")
        
        # 8. DWT Response
        l8 = QHBoxLayout(); self.pl_resp_orig, self.pl_resp_ds = MplCanvasWithToolbar(self), MplCanvasWithToolbar(self)
        l8.addWidget(self.pl_resp_orig); l8.addWidget(self.pl_resp_ds)
        self.tabs.addTab(self._create_scrollable_tab(l8), "8. DWT Freq. Response")

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.file_path_le.setText(file_path)
            try:
                headers = pd.read_csv(file_path, nrows=0).columns.tolist()
                self.column_combo.clear(); self.column_combo.addItems(headers)
                if (pi := next((i for i, c in enumerate(headers) if 'pleth' in c.lower()), -1)) != -1:
                    self.column_combo.setCurrentIndex(pi)
            except Exception as e: self.status_bar.showMessage(f"Header Error: {e}", 5000)

    def run_full_analysis(self):
        fp = self.file_path_le.text()
        if not fp or not os.path.exists(fp): self.status_bar.showMessage("Invalid File", 5000); return
        self.run_analysis_btn.setEnabled(False); self.progress_bar.show()
        
        self.th = AnalysisThread(self.analyzer, fp, self.column_combo.currentText(), 
                                 self.downsample_cb.isChecked(), self.downsample_sb.value(), 
                                 self.hrv_source_combo.currentText(), self.rr_source_combo.currentText(), 
                                 self.vma_source_combo.currentText())
        self.th.progress.connect(lambda p, m: [self.progress_bar.setValue(p), self.status_bar.showMessage(m)])
        self.th.finished.connect(self.on_done); self.th.error.connect(self.on_err); self.th.start()

    def on_done(self, res):
        self.results = res; self.update_plots(); self.update_text()
        self.status_bar.showMessage("Done.", 5000); self.run_analysis_btn.setEnabled(True); self.progress_bar.hide()

    def on_err(self, msg):
        self.status_bar.showMessage(f"Error: {msg}", 10000); self.results_te.setText(msg); self.run_analysis_btn.setEnabled(True); self.progress_bar.hide()

    def clear_all_plots(self):
        for w in self.findChildren(MplCanvasWithToolbar): w.canvas.axes.clear(); w.canvas.apply_dark_theme(); w.canvas.draw()
        self.results_te.clear()

    def update_plots(self):
        if not self.results: return
        r = self.results; h = r['hrv']; td, fd, nl = h['time_domain'], h['frequency_domain'], h['nonlinear']
        
        # 1. Signals
        self.pl_raw.canvas.plot_data([{'x': r['raw_time'], 'y': r['raw_signal'], 'title':'Raw Signal', 'xlabel':'Time (s)', 'ylabel':'Amplitude', 'style':{'color':'#63b3ed'}}])
        self.pl_fft_raw.canvas.plot_data([{'x': r['fft_original'][0], 'y': r['fft_original'][1], 'title':'FFT Raw', 'xlabel':'Hz', 'ylabel':'Mag', 'style':{'color':'#63b3ed'}, 'xlim':(0, r['orig_fs']/2)}])
        self.pl_ds.canvas.plot_data([{'x': r['ds_time'], 'y': r['ds_signal'], 'title':'Downsampled', 'xlabel':'Time (s)', 'ylabel':'Amp', 'style':{'color':'#faf089'}}])
        self.pl_fft_ds.canvas.plot_data([{'x': r['fft_downsampled'][0], 'y': r['fft_downsampled'][1], 'title':'FFT DS', 'xlabel':'Hz', 'ylabel':'Mag', 'style':{'color':'#faf089'}, 'xlim':(0, r['ds_fs']/2)}])
        self.pl_proc.canvas.plot_data([{'x': r['proc_time'], 'y': r['proc_signal'], 'title':'Pre-processed', 'xlabel':'Time (s)', 'ylabel':'Norm. Amp', 'style':{'color':'#68d391'}}])
        self.pl_fft_proc.canvas.plot_data([{'x': r['fft_proc'][0], 'y': r['fft_proc'][1], 'title':'FFT Proc', 'xlabel':'Hz', 'ylabel':'Mag', 'style':{'color':'#68d391'}, 'xlim':(0, r['ds_fs']/2)}])

        # 2. Peaks (FIXED FOR INTEGER INDEXING)
        pi = [{'x': r['proc_time'], 'y': r['proc_signal'], 'label':'Signal', 'title':'Detected Peaks', 'xlabel':'Time (s)', 'ylabel':'Amp', 'style':{'color':'#4299e1', 'alpha':0.7}}]
        
        if (pk := h.get('peaks')) is not None and len(pk) > 0: 
            pk_int = pk.astype(int) 
            pi.append({'type':'scatter', 'x':r['proc_time'][pk_int], 'y':r['proc_signal'][pk_int], 'label':'Peaks', 'style':{'color':'#fc8181', 'marker':'x', 's':50}})
            
        if (mn := h.get('minima')) is not None and len(mn) > 0: 
            mn_int = mn.astype(int)
            pi.append({'type':'scatter', 'x':r['proc_time'][mn_int], 'y':r['proc_signal'][mn_int], 'label':'Minima', 'style':{'color':'#68d391', 'marker':'o', 's':30}})
            
        self.pl_peaks.canvas.plot_data(pi)

        # 3. Tachogram & Hist
        if (rt := h['rr_times']) is not None and len(rt) > 1:
            self.pl_tach.canvas.plot_data([{'x': rt[1:], 'y': h['rr_intervals_s']*1000, 'title':'Tachogram', 'xlabel':'Time (s)', 'ylabel':'RR (ms)', 'style':{'color':'#faf089', 'marker':'.'}}])
        
        hist_c, hist_b = td['rr_histogram']
        if len(hist_c) > 0:
            bw = hist_b[1] - hist_b[0]
            self.pl_hist.canvas.plot_data([{'type':'bar', 'x': hist_b[:-1]+bw/2, 'y': hist_c, 'title':'RR Histogram', 'xlabel':'RR (ms)', 'ylabel':'Count', 'style':{'width':bw, 'color':'#f6ad55'}}])

        # 4. HRV PSD & Vasomotor
        if len(fd['psd_freqs']) > 0:
            self.pl_hrv_psd.canvas.plot_data([
                {'x': fd['psd_freqs'], 'y': fd['psd_values'], 'title':'HRV PSD', 'xlabel':'Hz', 'ylabel':'ms²/Hz'},
                {'type':'axvline', 'x':0.04, 'label':'VLF/LF', 'style':{'color':'orange', 'ls':'--'}},
                {'type':'axvline', 'x':0.15, 'label':'LF/HF', 'style':{'color':'cyan', 'ls':'--'}}
            ])
        
        vp_f, vp_v = r['vma_psd']
        if len(vp_f) > 0:
            self.pl_vaso.canvas.plot_data([
                {'x': vp_f, 'y': vp_v, 'title':'Vasomotor PSD', 'xlabel':'Hz', 'ylabel':'Power'},
                {'type':'fill_between', 'x': vp_f, 'y1': vp_v, 'y2': 0, 'label':'LF Band', 'style':{'where':(vp_f>=0.04)&(vp_f<=0.15), 'color':'#f6ad55', 'alpha':0.5}}
            ])

        # 5. Poincare
        if len(nl['poincare_x']) > 0:
            self.pl_poincare.canvas.plot_data([
                {'type':'scatter', 'x': nl['poincare_x'], 'y': nl['poincare_y'], 'title':'Poincaré Plot', 'xlabel':'RR[n]', 'ylabel':'RR[n+1]', 'style':{'color':'#81e6d9', 'alpha':0.6}},
                {'x':[0, 2000], 'y':[0, 2000], 'label':'Identity', 'style':{'color':'white', 'ls':'--'}}
            ])

        # 6. Autonomic Balance
        ln_lf = np.log(fd['lf_power']) if fd['lf_power'] > 1 else 0
        ln_hf = np.log(fd['hf_power']) if fd['hf_power'] > 1 else 0
        
        bg_instr = []
        bg_instr.append({'type':'rect', 'x':2, 'y':2, 'w':2.25, 'h':2.25, 'color':'#FFB6C1', 'text':'1'}) 
        bg_instr.append({'type':'rect', 'x':4.25, 'y':2, 'w':2.25, 'h':2.25, 'color':'#FFD700', 'text':'2'}) 
        bg_instr.append({'type':'rect', 'x':6.5, 'y':2, 'w':2.25, 'h':2.25, 'color':'#FFB6C1', 'text':'3'}) 
        bg_instr.append({'type':'rect', 'x':2, 'y':4.25, 'w':2.25, 'h':2.25, 'color':'#87CEFA', 'text':'4'}) 
        bg_instr.append({'type':'rect', 'x':4.25, 'y':4.25, 'w':2.25, 'h':2.25, 'color':'#90EE90', 'text':'5'}) 
        bg_instr.append({'type':'rect', 'x':6.5, 'y':4.25, 'w':2.25, 'h':2.25, 'color':'#87CEFA', 'text':'6'}) 
        bg_instr.append({'type':'rect', 'x':2, 'y':6.5, 'w':2.25, 'h':2.25, 'color':'#FFB6C1', 'text':'7'}) 
        bg_instr.append({'type':'rect', 'x':4.25, 'y':6.5, 'w':2.25, 'h':2.25, 'color':'#FFD700', 'text':'8'}) 
        bg_instr.append({'type':'rect', 'x':6.5, 'y':6.5, 'w':2.25, 'h':2.25, 'color':'#FFB6C1', 'text':'9'}) 
        
        if ln_lf > 0 and ln_hf > 0:
            bg_instr.append({'type':'scatter', 'x':[ln_lf], 'y':[ln_hf], 'label':'Patient', 'style':{'color':'red', 's':100, 'edgecolors':'black', 'zorder':10}})
            
        bg_instr[0].update({'title':'Autonomic Balance (Delphi)', 'xlabel':'Sympathetic (Ln LF)', 'ylabel':'Parasympathetic (Ln HF)', 'xlim':(2, 8.75), 'ylim':(2, 8.75)})
        self.pl_balance.canvas.plot_data(bg_instr)

        # 7 & 8. DWT (CRITICAL FIX FOR DIMENSION ERROR)
        dwt, dresp = r['dwt'], r['dwt_resp']
        for i in range(1, 9):
            if i in dwt:
                sig = dwt[i]
                df, dm = self.analyzer.fft_magnitude_and_frequencies(sig)
                pf, pv = welch_from_scratch(sig, r['ds_fs'])
                
                # FIXED: Use 'ds_time' instead of 'proc_time' for DWT plotting
                self.dwt_plots[i]['time'].canvas.plot_data([{'x': r['ds_time'], 'y': sig, 'title':f'Q{i} Time', 'xlabel':'s', 'ylabel':'Amp', 'style':{'color':'#faf089'}}])
                self.dwt_plots[i]['fft'].canvas.plot_data([{'x': df, 'y': dm, 'title':f'Q{i} FFT', 'xlabel':'Hz', 'ylabel':'Mag', 'style':{'color':'#f6ad55'}, 'xlim':(0, r['ds_fs']/2)}])
                self.dwt_plots[i]['psd'].canvas.plot_data([{'x': pf, 'y': pv, 'title':f'Q{i} PSD', 'xlabel':'Hz', 'ylabel':'PSD', 'style':{'color':'#81e6d9'}}])
                
                rf, rm = dresp[i]['ds']
                self.dwt_plots[i]['resp'].canvas.plot_data([{'x': rf, 'y': rm, 'title':f'Q{i} Resp', 'xlabel':'Hz', 'ylabel':'Mag', 'style':{'color':'#f6ad55'}}])

        c = plt.cm.viridis(np.linspace(0, 1, 8))
        ro = [{'x': dresp[i]['orig'][0], 'y': dresp[i]['orig'][1], 'label':f'Q{i}', 'style':{'color':c[i-1]}} for i in range(1,9)]
        ro[0].update({'title':'Response @ Orig FS', 'xlabel':'Hz', 'ylabel':'Mag'}); self.pl_resp_orig.canvas.plot_data(ro)
        
        rd = [{'x': dresp[i]['ds'][0], 'y': dresp[i]['ds'][1], 'label':f'Q{i}', 'style':{'color':c[i-1]}} for i in range(1,9)]
        rd[0].update({'title':'Response @ DS FS', 'xlabel':'Hz', 'ylabel':'Mag'}); self.pl_resp_ds.canvas.plot_data(rd)

    def update_text(self):
        if not self.results: return
        r = self.results; td = r['hrv']['time_domain']; fd = r['hrv']['frequency_domain']; nl = r['hrv']['nonlinear']
        t = f"FS Orig: {r['orig_fs']:.2f} Hz | FS DS: {r['ds_fs']:.2f} Hz\n"
        t += f"Resp Rate: {r['resp_brpm']:.2f} BrPM\n"
        t += f"Vaso Peak: {r['vaso_hz']:.4f} Hz\n\n"
        t += "TIME DOMAIN\n"
        for k, v in td.items(): 
            if isinstance(v, (int, float)): t += f"{k}: {v:.4f}\n"
        t += "\nFREQUENCY DOMAIN\n"
        for k, v in fd.items():
            if isinstance(v, (int, float)): t += f"{k}: {v:.4f}\n"
        t += "\nNON-LINEAR\n"
        for k, v in nl.items():
            if isinstance(v, (int, float)): t += f"{k}: {v:.4f}\n"
        self.results_te.setText(t)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())