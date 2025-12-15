"""
GUI.py

Purpose:
    - Main User Interface for EEG Analysis BCI Project (Version 4.0).
    - Features:
        1. Epoch Inspector (View raw trials C3/Cz/C4).
        2. Signal Processing Pipeline (CWT -> ERD).
        3. Machine Learning (CSP -> 8 Classifiers).
    - Theme: Retro Hacker Terminal (Dark Mode).

Dependencies:
    - PyQt6, matplotlib, numpy
    - Custom Modules: load_data_eeg_mne, CWT, filtering_BPF_EEG, 
      squaring_EEG, average_all_EEG_trials, moving_average_EEG, 
      percentage_ERD_ERS, csp_scratch, ml_analysis
"""

import sys
import os
import numpy as np
import shutil

# PyQt6 Components
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTabWidget, 
                             QFileDialog, QLineEdit, QComboBox, QMessageBox, 
                             QGroupBox, QTextEdit, QSizePolicy, QSpinBox)
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt

# Matplotlib Components
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Custom Modules
import load_data_eeg_mne
import CWT
import filtering_BPF_EEG
import squaring_EEG
import average_all_EEG_trials
import moving_average_EEG
import percentage_ERD_ERS
import csp_scratch
import ml_analysis

# =========================================================
# Custom Canvas with Toolbar (High Contrast)
# =========================================================
class PlotWidget(QWidget):
    """
    Widget wrapper for Matplotlib Figure + Navigation Toolbar.
    Configured for Dark Theme (Hacker Style).
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super(PlotWidget, self).__init__(parent)
        
        # Main Vertical Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Initialize Figure with Dark Background
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#0d0d0d') 
        
        # Initialize Canvas
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #0d0d0d;")
        
        # Initialize Axes (Subplot)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#0d0d0d')
        
        # Initialize Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #262626; color: white; border: none;")

        # Add to Layout
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

    def clear_plot(self):
        """Resets the plot axes and maintains dark theme."""
        self.axes.clear()
        self.axes.set_facecolor('#0d0d0d')
        self.fig.patch.set_facecolor('#0d0d0d')
        self.canvas.draw()

    def get_axes(self):
        return self.axes

    def draw(self):
        try:
            self.canvas.draw()
        except Exception as e:
            print(f"[GUI ERROR] Plotting failed: {e}")

    def style_axes(self, ax, title="", xlabel="", ylabel=""):
        """
        Applies high-contrast styling to the axes (White text on Black bg).
        """
        ax.set_title(title, color='white', fontsize=10, fontweight='bold')
        ax.set_xlabel(xlabel, color='white', fontsize=9)
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        
        # Tick parameters
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Spines (Borders)
        for spine in ax.spines.values():
            spine.set_color('white')
        
        # Grid
        ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)
        
        # Legend styling
        if ax.get_legend():
            plt.setp(ax.get_legend().get_texts(), color='white')
            ax.get_legend().get_frame().set_facecolor('#1a1a1a')
            ax.get_legend().get_frame().set_edgecolor('white')

# =========================================================
# Main GUI Window
# =========================================================
class EEGAnalysisWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SYSTEM::EEG_ANALYSIS_CORE_V4.0 [FULL SUITE]")
        self.setGeometry(50, 50, 1450, 950)

        # -- GLOBAL DATA VARIABLES --
        self.raw_mne = None
        self.fs = 250.0
        self.events = None
        self.raw_data_array = None   # Shape: (Channels, Samples)
        self.filtered_data = None
        self.squared_data = None
        
        # ERP Analysis Variables
        self.avg_left = None
        self.avg_right = None
        self.time_axis_epochs = None
        self.smoothed_left = None
        self.smoothed_right = None
        self.erd_left = None
        self.erd_right = None
        
        # Machine Learning Pipeline
        self.ml_pipeline = ml_analysis.ML_Pipeline()
        self.ml_epochs = None
        self.ml_labels = None
        self.ml_results = {} 

        # Apply Theme
        self.apply_hacker_theme()

        # -- MAIN LAYOUT --
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 1. Header
        self.create_header(main_layout)

        # 2. Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Initialize Tabs
        self.init_tab_load()        # Tab 1: Load & Inspect
        self.init_tab_cwt()         # Tab 2: Time-Frequency
        self.init_tab_filter()      # Tab 3: BPF
        self.init_tab_squaring()    # Tab 4: Power
        self.init_tab_averaging()   # Tab 5: Epoching
        self.init_tab_smoothing()   # Tab 6: MAV
        self.init_tab_erd()         # Tab 7: ERD%
        self.init_tab_csp()         # Tab 8: CSP
        self.init_tab_ml()          # Tab 9: Classification

        # 3. Log Console
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(120)
        self.log_console.setStyleSheet("border: 1px solid #00ff41; color: #00ff41; background-color: #000; font-family: Consolas;")
        main_layout.addWidget(self.log_console)
        
        self.log("SYSTEM V4.0 ONLINE. ALL MODULES LOADED.")

    def apply_hacker_theme(self):
        style = """
        QMainWindow { background-color: #0d0d0d; }
        QWidget { background-color: #0d0d0d; color: #00ff41; font-family: "Consolas"; }
        QTabWidget::pane { border: 1px solid #00ff41; background: #0d0d0d; }
        QTabBar::tab { background: #1a1a1a; color: #00ff41; padding: 10px; border: 1px solid #00ff41; margin-right: 2px; }
        QTabBar::tab:selected { background: #003300; font-weight: bold; border-bottom: 2px solid #00ff41; }
        QPushButton { background-color: #000; border: 1px solid #00ff41; padding: 6px; color: #00ff41; font-weight: bold; }
        QPushButton:hover { background-color: #00ff41; color: #000; }
        QLineEdit, QComboBox, QSpinBox { background-color: #000; border: 1px solid #00ff41; color: #fff; padding: 4px; }
        QComboBox QAbstractItemView { background-color: #000; color: #00ff41; selection-background-color: #00ff41; selection-color: #000; }
        QGroupBox { border: 1px solid #00ff41; margin-top: 15px; font-weight: bold; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; background-color: #0d0d0d; }
        """
        self.setStyleSheet(style)

    def create_header(self, layout):
        h = QHBoxLayout()
        lbl = QLabel(">> EEG BCI PROJECT: MOTOR IMAGERY ANALYSIS")
        lbl.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #00ff41; letter-spacing: 2px;")
        
        btn_clr = QPushButton("[ CLEAR MEMORY ]")
        btn_clr.setFixedWidth(150)
        btn_clr.clicked.connect(self.clear_all_data)
        
        btn_ext = QPushButton("[ SYSTEM EXIT ]")
        btn_ext.setFixedWidth(120)
        btn_ext.setStyleSheet("color: #ff3333; border: 1px solid #ff3333;")
        btn_ext.clicked.connect(self.close)
        
        h.addWidget(lbl)
        h.addStretch()
        h.addWidget(btn_clr)
        h.addWidget(btn_ext)
        layout.addLayout(h)

    def log(self, msg):
        self.log_console.append(f">> {msg}")

    def clear_all_data(self):
        confirm = QMessageBox.question(self, "Confirm Purge", "Clear all data and reset plots?", 
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.No:
            return

        # Reset Variables
        self.raw_mne = None
        self.raw_data_array = None
        self.filtered_data = None
        self.squared_data = None
        self.avg_left = None
        self.avg_right = None
        self.smoothed_left = None
        self.smoothed_right = None
        self.erd_left = None
        self.erd_right = None
        self.ml_epochs = None
        self.ml_labels = None
        self.ml_results = {}
        
        # Clear Plots
        self.plot_raw.clear_plot()
        self.plot_cwt.clear_plot()
        self.plot_filter.clear_plot()
        self.plot_square.clear_plot()
        self.plot_avg.clear_plot()
        self.plot_smooth.clear_plot()
        self.plot_erd.clear_plot()
        self.plot_csp.clear_plot()
        self.plot_ml.clear_plot()
        
        self.lbl_file_status.setText("STATUS: IDLE")
        self.txt_ml_results.clear()
        self.combo_ml_plot.clear()
        
        self.log("SYSTEM MEMORY FLUSHED.")

    # =========================================================
    # TAB 1: INGESTION & EPOCH INSPECTOR (NEW)
    # =========================================================
    def init_tab_load(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Row 1: File Loading
        h_load = QHBoxLayout()
        self.btn_load = QPushButton("[ 1. SELECT GDF FILE ]")
        self.btn_load.clicked.connect(self.load_data)
        self.lbl_file_status = QLabel("STATUS: IDLE")
        h_load.addWidget(self.btn_load)
        h_load.addWidget(self.lbl_file_status)
        layout.addLayout(h_load)
        
        # Row 2: Epoch Settings (Interactive)
        g_insp = QGroupBox("EPOCH INSPECTOR (VISUALIZE TRIALS)")
        h_insp = QHBoxLayout()
        
        self.spin_trial = QSpinBox()
        self.spin_trial.setMinimum(1)
        self.spin_trial.setMaximum(1)
        self.spin_trial.setPrefix("Trial: ")
        self.spin_trial.setFixedWidth(120)
        self.spin_trial.valueChanged.connect(self.update_raw_plot)
        
        self.input_insp_tmin = QLineEdit("-1.5")
        self.input_insp_tmin.setPlaceholderText("Tmin")
        self.input_insp_tmin.setFixedWidth(60)
        
        self.input_insp_tmax = QLineEdit("4.5")
        self.input_insp_tmax.setPlaceholderText("Tmax")
        self.input_insp_tmax.setFixedWidth(60)
        
        btn_update = QPushButton("[ UPDATE VIEW ]")
        btn_update.clicked.connect(self.update_raw_plot)
        
        h_insp.addWidget(self.spin_trial)
        h_insp.addWidget(QLabel("Time Window (s):"))
        h_insp.addWidget(self.input_insp_tmin)
        h_insp.addWidget(QLabel("to"))
        h_insp.addWidget(self.input_insp_tmax)
        h_insp.addWidget(btn_update)
        h_insp.addStretch()
        g_insp.setLayout(h_insp)
        layout.addWidget(g_insp)

        # Plot
        self.plot_raw = PlotWidget(self)
        layout.addWidget(self.plot_raw)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "1. INGESTION & INSPECT")

    def load_data(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open GDF", "", "GDF Files (*.gdf)")
        if fname:
            try:
                self.log(f"UPLOADING: {os.path.basename(fname)}...")
                raw, ev, ev_map, fs = load_data_eeg_mne.load_eeg_data(fname)
                if raw is None: return
                
                self.raw_mne = raw
                self.events = ev
                self.fs = fs
                self.raw_data_array = raw.get_data() * 1e6 # uV
                
                # Identify valid Motor Imagery trials (769/770)
                self.valid_trials = [e for e in ev if e[2] in [769, 770]]
                num_trials = len(self.valid_trials)
                
                self.lbl_file_status.setText(f"ACTIVE: {os.path.basename(fname)}")
                self.spin_trial.setMaximum(num_trials if num_trials > 0 else 1)
                
                self.log(f"UPLOAD SUCCESS. FS: {fs}Hz. VALID TRIALS: {num_trials}")
                
                # Trigger initial plot
                self.update_raw_plot()
                
            except Exception as e:
                self.log(f"LOAD EXCEPTION: {e}")

    def update_raw_plot(self):
        if self.raw_data_array is None or not hasattr(self, 'valid_trials'):
            return
            
        try:
            # 1. Parse Parameters
            trial_idx = self.spin_trial.value() - 1 # Convert 1-based to 0-based
            if trial_idx >= len(self.valid_trials): return
            
            tmin = float(self.input_insp_tmin.text())
            tmax = float(self.input_insp_tmax.text())
            
            # 2. Extract Event Info
            event = self.valid_trials[trial_idx]
            cue_sample = event[0]
            label_code = event[2]
            label_str = "LEFT (769)" if label_code == 769 else "RIGHT (770)"
            
            # 3. Define Window
            start = cue_sample + int(tmin * self.fs)
            end = cue_sample + int(tmax * self.fs)
            
            # Boundary Safety
            if start < 0: start = 0
            if end > self.raw_data_array.shape[1]: end = self.raw_data_array.shape[1]
            
            # 4. Slice Data (3 Channels: C3, Cz, C4)
            # Assuming channel indices 0, 1, 2 based on our loader logic
            data_segment = self.raw_data_array[0:3, start:end]
            time_axis = np.linspace(tmin, tmax, data_segment.shape[1])
            
            # 5. Plotting
            self.plot_raw.clear_plot()
            ax = self.plot_raw.get_axes()
            
            # Overlay 3 Channels
            ax.plot(time_axis, data_segment[0, :], color='#00ff41', linewidth=1.5, label='C3 (Left)')
            ax.plot(time_axis, data_segment[1, :], color='white', linewidth=0.8, alpha=0.6, label='Cz (Center)')
            ax.plot(time_axis, data_segment[2, :], color='#ff0055', linewidth=1.5, label='C4 (Right)')
            
            # Mark Cue Onset
            ax.axvline(0, color='yellow', linestyle='--', linewidth=1.5, label='Cue Onset')
            
            # Styling
            self.plot_raw.style_axes(ax, 
                                     title=f"RAW SIGNAL INSPECTOR - TRIAL #{trial_idx+1} [{label_str}]", 
                                     xlabel="Time relative to Cue [s]", 
                                     ylabel="Amplitude [uV]")
            ax.legend(loc='upper right')
            self.plot_raw.draw()
            
        except Exception as e:
            self.log(f"INSPECTOR ERROR: {e}")

    # =========================================================
    # TAB 2: TIME-FREQUENCY (CWT)
    # =========================================================
    def init_tab_cwt(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        g = QGroupBox("CWT CONFIGURATION")
        h = QHBoxLayout()
        self.combo_wavelet = QComboBox()
        self.combo_wavelet.addItems(["Morlet (Complex)", "Mexican Hat (Real)"])
        
        self.input_fmin = QLineEdit("4")
        self.input_fmax = QLineEdit("40")
        
        h.addWidget(QLabel("Wavelet:"))
        h.addWidget(self.combo_wavelet)
        h.addWidget(QLabel("F_min [Hz]:"))
        h.addWidget(self.input_fmin)
        h.addWidget(QLabel("F_max [Hz]:"))
        h.addWidget(self.input_fmax)
        
        btn = QPushButton("[ ANALYZE SPECTRUM ]")
        btn.clicked.connect(self.run_cwt_analysis)
        h.addWidget(btn)
        g.setLayout(h)
        layout.addWidget(g)
        
        self.plot_cwt = PlotWidget(self)
        layout.addWidget(self.plot_cwt)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "2. TIME-FREQ ANALYSIS")

    def run_cwt_analysis(self):
        if self.raw_data_array is None:
            self.log("ERROR: NO DATA LOADED.")
            return

        sel = self.combo_wavelet.currentText()
        w_type = 'mexican_hat' if "Mexican" in sel else 'morlet'
        
        try:
            fmin = float(self.input_fmin.text())
            fmax = float(self.input_fmax.text())
            
            self.log(f"RUNNING CWT ({w_type.upper()})...")
            
            if self.events is not None:
                ev_right = [e for e in self.events if e[2] == 770]
                if ev_right:
                    idx = ev_right[0][0]
                    start = max(0, idx - int(1.0*self.fs))
                    end = min(self.raw_data_array.shape[1], idx + int(4.0*self.fs))
                    segment = self.raw_data_array[0, start:end]
                    
                    tfr, freqs = CWT.run_cwt(segment, self.fs, fmin, fmax, 0.5, w_type)
                    
                    self.plot_cwt.clear_plot()
                    ax = self.plot_cwt.get_axes()
                    extent = [-1.0, 4.0, fmin, fmax]
                    
                    im = ax.imshow(tfr, extent=extent, aspect='auto', origin='lower', cmap='jet')
                    
                    cb = self.plot_cwt.fig.colorbar(im, ax=ax)
                    cb.set_label('Power Magnitude', color='white')
                    cb.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
                    
                    self.plot_cwt.style_axes(ax, title=f"SINGLE TRIAL SPECTROGRAM (C3, RIGHT HAND)", 
                                             xlabel="Time [s]", ylabel="Frequency [Hz]")
                    self.plot_cwt.draw()
                    self.log("SPECTROGRAM GENERATED.")
                else:
                    self.log("WARN: NO CLASS 770 EVENTS FOUND.")
            else:
                self.log("ERROR: EVENTS NOT PARSED.")
        except Exception as e:
            self.log(f"CWT ERROR: {e}")

    # =========================================================
    # TAB 3: FILTERING
    # =========================================================
    def init_tab_filter(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        g = QGroupBox("BPF PARAMETERS")
        h = QHBoxLayout()
        self.input_lowcut = QLineEdit("0.5")
        self.input_highcut = QLineEdit("30.0")
        self.input_order = QComboBox()
        self.input_order.addItems(["2 (Standard)", "4 (Steep Cascade)"])
        
        h.addWidget(QLabel("Low [Hz]:"))
        h.addWidget(self.input_lowcut)
        h.addWidget(QLabel("High [Hz]:"))
        h.addWidget(self.input_highcut)
        h.addWidget(QLabel("Order:"))
        h.addWidget(self.input_order)
        
        btn = QPushButton("[ APPLY FILTER ]")
        btn.clicked.connect(self.apply_filter)
        h.addWidget(btn)
        g.setLayout(h)
        layout.addWidget(g)
        
        self.plot_filter = PlotWidget(self)
        layout.addWidget(self.plot_filter)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "3. FILTERING")

    def apply_filter(self):
        if self.raw_data_array is None:
            self.log("ERROR: NO DATA.")
            return
        try:
            low = float(self.input_lowcut.text())
            high = float(self.input_highcut.text())
            order = 2 if "2" in self.input_order.currentText() else 4
            
            self.log(f"FILTERING: {low}-{high} Hz (Order {order})...")
            
            self.filtered_data = np.zeros_like(self.raw_data_array)
            for i in range(self.raw_data_array.shape[0]):
                self.filtered_data[i, :] = filtering_BPF_EEG.run_filter(
                    self.raw_data_array[i, :], self.fs, low, high, order)
            
            self.plot_filter.clear_plot()
            ax = self.plot_filter.get_axes()
            t = np.linspace(0, 5, int(5*self.fs))
            
            ax.plot(t, self.raw_data_array[0, :len(t)], color='gray', alpha=0.5, label='Raw Signal')
            ax.plot(t, self.filtered_data[0, :len(t)], color='cyan', linewidth=1.2, label='Filtered Signal')
            
            self.plot_filter.style_axes(ax, title="FILTER RESULT (CHANNEL C3, FIRST 5s)", 
                                        xlabel="Time [s]", ylabel="Amplitude [uV]")
            ax.legend(loc='upper right')
            self.plot_filter.draw()
            self.log("FILTERING COMPLETE.")
        except Exception as e:
            self.log(f"FILTER ERROR: {e}")

    # =========================================================
    # TAB 4: SQUARING
    # =========================================================
    def init_tab_squaring(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        btn = QPushButton("[ EXECUTE SIGNAL SQUARING (x^2) ]")
        btn.clicked.connect(self.apply_squaring)
        layout.addWidget(btn)
        
        self.plot_square = PlotWidget(self)
        layout.addWidget(self.plot_square)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "4. SQUARING")

    def apply_squaring(self):
        if self.filtered_data is None:
            self.log("ERROR: DATA NOT FILTERED.")
            return
        
        self.squared_data = squaring_EEG.square_signal(self.filtered_data)
        
        self.plot_square.clear_plot()
        ax = self.plot_square.get_axes()
        t = np.linspace(0, 5, int(5*self.fs))
        
        ax.plot(t, self.squared_data[0, :len(t)], color='magenta', linewidth=1, label='Instantaneous Power')
        
        self.plot_square.style_axes(ax, title="INSTANTANEOUS POWER (C3, FIRST 5s)", 
                                    xlabel="Time [s]", ylabel=r"Power [$\mu V^2$]")
        ax.legend(loc='upper right')
        self.plot_square.draw()
        self.log("SQUARING COMPLETE.")

    # =========================================================
    # TAB 5: AVERAGING
    # =========================================================
    def init_tab_averaging(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        g = QGroupBox("EPOCH PARAMETERS")
        h = QHBoxLayout()
        self.input_tmin = QLineEdit("-1.5")
        self.input_tmax = QLineEdit("4.5")
        h.addWidget(QLabel("T_start [s]:"))
        h.addWidget(self.input_tmin)
        h.addWidget(QLabel("T_end [s]:"))
        h.addWidget(self.input_tmax)
        
        btn = QPushButton("[ COMPUTE GRAND AVERAGE ]")
        btn.clicked.connect(self.apply_averaging)
        h.addWidget(btn)
        g.setLayout(h)
        layout.addWidget(g)
        
        self.plot_avg = PlotWidget(self)
        layout.addWidget(self.plot_avg)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "5. AVERAGING")

    def apply_averaging(self):
        if self.squared_data is None:
            self.log("ERROR: DATA NOT SQUARED.")
            return
        try:
            tmin = float(self.input_tmin.text())
            tmax = float(self.input_tmax.text())
            self.log("AVERAGING EPOCHS BY CLASS...")
            
            self.avg_left, self.avg_right, self.time_axis_epochs = \
                average_all_EEG_trials.extract_and_average_epochs(
                    self.squared_data, self.events, self.fs, tmin, tmax)
            
            self.plot_avg.clear_plot()
            ax = self.plot_avg.get_axes()
            
            ax.plot(self.time_axis_epochs, self.avg_right[0, :], color='red', linewidth=1.5, label='Class 2: Right Hand (C3)')
            ax.plot(self.time_axis_epochs, self.avg_left[0, :], color='cyan', linestyle='--', linewidth=1.0, label='Class 1: Left Hand (C3)')
            ax.axvline(0, color='white', linestyle=':', label='Cue Onset')
            
            self.plot_avg.style_axes(ax, title="SYNCHRONOUS AVERAGE (CHANNEL C3)", 
                                     xlabel="Time relative to Cue [s]", ylabel=r"Mean Power [$\mu V^2$]")
            ax.legend(loc='upper right')
            self.plot_avg.draw()
            self.log("AVERAGING DONE.")
        except Exception as e:
            self.log(f"AVG ERROR: {e}")

    # =========================================================
    # TAB 6: SMOOTHING
    # =========================================================
    def init_tab_smoothing(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        h = QHBoxLayout()
        self.input_window = QLineEdit("0.75")
        h.addWidget(QLabel("MAV Window Size [s]:"))
        h.addWidget(self.input_window)
        btn = QPushButton("[ APPLY SMOOTHING ]")
        btn.clicked.connect(self.apply_smoothing)
        h.addWidget(btn)
        layout.addLayout(h)
        
        self.plot_smooth = PlotWidget(self)
        layout.addWidget(self.plot_smooth)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "6. SMOOTHING")

    def apply_smoothing(self):
        if self.avg_left is None:
            self.log("ERROR: NO AVERAGED DATA.")
            return
        try:
            win = float(self.input_window.text())
            self.log(f"SMOOTHING (Window={win}s)...")
            
            self.smoothed_left = moving_average_EEG.apply_moving_average(self.avg_left, self.fs, win)
            self.smoothed_right = moving_average_EEG.apply_moving_average(self.avg_right, self.fs, win)
            
            self.plot_smooth.clear_plot()
            ax = self.plot_smooth.get_axes()
            
            ax.plot(self.time_axis_epochs, self.smoothed_right[0, :], color='red', linewidth=2.0, label='Right Hand (C3)')
            ax.plot(self.time_axis_epochs, self.smoothed_left[0, :], color='cyan', linestyle='--', linewidth=1.5, label='Left Hand (C3)')
            ax.axvline(0, color='white', linestyle=':')
            
            self.plot_smooth.style_axes(ax, title="SMOOTHED POWER ENVELOPE (C3)", 
                                        xlabel="Time [s]", ylabel=r"Power [$\mu V^2$]")
            ax.legend(loc='upper right')
            self.plot_smooth.draw()
            self.log("SMOOTHING COMPLETE.")
        except Exception as e:
            self.log(f"SMOOTH ERROR: {e}")

    # =========================================================
    # TAB 7: ERD/ERS
    # =========================================================
    def init_tab_erd(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        g = QGroupBox("BASELINE REFERENCE")
        h = QHBoxLayout()
        self.input_ref_start = QLineEdit("-1.5")
        self.input_ref_end = QLineEdit("-0.5")
        h.addWidget(QLabel("Start [s]:"))
        h.addWidget(self.input_ref_start)
        h.addWidget(QLabel("End [s]:"))
        h.addWidget(self.input_ref_end)
        
        btn = QPushButton("[ CALCULATE ERD/ERS ]")
        btn.clicked.connect(self.calc_erd)
        h.addWidget(btn)
        g.setLayout(h)
        layout.addWidget(g)
        
        self.plot_erd = PlotWidget(self)
        layout.addWidget(self.plot_erd)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "7. ERD RESULT")

    def calc_erd(self):
        if self.smoothed_left is None:
            self.log("ERROR: NO SMOOTHED DATA.")
            return
        try:
            t_start = float(self.input_ref_start.text())
            t_end = float(self.input_ref_end.text())
            tmin, tmax = self.time_axis_epochs[0], self.time_axis_epochs[-1]
            
            self.log("COMPUTING ERD/ERS PERCENTAGES...")
            
            self.erd_left, _ = percentage_ERD_ERS.calculate_erd_percent(
                self.smoothed_left, self.fs, tmin, tmax, (t_start, t_end))
            self.erd_right, _ = percentage_ERD_ERS.calculate_erd_percent(
                self.smoothed_right, self.fs, tmin, tmax, (t_start, t_end))
            
            self.plot_erd.clear_plot()
            ax = self.plot_erd.get_axes()
            
            # C3 Contralateral ERD Expectation: Right Hand (Red) should drop below 0
            ax.plot(self.time_axis_epochs, self.erd_right[0, :], color='red', linewidth=2.0, label='Right Hand (ERD)')
            ax.plot(self.time_axis_epochs, self.erd_left[0, :], color='cyan', linestyle='--', linewidth=1.5, label='Left Hand (Ref)')
            ax.axhline(0, color='white', linewidth=1)
            ax.axvline(0, color='white', linestyle=':')
            
            self.plot_erd.style_axes(ax, title="ERD/ERS PERCENTAGE (CHANNEL C3)", 
                                     xlabel="Time relative to Cue [s]", ylabel="Power Change [%]")
            ax.legend(loc='upper right')
            self.plot_erd.draw()
            self.log("ERD CALCULATION DONE.")
        except Exception as e:
            self.log(f"ERD ERROR: {e}")

    # =========================================================
    # TAB 8: CSP PATTERNS
    # =========================================================
    def init_tab_csp(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        btn_train_csp = QPushButton("[ TRAIN CSP (SPATIAL FILTERS) ]")
        btn_train_csp.clicked.connect(self.run_csp_training)
        layout.addWidget(btn_train_csp)
        
        self.plot_csp = PlotWidget(self)
        layout.addWidget(self.plot_csp)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "8. CSP PATTERNS")

    def run_csp_training(self):
        if self.filtered_data is None:
            QMessageBox.warning(self, "Pipeline Error", "Please run Filtering (Tab 3) first!")
            return
            
        try:
            self.log("PREPARING EPOCHS FOR CSP...")
            # Epoch window typically 0.5s to 2.5s for Motor Imagery
            epochs, labels = self.ml_pipeline.prepare_data(
                self.filtered_data, self.events, self.fs, tmin=0.5, tmax=2.5
            )
            
            if len(labels) == 0:
                self.log("ERROR: NO VALID TRIALS FOUND.")
                return
                
            self.ml_epochs = epochs
            self.ml_labels = labels
            self.log(f"EPOCHS EXTRACTED: {epochs.shape}")
            
            self.log("TRAINING CSP FROM SCRATCH...")
            self.ml_pipeline.csp = csp_scratch.CSP_Scratch(n_components=2)
            self.ml_pipeline.csp.fit(epochs, labels)
            
            patterns = self.ml_pipeline.csp.patterns_
            
            self.plot_csp.clear_plot()
            ax = self.plot_csp.get_axes()
            
            ch_names = ['C3', 'Cz', 'C4']
            x = np.arange(3)
            width = 0.35
            
            # Visualize the pattern weights
            ax.bar(x - width/2, patterns[0, :], width, label='CSP Comp 1 (LH)', color='cyan')
            ax.bar(x + width/2, patterns[1, :], width, label='CSP Comp 2 (RH)', color='magenta')
            
            ax.set_xticks(x)
            ax.set_xticklabels(ch_names)
            
            self.plot_csp.style_axes(ax, title="COMMON SPATIAL PATTERNS (WEIGHTS)", 
                                     xlabel="Channels", ylabel="Pattern Weights")
            ax.legend()
            self.plot_csp.draw()
            self.log("CSP FILTERS COMPUTED.")
            
        except Exception as e:
            self.log(f"CSP ERROR: {e}")

    # =========================================================
    # TAB 9: MACHINE LEARNING
    # =========================================================
    def init_tab_ml(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Controls Row
        h = QHBoxLayout()
        
        btn_compare = QPushButton("[ TRAIN & COMPARE ALL 8 MODELS ]")
        btn_compare.clicked.connect(self.run_ml_comparison)
        
        self.combo_ml_plot = QComboBox()
        self.combo_ml_plot.setFixedWidth(250)
        self.combo_ml_plot.addItem("Select Model to Plot CM...")
        self.combo_ml_plot.currentIndexChanged.connect(self.plot_selected_confusion_matrix)
        
        h.addWidget(btn_compare)
        h.addWidget(self.combo_ml_plot)
        layout.addLayout(h)
        
        # Content Area
        content_layout = QHBoxLayout()
        
        # Left: Results Text
        self.txt_ml_results = QTextEdit()
        self.txt_ml_results.setReadOnly(True)
        self.txt_ml_results.setFixedWidth(400)
        self.txt_ml_results.setStyleSheet("border: 1px solid #00ff41; background: #000; color: #00ff41; font-family: Consolas;")
        content_layout.addWidget(self.txt_ml_results)
        
        # Right: Confusion Matrix Plot
        self.plot_ml = PlotWidget(self)
        content_layout.addWidget(self.plot_ml)
        
        layout.addLayout(content_layout)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "9. CLASSIFICATION")

    def run_ml_comparison(self):
        if self.ml_epochs is None:
            QMessageBox.warning(self, "Pipeline Error", "Please run CSP Training (Tab 8) first to extract features!")
            return
            
        try:
            self.log("STARTING FULL MODEL COMPARISON...")
            self.txt_ml_results.setText(">> TRAINING IN PROGRESS...\n")
            QApplication.processEvents() # Update GUI
            
            # Run Pipeline
            self.ml_results = self.ml_pipeline.run_full_comparison(self.ml_epochs, self.ml_labels, test_size=0.25)
            
            # Display Results
            report = ">> MODEL ACCURACY REPORT:\n" + "="*30 + "\n"
            sorted_results = sorted(self.ml_results.items(), key=lambda x: x[1], reverse=True)
            
            self.combo_ml_plot.clear()
            self.combo_ml_plot.addItem("Select Model to Plot CM...")
            
            for name, acc in sorted_results:
                report += f"{name:<20}: {acc*100:.2f}%\n"
                self.combo_ml_plot.addItem(name)
                
            report += "="*30 + "\n"
            report += f"BEST MODEL: {sorted_results[0][0]}"
            
            self.txt_ml_results.setText(report)
            self.log("ML COMPARISON COMPLETE.")
            
            # Automatically plot best model
            self.combo_ml_plot.setCurrentText(sorted_results[0][0])
            
        except Exception as e:
            self.log(f"ML ERROR: {e}")
            self.txt_ml_results.setText(f"ERROR: {e}")

    def plot_selected_confusion_matrix(self):
        model_name = self.combo_ml_plot.currentText()
        if "Select" in model_name or model_name not in self.ml_results:
            return
            
        try:
            y_true, y_pred = self.ml_pipeline.get_prediction(model_name)
            if y_true is None: return
            
            # Calculate Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            self.plot_ml.clear_plot()
            ax = self.plot_ml.get_axes()
            
            im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
            
            # Colorbar
            cb = self.plot_ml.fig.colorbar(im, ax=ax)
            cb.ax.yaxis.set_tick_params(color='white')
            
            classes = ['Left', 'Right']
            ax.set_xticks(np.arange(2))
            ax.set_yticks(np.arange(2))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            
            # Annotate
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            self.plot_ml.style_axes(ax, title=f"CONFUSION MATRIX: {model_name}", 
                                    xlabel="Predicted Label", ylabel="True Label")
            self.plot_ml.draw()
            
        except Exception as e:
            self.log(f"PLOTTING ERROR: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGAnalysisWindow()
    window.show()
    sys.exit(app.exec())