"""
GUI.py

Purpose:
    - Main User Interface for the EEG Motor Imagery Analysis Project (Version 8.0).
    - This script orchestrates the entire BCI pipeline, from data loading to Machine Learning.
    - It is designed to be robust, user-friendly, and visually clear using a High-Contrast Dark Theme.

Pipeline Overview:
    1.  **Data Ingestion:** Load GDF files, inspect raw signals (C3, Cz, C4), and visualize trials.
    2.  **Time-Frequency Analysis:** Continuous Wavelet Transform (CWT) to visualize ERD/ERS.
    3.  **Preprocessing:** Bandpass Filtering (8-30Hz), Signal Squaring, and Synchronous Averaging.
    4.  **Feature Extraction:** Extracting Temporal Features (Mean, Var, Skew, Kurt) and Spatial Features (CSP).
    5.  **Machine Learning:** Training and comparing 8 classifiers (SVM, RF, MLP, etc.) with detailed metrics.

Dependencies:
    - PyQt6: For the Graphical User Interface.
    - Matplotlib: For generating embedded plots.
    - NumPy: For numerical array manipulation.
    - MNE: For loading EEG datasets.
    - Custom Modules: load_data_eeg_mne, CWT, filtering_BPF_EEG, squaring_EEG, 
      average_all_EEG_trials, moving_average_EEG, percentage_ERD_ERS, csp_scratch, ml_analysis.
"""

import sys
import os
import numpy as np
import mne

# PyQt6 Components
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTabWidget, QFileDialog, QLineEdit, 
    QComboBox, QMessageBox, QGroupBox, QTextEdit, QSpinBox, 
    QSplitter, QScrollArea, QTableWidget, QTableWidgetItem, 
    QHeaderView, QCheckBox, QSizePolicy, QFrame
)
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon
from PyQt6.QtCore import Qt

# Matplotlib Components
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Custom Project Modules
import load_data_eeg_mne
import CWT
import filtering_BPF_EEG
import squaring_EEG
import average_all_EEG_trials
import moving_average_EEG
import percentage_ERD_ERS
import csp_scratch
import ml_analysis

# ==================================================================================
# CLASS: PlotWidget
# Purpose: A wrapper for a Matplotlib Figure that includes a Navigation Toolbar.
#          It is styled for the Dark "Hacker" theme.
# ==================================================================================
class PlotWidget(QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100, min_height=None):
        super(PlotWidget, self).__init__(parent)
        
        # Define Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(2)

        # Create Figure with Dark Background
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#0d0d0d') 
        
        # Create Canvas
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #0d0d0d;")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Set Minimum Height if requested (Crucial for scrollable layouts)
        if min_height:
            self.canvas.setMinimumHeight(min_height)
        
        # Initialize Default Axes
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#0d0d0d')
        
        # Initialize Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #262626; color: white; border: 1px solid #333;")

        # Add widgets to layout
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

    def clear_plot(self):
        """
        Clears the entire figure content to prepare for new plots.
        """
        self.fig.clf() 
        self.axes = self.fig.add_subplot(111) # Re-add a default subplot
        self.axes.set_facecolor('#0d0d0d')
        self.fig.patch.set_facecolor('#0d0d0d')
        self.canvas.draw()

    def get_axes(self):
        """Returns the current active axes."""
        return self.axes

    def get_figure(self):
        """Returns the figure object for advanced subplotting."""
        return self.fig

    def draw(self):
        """Redraws the canvas."""
        try:
            self.canvas.draw()
        except Exception as e:
            print(f"[GUI ERROR] Plotting failed: {e}")

    def style_axes(self, ax, title="", xlabel="", ylabel=""):
        """
        Applies consistent High-Contrast styling to any given axes object.
        """
        ax.set_facecolor('#0d0d0d')
        ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel(xlabel, color='white', fontsize=9)
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        
        # Set Tick Colors
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        # Set Spine (Border) Colors
        for spine in ax.spines.values():
            spine.set_color('white')
        
        # Set Grid Style
        ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)
        
        # Set Legend Style (if present)
        if ax.get_legend():
            plt.setp(ax.get_legend().get_texts(), color='white')
            ax.get_legend().get_frame().set_facecolor('#1a1a1a')
            ax.get_legend().get_frame().set_edgecolor('white')

# ==================================================================================
# CLASS: ScrollableChannelLayout
# Purpose: Holds 3 PlotWidgets (C3, Cz, C4) inside a ScrollArea.
#          This ensures graphs are large enough to be readable and do not overlap.
# ==================================================================================
class ScrollableChannelLayout(QWidget):
    def __init__(self, parent=None):
        super(ScrollableChannelLayout, self).__init__(parent)
        
        # Main Layout for this widget
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Create the Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True) # Allow inner widget to expand
        self.scroll_area.setStyleSheet("border: none; background-color: #0d0d0d;")
        
        # 2. Create the Container Widget that holds the plots
        self.container_widget = QWidget()
        self.container_layout = QVBoxLayout(self.container_widget)
        self.container_layout.setSpacing(15) # Add space between plots
        
        # 3. Initialize 3 Plot Widgets (One for each Channel)
        # We set min_height=350 to ensure they are readable
        self.plot_c3 = PlotWidget(min_height=350)
        self.plot_cz = PlotWidget(min_height=350)
        self.plot_c4 = PlotWidget(min_height=350)
        
        # Add labels to indicate which plot is which (Optional, but good for UI)
        self.lbl_c3 = QLabel("CHANNEL C3 (LEFT MOTOR CORTEX)")
        self.lbl_c3.setStyleSheet("color: #00ff41; font-weight: bold; padding-left: 5px;")
        
        self.lbl_cz = QLabel("CHANNEL Cz (VERTEX)")
        self.lbl_cz.setStyleSheet("color: cyan; font-weight: bold; padding-left: 5px;")
        
        self.lbl_c4 = QLabel("CHANNEL C4 (RIGHT MOTOR CORTEX)")
        self.lbl_c4.setStyleSheet("color: #ff0055; font-weight: bold; padding-left: 5px;")
        
        # Add to Container Layout
        self.container_layout.addWidget(self.lbl_c3)
        self.container_layout.addWidget(self.plot_c3)
        self.container_layout.addWidget(self.make_separator()) # Visual line
        
        self.container_layout.addWidget(self.lbl_cz)
        self.container_layout.addWidget(self.plot_cz)
        self.container_layout.addWidget(self.make_separator())
        
        self.container_layout.addWidget(self.lbl_c4)
        self.container_layout.addWidget(self.plot_c4)
        
        # 4. Set the Container as the Widget for Scroll Area
        self.scroll_area.setWidget(self.container_widget)
        
        # 5. Add Scroll Area to Main Layout
        self.main_layout.addWidget(self.scroll_area)

    def make_separator(self):
        """Creates a horizontal line separator for visual clarity."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #333;")
        return line

    def get_plots(self):
        """Returns the list of plots for easy iteration."""
        return [self.plot_c3, self.plot_cz, self.plot_c4]

    def clear_all(self):
        """Clears all 3 plots."""
        self.plot_c3.clear_plot()
        self.plot_cz.clear_plot()
        self.plot_c4.clear_plot()

# ==================================================================================
# CLASS: EEGAnalysisWindow
# Purpose: The Main Application Window.
# ==================================================================================
class EEGAnalysisWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window Configuration
        self.setWindowTitle("SYSTEM::EEG_ANALYSIS_CORE_V8.0 [FINAL PROJECT EDITION]")
        self.setGeometry(50, 50, 1600, 1000)

        # -- GLOBAL DATA STATE VARIABLES --
        self.raw_mne = None          # MNE Raw Object
        self.fs = 250.0              # Sampling Frequency
        self.events = None           # Event Matrix
        self.event_id_map = None     # Event ID Dictionary
        self.session_info = None     # Session Metadata
        
        # Raw Data Arrays: Expected Shape (3, n_samples) -> [C3, Cz, C4]
        self.raw_data_array = None   
        
        # Processing Intermediate Data
        self.filtered_data = None
        self.squared_data = None
        
        # ERP Analysis Variables (Averaged Trials)
        self.avg_left = None
        self.avg_right = None
        self.time_axis_epochs = None
        
        # Smoothing Variables
        self.smoothed_left = None
        self.smoothed_right = None
        
        # ERD Variables
        self.erd_left = None
        self.erd_right = None
        
        # Machine Learning Pipeline State
        self.ml_pipeline = ml_analysis.ML_Pipeline()
        self.ml_epochs = None
        self.ml_labels = None
        self.ml_metrics = {} 

        # Apply Global Theme
        self.apply_hacker_theme()

        # -- MAIN LAYOUT CONSTRUCTION --
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 1. Header Section
        self.create_header(main_layout)

        # 2. Main Tab Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # 3. Initialize All Analysis Tabs
        self.init_tab_load()        # Tab 1: Load Data & Inspect Raw Signals
        self.init_tab_cwt()         # Tab 2: Time-Frequency Analysis (3 Channels)
        self.init_tab_filter()      # Tab 3: Bandpass Filtering (3 Channels)
        self.init_tab_squaring()    # Tab 4: Signal Power (3 Channels)
        self.init_tab_averaging()   # Tab 5: Synchronous Averaging (3 Channels)
        self.init_tab_smoothing()   # Tab 6: Moving Average (3 Channels)
        self.init_tab_erd()         # Tab 7: ERD/ERS Percentage (3 Channels)
        self.init_tab_csp()         # Tab 8: Feature Extraction (CSP + Temporal)
        self.init_tab_ml()          # Tab 9: Machine Learning Classification

        # 4. Log Console at Bottom
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(150)
        self.log_console.setStyleSheet("border: 1px solid #00ff41; color: #00ff41; background-color: #000; font-family: Consolas;")
        main_layout.addWidget(self.log_console)
        
        self.log("SYSTEM V8.0 ONLINE. MULTI-CHANNEL SCROLLABLE VIEW READY.")

    def apply_hacker_theme(self):
        """
        Applies a high-contrast Green-on-Black theme using Qt Style Sheets (QSS).
        """
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
        QGroupBox { border: 1px solid #00ff41; margin-top: 10px; font-weight: bold; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px; background-color: #0d0d0d; }
        QTableWidget { gridline-color: #00ff41; color: #fff; }
        QHeaderView::section { background-color: #1a1a1a; color: #00ff41; border: 1px solid #00ff41; }
        QScrollArea { border: none; }
        QCheckBox { color: #00ff41; }
        """
        self.setStyleSheet(style)

    def create_header(self, layout):
        """Creates the top header with Title and Global Controls."""
        h_layout = QHBoxLayout()
        
        lbl_title = QLabel(">> EEG BCI PROJECT: 3-CHANNEL (C3, Cz, C4) MOTOR IMAGERY ANALYSIS")
        lbl_title.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        lbl_title.setStyleSheet("color: #00ff41; letter-spacing: 2px;")
        
        btn_clear = QPushButton("[ CLEAR MEMORY ]")
        btn_clear.setFixedWidth(150)
        btn_clear.clicked.connect(self.clear_all_data)
        
        btn_exit = QPushButton("[ SYSTEM EXIT ]")
        btn_exit.setFixedWidth(120)
        btn_ext_style = "color: #ff3333; border: 1px solid #ff3333;"
        btn_exit.setStyleSheet(btn_ext_style)
        btn_exit.clicked.connect(self.close)
        
        h_layout.addWidget(lbl_title)
        h_layout.addStretch()
        h_layout.addWidget(btn_clear)
        h_layout.addWidget(btn_exit)
        
        layout.addLayout(h_layout)

    def log(self, msg):
        """Appends a message to the on-screen console."""
        self.log_console.append(f">> {msg}")

    def clear_all_data(self):
        """Resets the entire application state."""
        confirm = QMessageBox.question(self, "Confirm Purge", "Clear all data and reset plots?", 
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.No: return

        # Clear Variables
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
        self.ml_metrics = {}
        self.session_info = None
        
        # Clear Plots using their specific clear methods
        self.plot_raw.clear_plot()
        self.scroll_cwt.clear_all()
        self.scroll_filter.clear_all()
        self.scroll_square.clear_all()
        self.scroll_avg.clear_all()
        self.scroll_smooth.clear_all()
        self.scroll_erd.clear_all()
        self.plot_csp_scatter.clear_plot()
        self.plot_temp_box.clear_plot()
        
        self.lbl_file_status.setText("STATUS: IDLE")
        self.txt_ml_results.clear()
        self.table_ml_details.setRowCount(0)
        
        self.log("SYSTEM MEMORY FLUSHED.")

    # ==================================================================================
    # TAB 1: DATA INGESTION & EPOCH INSPECTOR
    # ==================================================================================
    def init_tab_load(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 1. File Selection Row
        h_load = QHBoxLayout()
        self.btn_load = QPushButton("[ 1. SELECT GDF FILE ]")
        self.btn_load.clicked.connect(self.load_data)
        self.lbl_file_status = QLabel("STATUS: IDLE")
        self.lbl_session_type = QLabel("SESSION: UNKNOWN")
        h_load.addWidget(self.btn_load)
        h_load.addWidget(self.lbl_file_status)
        h_load.addStretch()
        h_load.addWidget(self.lbl_session_type)
        layout.addLayout(h_load)
        
        # 2. View Options
        h_view = QHBoxLayout()
        self.chk_show_all = QCheckBox("Show All Channels (Including EOG)")
        self.chk_show_all.toggled.connect(self.update_raw_plot)
        h_view.addWidget(self.chk_show_all)
        h_view.addStretch()
        layout.addLayout(h_view)
        
        # 3. Epoch Inspector Controls
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

        # 4. Raw Plot Widget (Single large plot for overview)
        self.plot_raw = PlotWidget(min_height=400)
        layout.addWidget(self.plot_raw)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "1. INGESTION")

    def load_data(self):
        """Handles the GDF loading process."""
        fname, _ = QFileDialog.getOpenFileName(self, "Open GDF", "", "GDF Files (*.gdf)")
        if fname:
            try:
                self.log(f"UPLOADING: {os.path.basename(fname)}...")
                
                # Load Full Data
                raw, ev, ev_map, fs, sess_info = load_data_eeg_mne.load_eeg_data(fname)
                if raw is None: return
                
                # Store Globally
                self.raw_mne = raw
                self.events = ev
                self.fs = fs
                self.session_info = sess_info
                
                # Filter Valid Motor Imagery Trials (769=Left, 770=Right)
                self.valid_trials = [e for e in ev if e[2] in [769, 770]]
                num_trials = len(self.valid_trials)
                
                # Update UI
                self.lbl_file_status.setText(f"ACTIVE: {os.path.basename(fname)}")
                self.lbl_session_type.setText(f"TYPE: {sess_info['type'].upper()}")
                self.spin_trial.setMaximum(num_trials if num_trials > 0 else 1)
                self.log(f"UPLOAD SUCCESS. FS: {fs}Hz. VALID TRIALS: {num_trials}")
                
                # Extract Specific Channels (C3, Cz, C4)
                # Important: We perform this extraction here so subsequent tabs use only these 3.
                picks = mne.pick_channels(raw.ch_names, include=['C3', 'Cz', 'C4'])
                if len(picks) < 3:
                    self.log("WARN: Could not find all motor channels (C3, Cz, C4).")
                
                # Store as 2D Array (Channels x Samples)
                self.raw_data_array = raw.get_data(picks=picks) * 1e6 # Convert Volts to uV
                
                # Initial Plot
                self.update_raw_plot()
                
            except Exception as e:
                self.log(f"LOAD EXCEPTION: {e}")

    def update_raw_plot(self):
        """Updates the Inspector Plot based on selected trial and settings."""
        if self.raw_mne is None:
            return
            
        try:
            # Parse GUI Inputs
            trial_idx = self.spin_trial.value() - 1 
            tmin = float(self.input_insp_tmin.text())
            tmax = float(self.input_insp_tmax.text())
            show_all = self.chk_show_all.isChecked()
            
            cue_sample = 0
            label_str = "N/A"
            
            # Get Event Info
            if hasattr(self, 'valid_trials') and len(self.valid_trials) > 0 and trial_idx < len(self.valid_trials):
                event = self.valid_trials[trial_idx]
                cue_sample = event[0]
                label_code = event[2]
                label_str = "LEFT (769)" if label_code == 769 else "RIGHT (770)"
            
            # Calculate sample range
            start = max(0, cue_sample + int(tmin * self.fs))
            end = min(self.raw_mne.n_times, cue_sample + int(tmax * self.fs))
            
            # Data Selection Logic
            if show_all:
                # Get all channels from MNE object
                data_segment = self.raw_mne.get_data(start=start, stop=end) * 1e6
                ch_names = self.raw_mne.ch_names
            else:
                # Use cached motor channels
                if self.raw_data_array is None: return
                data_segment = self.raw_data_array[:, start:end]
                ch_names = ['C3', 'Cz', 'C4']

            # Time Vector (Relative)
            times = np.linspace(tmin, tmax, data_segment.shape[1])
            
            # Plotting
            self.plot_raw.clear_plot()
            ax = self.plot_raw.get_axes()
            
            # Add vertical offset to stack signals neatly
            offset_step = 30.0 # uV separation
            
            for i in range(len(ch_names)):
                signal = data_segment[i, :]
                offset = i * offset_step
                
                # Color coding
                color = 'white'
                if 'C3' in ch_names[i]: color = '#00ff41'   # Green
                elif 'C4' in ch_names[i]: color = '#ff0055' # Red/Pink
                elif 'Cz' in ch_names[i]: color = 'cyan'
                elif 'EOG' in ch_names[i]: color = 'yellow'
                
                ax.plot(times, signal + offset, color=color, linewidth=1.0, label=ch_names[i])
            
            ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, label='Cue Onset')
            
            # Legend styling (limit to fit)
            if len(ch_names) <= 6:
                ax.legend(loc='upper right', fontsize='small', ncol=3)
                
            self.plot_raw.style_axes(ax, 
                                     title=f"SIGNAL INSPECTOR - TRIAL #{trial_idx+1} [{label_str}]", 
                                     xlabel="Time relative to Cue (s)", 
                                     ylabel="Amplitude (Stacked uV)")
            self.plot_raw.draw()
            
        except Exception as e:
            self.log(f"INSPECTOR ERROR: {e}")

    # ==================================================================================
    # TAB 2: TIME-FREQUENCY (CWT) - SCROLLABLE MULTI-PLOT
    # ==================================================================================
    def init_tab_cwt(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Configuration Group
        g = QGroupBox("CWT CONFIGURATION")
        h = QHBoxLayout()
        self.combo_wavelet = QComboBox()
        self.combo_wavelet.addItems(["Morlet (Complex)", "Mexican Hat (Real)"])
        
        self.input_fmin = QLineEdit("4")
        self.input_fmax = QLineEdit("40")
        
        btn = QPushButton("[ RUN CWT (ALL 3 CHANNELS) ]")
        btn.clicked.connect(self.run_cwt_analysis)
        
        h.addWidget(QLabel("Wavelet:"))
        h.addWidget(self.combo_wavelet)
        h.addWidget(QLabel("F_min [Hz]:"))
        h.addWidget(self.input_fmin)
        h.addWidget(QLabel("F_max [Hz]:"))
        h.addWidget(self.input_fmax)
        h.addWidget(btn)
        g.setLayout(h)
        layout.addWidget(g)
        
        # Educational Text
        self.txt_cwt_desc = QTextEdit()
        self.txt_cwt_desc.setReadOnly(True)
        self.txt_cwt_desc.setMaximumHeight(60)
        self.txt_cwt_desc.setText(CWT.get_cwt_interpretation("Multi-Channel"))
        layout.addWidget(self.txt_cwt_desc)

        # Scrollable Layout for 3 Plots
        self.scroll_cwt = ScrollableChannelLayout()
        layout.addWidget(self.scroll_cwt)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "2. TIME-FREQ ANALYSIS")

    def run_cwt_analysis(self):
        if self.raw_data_array is None:
            self.log("ERROR: NO DATA LOADED.")
            return

        w_type = 'mexican_hat' if "Mexican" in self.combo_wavelet.currentText() else 'morlet'
        
        try:
            fmin = float(self.input_fmin.text())
            fmax = float(self.input_fmax.text())
            
            self.log(f"RUNNING CWT ({w_type.upper()}) on C3, Cz, C4...")
            
            if hasattr(self, 'valid_trials') and len(self.valid_trials) > 0:
                # Use current trial index from Tab 1
                trial_idx = self.spin_trial.value() - 1
                if trial_idx < len(self.valid_trials):
                    event = self.valid_trials[trial_idx]
                    idx = event[0]
                    label_str = "LEFT" if event[2] == 769 else "RIGHT"
                    
                    # Window: -1s to +4s
                    start = max(0, idx - int(1.0*self.fs))
                    end = min(self.raw_data_array.shape[1], idx + int(4.0*self.fs))
                    
                    # Get the plots from the scroll group
                    plots = self.scroll_cwt.get_plots()
                    ch_names = ['C3 (Left Hemi)', 'Cz (Vertex)', 'C4 (Right Hemi)']
                    
                    for i in range(3): # Loop C3, Cz, C4
                        segment = self.raw_data_array[i, start:end]
                        tfr, freqs = CWT.run_cwt(segment, self.fs, fmin, fmax, 0.5, w_type)
                        
                        p = plots[i]
                        p.clear_plot()
                        ax = p.get_axes()
                        
                        extent = [-1.0, 4.0, fmin, fmax]
                        im = ax.imshow(tfr, extent=extent, aspect='auto', origin='lower', cmap='jet')
                        
                        # Colorbar
                        fig = p.get_figure()
                        cb = fig.colorbar(im, ax=ax)
                        cb.set_label('Power', color='white')
                        cb.ax.yaxis.set_tick_params(color='white')
                        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
                        
                        p.style_axes(ax, title=f"{ch_names[i]} - Trial #{trial_idx+1} ({label_str})", 
                                     xlabel="Time (s)", ylabel="Frequency (Hz)")
                        p.draw()
                        
                    self.log("SPECTROGRAMS GENERATED SUCCESSFULLY.")
                else:
                    self.log("ERROR: Invalid Trial Index.")
            else:
                self.log("ERROR: NO VALID TRIALS FOUND.")
        except Exception as e:
            self.log(f"CWT ERROR: {e}")

    # ==================================================================================
    # TAB 3: FILTERING - SCROLLABLE MULTI-PLOT
    # ==================================================================================
    def init_tab_filter(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        g = QGroupBox("BPF PARAMETERS")
        h = QHBoxLayout()
        self.input_lowcut = QLineEdit("8.0")
        self.input_highcut = QLineEdit("30.0")
        self.input_order = QComboBox()
        self.input_order.addItems(["2 (Standard)", "4 (Steep Cascade)"])
        
        h.addWidget(QLabel("Low [Hz]:"))
        h.addWidget(self.input_lowcut)
        h.addWidget(QLabel("High [Hz]:"))
        h.addWidget(self.input_highcut)
        h.addWidget(QLabel("Order:"))
        h.addWidget(self.input_order)
        
        btn = QPushButton("[ APPLY FILTER (ALL 3 CHANNELS) ]")
        btn.clicked.connect(self.apply_filter)
        h.addWidget(btn)
        g.setLayout(h)
        layout.addWidget(g)
        
        self.txt_filter_desc = QTextEdit()
        self.txt_filter_desc.setReadOnly(True)
        self.txt_filter_desc.setMaximumHeight(60)
        self.txt_filter_desc.setText(filtering_BPF_EEG.get_filter_description())
        layout.addWidget(self.txt_filter_desc)
        
        # Scrollable Layout
        self.scroll_filter = ScrollableChannelLayout()
        layout.addWidget(self.scroll_filter)
        
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
            
            self.filtered_data = filtering_BPF_EEG.run_filter_multi_channel(
                self.raw_data_array, self.fs, low, high, order
            )
            
            plots = self.scroll_filter.get_plots()
            ch_names = ['C3', 'Cz', 'C4']
            t = np.linspace(0, 5, int(5*self.fs))
            
            for i in range(3):
                p = plots[i]
                p.clear_plot()
                ax = p.get_axes()
                
                # Plot Raw (Gray) vs Filtered (Cyan) for comparison
                ax.plot(t, self.raw_data_array[i, :len(t)], color='gray', alpha=0.5, label='Raw Signal')
                ax.plot(t, self.filtered_data[i, :len(t)], color='cyan', linewidth=1.2, label='Filtered Signal')
                
                p.style_axes(ax, title=f"FILTER CHECK: {ch_names[i]} (First 5s)", 
                             xlabel="Time (s)", ylabel="Amplitude (uV)")
                ax.legend(loc='upper right')
                p.draw()
                
            self.log("FILTERING COMPLETE.")
        except Exception as e:
            self.log(f"FILTER ERROR: {e}")

    # ==================================================================================
    # TAB 4: SQUARING - SCROLLABLE MULTI-PLOT
    # ==================================================================================
    def init_tab_squaring(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        btn = QPushButton("[ EXECUTE SQUARING (ALL 3 CHANNELS) ]")
        btn.clicked.connect(self.apply_squaring)
        layout.addWidget(btn)
        
        self.txt_square_desc = QTextEdit()
        self.txt_square_desc.setReadOnly(True)
        self.txt_square_desc.setMaximumHeight(60)
        self.txt_square_desc.setText(squaring_EEG.get_squaring_description())
        layout.addWidget(self.txt_square_desc)
        
        self.scroll_square = ScrollableChannelLayout()
        layout.addWidget(self.scroll_square)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "4. SQUARING")

    def apply_squaring(self):
        if self.filtered_data is None:
            self.log("ERROR: DATA NOT FILTERED.")
            return
        
        self.log("SQUARING SIGNALS...")
        self.squared_data = squaring_EEG.square_signal(self.filtered_data)
        
        plots = self.scroll_square.get_plots()
        ch_names = ['C3', 'Cz', 'C4']
        t = np.linspace(0, 5, int(5*self.fs))
        
        for i in range(3):
            p = plots[i]
            p.clear_plot()
            ax = p.get_axes()
            
            ax.plot(t, self.filtered_data[i, :len(t)], color='gray', alpha=0.5, label='Amplitude')
            ax.plot(t, self.squared_data[i, :len(t)], color='magenta', linewidth=1.0, label='Power (Squared)')
            
            p.style_axes(ax, title=f"INSTANTANEOUS POWER: {ch_names[i]}", 
                         xlabel="Time (s)", ylabel="Power (uV^2)")
            ax.legend(loc='upper right')
            p.draw()
            
        self.log("SQUARING COMPLETE.")

    # ==================================================================================
    # TAB 5: AVERAGING - SCROLLABLE MULTI-PLOT
    # ==================================================================================
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
        
        btn = QPushButton("[ COMPUTE GRAND AVERAGE (ALL CHANNELS) ]")
        btn.clicked.connect(self.apply_averaging)
        h.addWidget(btn)
        g.setLayout(h)
        layout.addWidget(g)
        
        self.txt_avg_desc = QTextEdit()
        self.txt_avg_desc.setReadOnly(True)
        self.txt_avg_desc.setMaximumHeight(60)
        self.txt_avg_desc.setText(average_all_EEG_trials.get_averaging_description())
        layout.addWidget(self.txt_avg_desc)
        
        self.scroll_avg = ScrollableChannelLayout()
        layout.addWidget(self.scroll_avg)
        
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
            
            plots = self.scroll_avg.get_plots()
            ch_names = ['C3', 'Cz', 'C4']
            
            for i in range(3):
                p = plots[i]
                p.clear_plot()
                ax = p.get_axes()
                
                # Plot Class 1 (Left Hand) vs Class 2 (Right Hand)
                ax.plot(self.time_axis_epochs, self.avg_left[i, :], color='cyan', linewidth=1.5, label='Left Hand Class')
                ax.plot(self.time_axis_epochs, self.avg_right[i, :], color='red', linestyle='--', linewidth=1.5, label='Right Hand Class')
                
                ax.axvline(0, color='white', linestyle=':', label='Cue Onset')
                
                p.style_axes(ax, title=f"GRAND AVERAGE: {ch_names[i]}", 
                             xlabel="Time relative to Cue (s)", ylabel="Mean Power")
                ax.legend(loc='upper right')
                p.draw()
                
            self.log("AVERAGING DONE.")
        except Exception as e:
            self.log(f"AVG ERROR: {e}")

    # ==================================================================================
    # TAB 6: SMOOTHING - SCROLLABLE MULTI-PLOT
    # ==================================================================================
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
        
        self.txt_smooth_desc = QTextEdit()
        self.txt_smooth_desc.setReadOnly(True)
        self.txt_smooth_desc.setMaximumHeight(60)
        self.txt_smooth_desc.setText(moving_average_EEG.get_smoothing_description())
        layout.addWidget(self.txt_smooth_desc)
        
        self.scroll_smooth = ScrollableChannelLayout()
        layout.addWidget(self.scroll_smooth)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "6. SMOOTHING")

    def apply_smoothing(self):
        if self.avg_left is None:
            self.log("ERROR: NO AVERAGED DATA.")
            return
        try:
            win = float(self.input_window.text())
            self.log(f"SMOOTHING (Window={win}s)...")
            
            # Smooth both classes (Multi-channel wrapper handles 3 channels automatically)
            self.smoothed_left = moving_average_EEG.apply_moving_average(self.avg_left, self.fs, win)
            self.smoothed_right = moving_average_EEG.apply_moving_average(self.avg_right, self.fs, win)
            
            plots = self.scroll_smooth.get_plots()
            ch_names = ['C3', 'Cz', 'C4']
            
            for i in range(3):
                p = plots[i]
                p.clear_plot()
                ax = p.get_axes()
                
                # Plot Noisy Background vs Smoothed Envelopes
                ax.plot(self.time_axis_epochs, self.avg_left[i, :], color='gray', alpha=0.3, label='Noisy')
                ax.plot(self.time_axis_epochs, self.smoothed_left[i, :], color='cyan', linewidth=2.0, label='Smoothed Left')
                ax.plot(self.time_axis_epochs, self.smoothed_right[i, :], color='red', linewidth=2.0, linestyle='--', label='Smoothed Right')
                
                ax.axvline(0, color='white', linestyle=':')
                
                p.style_axes(ax, title=f"SMOOTHED ENVELOPE: {ch_names[i]}", 
                             xlabel="Time (s)", ylabel="Power")
                ax.legend(loc='upper right')
                p.draw()
                
            self.log("SMOOTHING COMPLETE.")
        except Exception as e:
            self.log(f"SMOOTH ERROR: {e}")

    # ==================================================================================
    # TAB 7: ERD/ERS PERCENTAGE - SCROLLABLE MULTI-PLOT
    # ==================================================================================
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
        
        self.txt_erd_desc = QTextEdit()
        self.txt_erd_desc.setReadOnly(True)
        self.txt_erd_desc.setMaximumHeight(60)
        self.txt_erd_desc.setText(percentage_ERD_ERS.get_erd_description())
        layout.addWidget(self.txt_erd_desc)
        
        self.scroll_erd = ScrollableChannelLayout()
        layout.addWidget(self.scroll_erd)
        
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
            
            plots = self.scroll_erd.get_plots()
            ch_names = ['C3', 'Cz', 'C4']
            
            for i in range(3):
                p = plots[i]
                p.clear_plot()
                ax = p.get_axes()
                
                # Plot ERD % Curves
                ax.plot(self.time_axis_epochs, self.erd_left[i, :], color='cyan', linewidth=2.0, label='Left Hand')
                ax.plot(self.time_axis_epochs, self.erd_right[i, :], color='magenta', linewidth=2.0, label='Right Hand')
                
                ax.axhline(0, color='white', linewidth=1)
                ax.axvline(0, color='gray', linestyle=':')
                
                # Fill ERD (Negative area) for visual clarity
                # Example: Highlighting Right Hand ERD
                ax.fill_between(self.time_axis_epochs, self.erd_right[i, :], 0, 
                                where=(self.erd_right[i, :] < 0), color='magenta', alpha=0.15)
                
                p.style_axes(ax, title=f"ERD/ERS %: {ch_names[i]}", 
                             xlabel="Time relative to Cue (s)", ylabel="Power Change (%)")
                ax.legend(loc='upper right')
                p.draw()
                
            self.log("ERD CALCULATION DONE.")
        except Exception as e:
            self.log(f"ERD ERROR: {e}")

    # ==================================================================================
    # TAB 8: CSP & FEATURES
    # ==================================================================================
    def init_tab_csp(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        btn_train_csp = QPushButton("[ TRAIN CSP & EXTRACT FEATURES ]")
        btn_train_csp.clicked.connect(self.run_csp_training)
        layout.addWidget(btn_train_csp)
        
        self.txt_csp_desc = QTextEdit()
        self.txt_csp_desc.setReadOnly(True)
        self.txt_csp_desc.setMaximumHeight(80)
        self.txt_csp_desc.setText(csp_scratch.get_csp_description())
        layout.addWidget(self.txt_csp_desc)
        
        # Splitter to hold Scatter Plot and Temporal BoxPlot side-by-side
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel: Feature Scatter
        self.plot_csp_scatter = PlotWidget()
        splitter.addWidget(self.plot_csp_scatter)
        
        # Right Panel: Temporal BoxPlot
        self.plot_temp_box = PlotWidget()
        splitter.addWidget(self.plot_temp_box)
        
        layout.addWidget(splitter)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "8. CSP & FEATURES")

    def run_csp_training(self):
        if self.filtered_data is None:
            QMessageBox.warning(self, "Pipeline Error", "Please run Filtering (Tab 3) first!")
            return
            
        try:
            self.log("PREPARING EPOCHS FOR CSP...")
            epochs, labels = self.ml_pipeline.prepare_data(
                self.filtered_data, self.events, self.fs, tmin=0.5, tmax=3.5
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
            
            # Extract Features for Visualization
            csp_feats = self.ml_pipeline.csp.transform(epochs)
            
            # Extract Temporal Features for Visualization
            temp_extractor = csp_scratch.TemporalFeatureExtractor()
            temp_feats = temp_extractor.transform(epochs)
            
            # 1. Plot Scatter
            self.plot_csp_scatter.clear_plot()
            ax1 = self.plot_csp_scatter.get_axes()
            
            ax1.scatter(csp_feats[labels==0, 0], csp_feats[labels==0, 1], 
                        color='cyan', label='Left Hand', alpha=0.7, edgecolors='white')
            ax1.scatter(csp_feats[labels==1, 0], csp_feats[labels==1, 1], 
                        color='magenta', label='Right Hand', alpha=0.7, edgecolors='white')
            
            self.plot_csp_scatter.style_axes(ax1, title="CSP FEATURE SPACE (CLUSTERING)", 
                                             xlabel="Log-Var (Comp 1)", ylabel="Log-Var (Comp 2)")
            ax1.legend()
            self.plot_csp_scatter.draw()
            
            # 2. Plot BoxPlot (Variance of C3)
            self.plot_temp_box.clear_plot()
            ax2 = self.plot_temp_box.get_axes()
            
            # Metric index 1 = Variance (Feature order: Mean, Var, Std, Skew, Kurt)
            # Channel 0 = C3
            n_metrics = 5
            col_idx = (0 * n_metrics) + 1 
            
            data_c0 = temp_feats[labels==0, col_idx]
            data_c1 = temp_feats[labels==1, col_idx]
            
            # Matplotlib 3.9+ 'tick_labels' compatibility
            box = ax2.boxplot([data_c0, data_c1], tick_labels=['Left', 'Right'], patch_artist=True,
                              medianprops=dict(color="white"), 
                              whiskerprops=dict(color="white"),  # Makes vertical lines white
                              capprops=dict(color="white"))
            
            colors = ['cyan', 'magenta']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            self.plot_temp_box.style_axes(ax2, title="TEMPORAL FEATURE: VARIANCE (C3)", 
                                          xlabel="Class", ylabel="Value")
            self.plot_temp_box.draw()
            
            self.log("CSP & FEATURES COMPUTED.")
            
        except Exception as e:
            self.log(f"CSP ERROR: {e}")

    # ==================================================================================
    # TAB 9: MACHINE LEARNING
    # ==================================================================================
    def init_tab_ml(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        # --- Controls ---
        h_ctrl = QHBoxLayout()
        btn_compare = QPushButton("[ TRAIN & COMPARE ALL 8 MODELS ]")
        btn_compare.clicked.connect(self.run_ml_comparison)
        
        self.combo_ml_plot = QComboBox()
        self.combo_ml_plot.setFixedWidth(250)
        self.combo_ml_plot.addItem("Select Model to View...")
        self.combo_ml_plot.currentIndexChanged.connect(self.update_ml_visuals)
        
        h_ctrl.addWidget(btn_compare)
        h_ctrl.addWidget(self.combo_ml_plot)
        layout.addLayout(h_ctrl)
        
        # --- Content Area (Sub-Tabs) ---
        self.ml_tabs = QTabWidget()
        
        # Sub-tab 1: Performance Bar Chart
        self.tab_ml_bar = QWidget()
        l_bar = QVBoxLayout(self.tab_ml_bar)
        self.plot_ml_bar = PlotWidget()
        l_bar.addWidget(self.plot_ml_bar)
        self.ml_tabs.addTab(self.tab_ml_bar, "ACCURACY COMPARISON")
        
        # Sub-tab 2: Confusion Matrix
        self.tab_ml_cm = QWidget()
        l_cm = QVBoxLayout(self.tab_ml_cm)
        self.plot_ml_cm = PlotWidget()
        l_cm.addWidget(self.plot_ml_cm)
        self.ml_tabs.addTab(self.tab_ml_cm, "CONFUSION MATRIX")
        
        # Sub-tab 3: Learning Curve
        self.tab_ml_perf = QWidget()
        l_perf = QVBoxLayout(self.tab_ml_perf)
        self.plot_ml_perf = PlotWidget()
        l_perf.addWidget(self.plot_ml_perf)
        self.ml_tabs.addTab(self.tab_ml_perf, "LEARNING/LOSS CURVES")
        
        # Sub-tab 4: Detailed Results Table
        self.tab_ml_details = QWidget()
        l_det = QVBoxLayout(self.tab_ml_details)
        self.txt_ml_results = QTextEdit()
        self.txt_ml_results.setReadOnly(True)
        self.txt_ml_results.setMaximumHeight(150)
        
        self.table_ml_details = QTableWidget()
        self.table_ml_details.setColumnCount(4)
        self.table_ml_details.setHorizontalHeaderLabels(["Trial ID", "True Label", "Predicted", "Status"])
        self.table_ml_details.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_ml_details.setStyleSheet("gridline-color: #00ff41; color: white;")
        
        l_det.addWidget(self.txt_ml_results)
        l_det.addWidget(self.table_ml_details)
        self.ml_tabs.addTab(self.tab_ml_details, "DETAILED PREDICTIONS")
        
        layout.addWidget(self.ml_tabs)
        
        # Educational Text
        self.txt_ml_desc = QTextEdit()
        self.txt_ml_desc.setReadOnly(True)
        self.txt_ml_desc.setMaximumHeight(80)
        try:
            self.txt_ml_desc.setText(ml_analysis.get_ml_description())
        except AttributeError:
            self.txt_ml_desc.setText("Machine Learning Description Loading...")
            
        layout.addWidget(self.txt_ml_desc)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "9. CLASSIFICATION")

    def run_ml_comparison(self):
        if self.ml_epochs is None:
            QMessageBox.warning(self, "Pipeline Error", "Please run CSP Training (Tab 8) first!")
            return
            
        try:
            self.log("STARTING FULL MODEL COMPARISON...")
            self.txt_ml_results.setText(">> TRAINING IN PROGRESS... PLEASE WAIT.\n")
            QApplication.processEvents()
            
            # Run Pipeline
            self.ml_metrics = self.ml_pipeline.run_full_comparison(self.ml_epochs, self.ml_labels, test_size=0.25)
            
            # Sort Results
            sorted_metrics = sorted(self.ml_metrics.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
            
            # Update Dropdown
            self.combo_ml_plot.clear()
            self.combo_ml_plot.addItem("Select Model to View...")
            for name, _ in sorted_metrics:
                self.combo_ml_plot.addItem(name)
                
            # Generate Report Text
            report = ">> MODEL METRICS REPORT:\n" + "="*50 + "\n"
            report += f"{'MODEL':<20} | {'ACC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8}\n"
            report += "-"*50 + "\n"
            
            names = []
            accs = []
            
            for name, m in sorted_metrics:
                report += f"{name:<20} | {m['Accuracy']*100:.2f}     | {m['Precision']*100:.2f}     | {m['Recall']*100:.2f}     | {m['F1']:.2f}\n"
                names.append(name)
                accs.append(m['Accuracy'] * 100)
                
            report += "="*50 + "\n"
            report += f"BEST MODEL: {sorted_metrics[0][0]}"
            self.txt_ml_results.setText(report)
            
            # Plot Bar Chart
            self.plot_ml_bar.clear_plot()
            ax = self.plot_ml_bar.get_axes()
            bars = ax.bar(names, accs, color='#00ff41', alpha=0.7)
            self.plot_ml_bar.style_axes(ax, title="MODEL ACCURACY COMPARISON", xlabel="Model", ylabel="Accuracy (%)")
            
            # Fix Layout for rotated labels
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(len(names))))
            ax.set_xticklabels(names, rotation=45, ha='right')
            self.plot_ml_bar.fig.subplots_adjust(bottom=0.25)
            
            # Add labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', color='white', fontsize=9)
            
            self.plot_ml_bar.draw()
            
            self.combo_ml_plot.setCurrentText(sorted_metrics[0][0]) # Select best model automatically
            self.log("ML COMPARISON COMPLETE.")
            
        except Exception as e:
            self.log(f"ML ERROR: {e}")
            self.txt_ml_results.setText(f"ERROR: {e}")

    def update_ml_visuals(self):
        model_name = self.combo_ml_plot.currentText()
        if "Select" in model_name or model_name not in self.ml_metrics:
            return
            
        try:
            # 1. Plot Confusion Matrix
            y_true, y_pred = self.ml_pipeline.get_prediction(model_name)
            if y_true is None: return
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            self.plot_ml_cm.clear_plot()
            ax = self.plot_ml_cm.get_axes()
            im = ax.imshow(cm, interpolation='nearest', cmap='Greens')
            cb = self.plot_ml_cm.fig.colorbar(im, ax=ax)
            cb.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')
            
            classes = ['Left', 'Right']
            ax.set_xticks(np.arange(2))
            ax.set_yticks(np.arange(2))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
            
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            self.plot_ml_cm.style_axes(ax, title=f"CONFUSION MATRIX: {model_name}", 
                                       xlabel="Predicted Label", ylabel="True Label")
            self.plot_ml_cm.draw()
            
            # 2. Plot Performance Curve (Learning Curve or Loss Curve)
            self.plot_ml_perf.clear_plot()
            
            # If MLP, try to show Loss Curve first
            if "MLP" in model_name:
                fig_loss = self.ml_pipeline.generate_loss_curve(model_name)
                if fig_loss:
                    model = self.ml_pipeline.trained_models[model_name]
                    ax_perf = self.plot_ml_perf.get_axes()
                    ax_perf.plot(model.loss_curve_, color='#00ff41', linewidth=2)
                    self.plot_ml_perf.style_axes(ax_perf, title=f"LOSS CURVE: {model_name}", 
                                                 xlabel="Epochs", ylabel="Loss")
                    self.plot_ml_perf.draw()
            else:
                # Standard Learning Curve
                from sklearn.model_selection import learning_curve
                model = self.ml_pipeline.trained_models[model_name]
                ax_perf = self.plot_ml_perf.get_axes()
                
                # Wrap Learning Curve in Try-Except for Safety
                try:
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, self.ml_pipeline.X_train, self.ml_pipeline.y_train, 
                        cv=5, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5)
                    )
                    train_mean = np.mean(train_scores, axis=1) * 100
                    test_mean = np.mean(test_scores, axis=1) * 100
                    
                    ax_perf.plot(train_sizes, train_mean, 'o-', color="cyan", label="Training")
                    ax_perf.plot(train_sizes, test_mean, 'o-', color="magenta", label="Validation")
                    ax_perf.legend()
                    self.plot_ml_perf.style_axes(ax_perf, title=f"LEARNING CURVE: {model_name}", 
                                                 xlabel="Training Samples", ylabel="Accuracy (%)")
                    self.plot_ml_perf.draw()
                except ValueError:
                    ax_perf.text(0.5, 0.5, "Insufficient Data for Curve", color='white', ha='center')
                    self.plot_ml_perf.draw()

            # 3. Update Detailed Table
            details = self.ml_pipeline.get_detailed_predictions(model_name)
            self.table_ml_details.setRowCount(len(details))
            for i, (tid, true_l, pred_l, status) in enumerate(details):
                self.table_ml_details.setItem(i, 0, QTableWidgetItem(str(tid)))
                self.table_ml_details.setItem(i, 1, QTableWidgetItem(true_l))
                self.table_ml_details.setItem(i, 2, QTableWidgetItem(pred_l))
                
                item_status = QTableWidgetItem(status)
                if status == "CORRECT":
                    item_status.setForeground(QColor("#00ff41"))
                else:
                    item_status.setForeground(QColor("#ff3333"))
                self.table_ml_details.setItem(i, 3, item_status)
                
        except Exception as e:
            self.log(f"VISUAL UPDATE ERROR: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGAnalysisWindow()
    window.show()
    sys.exit(app.exec())