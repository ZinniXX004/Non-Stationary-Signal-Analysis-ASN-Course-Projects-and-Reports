"""
GUI.py

Purpose:
    - Main User Interface for the EEG Motor Imagery Analysis Project (Version 11.0).
    - Orchestrates the full pipeline with Multi-Dataset inputs (Train x2, Test x1, Eval x1).
    - Features:
      1. Complex Data Loading: Manage multiple files for Training vs Testing vs Evaluation.
      2. Multi-Band Processing: Select between Mu (8-13Hz) and Beta (13-30Hz) bands for analysis.
      3. Active Context: Select which dataset to visualize in Tabs 2-7.
      4. Complete Feature Visualization: CSP Scatter + All 5 Time-Domain Boxplots.
      5. ML Pipeline: Train on merged datasets, Test on independent dataset, Infer on Evaluation dataset.
    - Theme: Retro Hacker Terminal (Dark Mode).

Dependencies:
    - PyQt6, matplotlib, numpy, mne
    - Custom Modules
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
    QHeaderView, QCheckBox, QListWidget, QAbstractItemView, QSizePolicy, QFrame, QRadioButton, QButtonGroup
)
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon
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

# CLASS: PlotWidget
class PlotWidget(QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100, min_height=None):
        super(PlotWidget, self).__init__(parent)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(2)

        # Initialize Figure with Dark Background
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#0d0d0d') 
        
        # Initialize Canvas
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #0d0d0d;")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        if min_height:
            self.canvas.setMinimumHeight(min_height)
        
        # Initialize Axes
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#0d0d0d')
        
        # Initialize Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #262626; color: white; border: 1px solid #333;")

        # Add to Layout
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

    def clear_plot(self):
        """Resets the plot axes and maintains dark theme."""
        self.fig.clf() 
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#0d0d0d')
        self.fig.patch.set_facecolor('#0d0d0d')
        self.canvas.draw()

    def get_axes(self):
        return self.axes

    def get_figure(self):
        return self.fig

    def draw(self):
        try:
            self.canvas.draw()
        except Exception as e:
            print(f"[GUI ERROR] Plotting failed: {e}")

    def style_axes(self, ax, title="", xlabel="", ylabel=""):
        """Applies high-contrast styling to the axes."""
        ax.set_facecolor('#0d0d0d')
        ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
        ax.set_xlabel(xlabel, color='white', fontsize=9)
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values(): spine.set_color('white')
        ax.grid(True, color='#333333', linestyle='--', linewidth=0.5)
        if ax.get_legend():
            plt.setp(ax.get_legend().get_texts(), color='white')
            ax.get_legend().get_frame().set_facecolor('#1a1a1a')
            ax.get_legend().get_frame().set_edgecolor('white')

# CLASS: ScrollableChannelLayout
class ScrollableChannelLayout(QWidget):
    def __init__(self, parent=None):
        super(ScrollableChannelLayout, self).__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background-color: #0d0d0d;")
        
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container.setStyleSheet("background-color: #0d0d0d;")
        
        self.plot_c3 = PlotWidget(min_height=350)
        self.plot_cz = PlotWidget(min_height=350)
        self.plot_c4 = PlotWidget(min_height=350)
        
        # Labels
        lbl_style = "color: #00ff41; font-weight: bold; padding: 5px; background: #111;"
        self.lbl_c3 = QLabel("CHANNEL C3 (LEFT MOTOR CORTEX)")
        self.lbl_c3.setStyleSheet(lbl_style)
        
        self.lbl_cz = QLabel("CHANNEL Cz (VERTEX)")
        self.lbl_cz.setStyleSheet(lbl_style.replace("#00ff41", "cyan"))
        
        self.lbl_c4 = QLabel("CHANNEL C4 (RIGHT MOTOR CORTEX)")
        self.lbl_c4.setStyleSheet(lbl_style.replace("#00ff41", "#ff0055"))
        
        self.container_layout.addWidget(self.lbl_c3)
        self.container_layout.addWidget(self.plot_c3)
        self.container_layout.addWidget(self.make_separator())
        
        self.container_layout.addWidget(self.lbl_cz)
        self.container_layout.addWidget(self.plot_cz)
        self.container_layout.addWidget(self.make_separator())
        
        self.container_layout.addWidget(self.lbl_c4)
        self.container_layout.addWidget(self.plot_c4)
        
        self.scroll_area.setWidget(self.container)
        self.main_layout.addWidget(self.scroll_area)

    def make_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #333;")
        return line

    def get_plots(self):
        return [self.plot_c3, self.plot_cz, self.plot_c4]

    def clear_all(self):
        self.plot_c3.clear_plot()
        self.plot_cz.clear_plot()
        self.plot_c4.clear_plot()

# CLASS: ScrollableFeatureLayout
class ScrollableFeatureLayout(QWidget):
    def __init__(self, parent=None):
        super(ScrollableFeatureLayout, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background: #0d0d0d;")
        
        self.container = QWidget()
        self.cont_layout = QVBoxLayout(self.container)
        self.container.setStyleSheet("background-color: #0d0d0d;")
        
        # 1. CSP Scatter
        self.plot_csp = PlotWidget(min_height=450)
        lbl_csp = QLabel("1. CSP FEATURE SPACE (SCATTER)")
        lbl_csp.setStyleSheet("color:white; font-weight:bold; padding: 5px; background: #222;")
        self.cont_layout.addWidget(lbl_csp)
        self.cont_layout.addWidget(self.plot_csp)
        
        # Separator
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setStyleSheet("background-color: #444;")
        self.cont_layout.addWidget(line)
        
        # 2. Temporal Boxplots (5 Metrics)
        self.plots_temporal = []
        metrics = ["Mean", "Variance", "StdDev", "Skewness", "Kurtosis"]
        for m in metrics:
            p = PlotWidget(min_height=350)
            lbl = QLabel(f"TEMPORAL FEATURE: {m.upper()} (DISTRIBUTION)")
            lbl.setStyleSheet("color:white; font-weight:bold; padding: 5px; background: #222;")
            self.cont_layout.addWidget(lbl)
            self.cont_layout.addWidget(p)
            self.plots_temporal.append(p)
            
        self.scroll.setWidget(self.container)
        self.layout.addWidget(self.scroll)

    def get_csp_plot(self):
        return self.plot_csp
    
    def get_temporal_plots(self):
        return self.plots_temporal
        
    def clear_all(self):
        self.plot_csp.clear_plot()
        for p in self.plots_temporal:
            p.clear_plot()

# CLASS: EEGAnalysisWindow (Main Application)
class EEGAnalysisWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("SYSTEM::EEG_ANALYSIS_CORE [MULTI-BAND SUPPORT]")
        self.setGeometry(50, 50, 1600, 1000)

        # GLOBAL DATA CONTAINERS
        self.datasets = {
            "Train": [], 
            "Test": None, 
            "Eval": None  
        }
        self.active_viz_data = None 

        # Processing Results
        self.filtered_data = None
        self.squared_data = None
        self.avg_left, self.avg_right = None, None
        self.smoothed_left, self.smoothed_right = None, None
        self.erd_left, self.erd_right = None, None
        self.time_axis = None
        
        # ML Pipeline
        self.ml_pipeline = ml_analysis.ML_Pipeline()
        self.ml_epochs = None
        self.ml_labels = None
        self.ml_metrics = {}

        self.apply_hacker_theme()

        # MAIN LAYOUT 
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self.create_header(main_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Initialize All Tabs
        self.init_tab_ingestion()   # Tab 1
        self.init_tab_cwt()         # Tab 2
        self.init_tab_filter()      # Tab 3 (Updated with Band Selector)
        self.init_tab_squaring()    # Tab 4
        self.init_tab_averaging()   # Tab 5
        self.init_tab_smoothing()   # Tab 6
        self.init_tab_erd()         # Tab 7
        self.init_tab_features()    # Tab 8
        self.init_tab_ml()          # Tab 9

        # 3. Log Console
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(150)
        self.log_console.setStyleSheet("border: 1px solid #00ff41; color: #00ff41; background-color: #000; font-family: Consolas;")
        main_layout.addWidget(self.log_console)
        
        self.log("SYSTEM ONLINE. READY FOR MULTI-BAND PROCESSING.")

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
        QListWidget { border: 1px solid #00ff41; }
        QTableWidget { gridline-color: #00ff41; color: #fff; }
        QHeaderView::section { background-color: #1a1a1a; color: #00ff41; border: 1px solid #00ff41; }
        QScrollArea { border: none; background-color: #0d0d0d; }
        QCheckBox { color: #00ff41; }
        QRadioButton { color: #00ff41; }
        """
        self.setStyleSheet(style)

    def create_header(self, layout):
        h = QHBoxLayout()
        lbl = QLabel(">> EEG BCI PROJECT: MULTI-BAND ANALYSIS FRAMEWORK")
        lbl.setFont(QFont("Consolas", 18, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #00ff41; letter-spacing: 2px;")
        
        btn_clr = QPushButton("[ CLEAR MEMORY ]")
        btn_clr.setFixedWidth(150)
        btn_clr.clicked.connect(self.clear_all_data)
        
        btn_ext = QPushButton("[ EXIT ]")
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
        if confirm == QMessageBox.StandardButton.No: return

        # 1. Clear Data Containers
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
        self.datasets = {
            "Train": [], 
            "Test": None, 
            "Eval": None  
        }
        self.active_viz_data = None

        # 2. Clear UI Inputs (Safe Checks)
        if hasattr(self, 'list_train'): self.list_train.clear()
        if hasattr(self, 'input_test_file'): self.input_test_file.clear()
        if hasattr(self, 'input_eval_file'): self.input_eval_file.clear()
        if hasattr(self, 'combo_viz_source'): 
            self.combo_viz_source.clear()
            self.combo_viz_source.addItem("None")

        # 3. Clear Plots (Safe Checks)
        if hasattr(self, 'plot_raw'): self.plot_raw.clear_plot()
        
        if hasattr(self, 'scroll_cwt'): self.scroll_cwt.clear_all()
        if hasattr(self, 'scroll_filter'): self.scroll_filter.clear_all()
        if hasattr(self, 'scroll_square'): self.scroll_square.clear_all()
        if hasattr(self, 'scroll_avg'): self.scroll_avg.clear_all()
        if hasattr(self, 'scroll_smooth'): self.scroll_smooth.clear_all()
        if hasattr(self, 'scroll_erd'): self.scroll_erd.clear_all()
        
        if hasattr(self, 'scroll_feats'): self.scroll_feats.clear_all()
        
        if hasattr(self, 'plot_ml_bar'): self.plot_ml_bar.clear_plot()
        if hasattr(self, 'plot_ml_cm'): self.plot_ml_cm.clear_plot()
        if hasattr(self, 'plot_ml_perf'): self.plot_ml_perf.clear_plot()
        
        if hasattr(self, 'txt_erd_vals'): self.txt_erd_vals.clear()
        if hasattr(self, 'txt_ml_results'): self.txt_ml_results.clear()
        if hasattr(self, 'table_ml_details'): self.table_ml_details.setRowCount(0)
        if hasattr(self, 'table_eval'): self.table_eval.setRowCount(0)
        
        if hasattr(self, 'lbl_file_status'):
            self.lbl_file_status.setText("STATUS: IDLE")
        
        self.log("SYSTEM MEMORY FLUSHED.")

    # TAB 1: COMPLEX INGESTION
    def init_tab_ingestion(self):
        tab = QWidget()
        
        # 1. Main Layout (Scroll Area Container)
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: #0d0d0d; }")
        
        # 2. Content Container
        content_widget = QWidget()
        l_content = QVBoxLayout(content_widget)
        l_content.setSpacing(20) 
        l_content.setContentsMargins(10, 10, 10, 10) 
        
        # Section A: Dataset Management
        g_data = QGroupBox("DATASET MANAGEMENT")
        l_data = QVBoxLayout()
        
        # 1. Training Files
        h_train = QHBoxLayout()
        h_train.addWidget(QLabel("Training Sets (e.g., B01T, B02T):"))
        self.list_train = QListWidget()
        self.list_train.setMaximumHeight(80)
        btn_add_train = QPushButton("[ + ADD TRAIN FILE ]")
        btn_add_train.clicked.connect(self.add_training_file)
        h_train.addWidget(self.list_train)
        h_train.addWidget(btn_add_train)
        l_data.addLayout(h_train)
        
        # 2. Test File
        h_test = QHBoxLayout()
        h_test.addWidget(QLabel("Test/Validation Set (e.g., B03T):"))
        self.input_test_file = QLineEdit(); self.input_test_file.setReadOnly(True)
        btn_set_test = QPushButton("[ SET TEST FILE ]")
        btn_set_test.clicked.connect(self.set_test_file)
        h_test.addWidget(self.input_test_file)
        h_test.addWidget(btn_set_test)
        l_data.addLayout(h_test)
        
        # 3. Evaluation File
        h_eval = QHBoxLayout()
        h_eval.addWidget(QLabel("Evaluation Set (e.g., B04E - Unlabeled):"))
        self.input_eval_file = QLineEdit(); self.input_eval_file.setReadOnly(True)
        btn_set_eval = QPushButton("[ SET EVAL FILE ]")
        btn_set_eval.clicked.connect(self.set_eval_file)
        h_eval.addWidget(self.input_eval_file)
        h_eval.addWidget(btn_set_eval)
        l_data.addLayout(h_eval)
        
        g_data.setLayout(l_data)
        l_content.addWidget(g_data)
        
        # Section B: Visualization Source Selection
        g_viz = QGroupBox("ACTIVE VISUALIZATION SOURCE (FOR TABS 2-7)")
        h_viz = QHBoxLayout()
        self.combo_viz_source = QComboBox()
        self.combo_viz_source.addItem("None")
        self.combo_viz_source.currentIndexChanged.connect(self.change_active_visualization)
        
        h_viz.addWidget(QLabel("Select Dataset to Visualize:"))
        h_viz.addWidget(self.combo_viz_source)
        g_viz.setLayout(h_viz)
        l_content.addWidget(g_viz)
        
        # Section C: Inspector
        g_insp = QGroupBox("SIGNAL INSPECTOR")
        h_insp = QHBoxLayout()
        self.spin_trial = QSpinBox(); self.spin_trial.setPrefix("Trial: ")
        self.spin_trial.setFixedWidth(120); self.spin_trial.valueChanged.connect(self.update_raw_plot)
        
        self.input_insp_tmin = QLineEdit("-1.5"); self.input_insp_tmin.setFixedWidth(60)
        self.input_insp_tmax = QLineEdit("4.5"); self.input_insp_tmax.setFixedWidth(60)
        btn_update = QPushButton("[ UPDATE VIEW ]"); btn_update.clicked.connect(self.update_raw_plot)
        self.chk_show_all = QCheckBox("Show All Channels")
        self.chk_show_all.toggled.connect(self.update_raw_plot)
        
        h_insp.addWidget(self.spin_trial)
        h_insp.addWidget(QLabel("Window (s):"))
        h_insp.addWidget(self.input_insp_tmin)
        h_insp.addWidget(QLabel("to"))
        h_insp.addWidget(self.input_insp_tmax)
        h_insp.addWidget(btn_update)
        h_insp.addWidget(self.chk_show_all)
        h_insp.addStretch()
        g_insp.setLayout(h_insp)
        l_content.addWidget(g_insp)
        
        # Plot
        self.plot_raw = PlotWidget(min_height=450)
        l_content.addWidget(self.plot_raw)
        
        # Finalize Scroll Area
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        self.tabs.addTab(tab, "1. DATA MANAGEMENT")

    def _load_gdf_generic(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open GDF", "", "GDF Files (*.gdf)")
        if not fname: return None
        try:
            self.log(f"UPLOADING: {os.path.basename(fname)}...")
            raw, ev, ev_map, fs, info = load_data_eeg_mne.load_eeg_data(fname)
            if raw is None: return None
            
            # Identify motor channels
            picks = mne.pick_channels(raw.ch_names, include=['C3', 'Cz', 'C4'])
            if len(picks) < 3:
                self.log(f"WARN: Motor channels missing in {fname}")
                
            data_3ch = raw.get_data(picks=picks) * 1e6
            
            return {
                "name": os.path.basename(fname),
                "raw_obj": raw,
                "data_3ch": data_3ch,
                "events": ev,
                "fs": fs,
                "info": info
            }
        except Exception as e:
            self.log(f"LOAD ERROR: {e}")
            return None

    def add_training_file(self):
        data = self._load_gdf_generic()
        if data:
            self.datasets["Train"].append(data)
            self.list_train.addItem(data["name"])
            self.combo_viz_source.addItem(f"Train: {data['name']}")
            self.log(f"ADDED TRAIN FILE: {data['name']}")

    def set_test_file(self):
        data = self._load_gdf_generic()
        if data:
            self.datasets["Test"] = data
            self.input_test_file.setText(data["name"])
            self.combo_viz_source.addItem(f"Test: {data['name']}")
            self.log(f"SET TEST FILE: {data['name']}")

    def set_eval_file(self):
        data = self._load_gdf_generic()
        if data:
            self.datasets["Eval"] = data
            self.input_eval_file.setText(data["name"])
            self.combo_viz_source.addItem(f"Eval: {data['name']}")
            self.log(f"SET EVAL FILE: {data['name']}")
    
    def change_active_visualization(self):
        txt = self.combo_viz_source.currentText()
        
        if not txt or txt == "None":
            self.active_viz_data = None
            return
            
        parts = txt.split(": ")
        if len(parts) < 2:
            return
            
        target_name = parts[1]
        
        found = False
        for d in self.datasets["Train"]:
            if d["name"] == target_name:
                self.active_viz_data = d; found = True; break
                
        if not found and self.datasets["Test"] and self.datasets["Test"]["name"] == target_name:
            self.active_viz_data = self.datasets["Test"]; found = True
            
        if not found and self.datasets["Eval"] and self.datasets["Eval"]["name"] == target_name:
            self.active_viz_data = self.datasets["Eval"]; found = True
            
        if found:
            self.log(f"ACTIVE VISUALIZATION: {target_name}")
            valid_ev = [e for e in self.active_viz_data["events"] if e[2] in [769, 770, 783]]
            self.spin_trial.setMaximum(len(valid_ev) if len(valid_ev) > 0 else 1)
            self.update_raw_plot()

    def update_raw_plot(self):
        if self.active_viz_data is None: return
        try:
            trial_idx = self.spin_trial.value() - 1 
            tmin = float(self.input_insp_tmin.text())
            tmax = float(self.input_insp_tmax.text())
            show_all = self.chk_show_all.isChecked()
            
            raw = self.active_viz_data["raw_obj"]
            fs = self.active_viz_data["fs"]
            events = self.active_viz_data["events"]
            
            valid_trials = [e for e in events if e[2] in [769, 770, 783]]
            
            cue_sample = 0
            label_str = "N/A"
            if valid_trials and trial_idx < len(valid_trials):
                ev = valid_trials[trial_idx]
                cue_sample = ev[0]
                label_str = "LEFT" if ev[2] == 769 else "RIGHT" if ev[2] == 770 else "UNKNOWN"
                
            start = max(0, cue_sample + int(tmin * fs))
            end = min(raw.n_times, cue_sample + int(tmax * fs))
            
            if show_all:
                data = raw.get_data(start=start, stop=end) * 1e6
                names = raw.ch_names
            else:
                data = self.active_viz_data["data_3ch"][:, start:end]
                names = ['C3', 'Cz', 'C4']
                
            times = np.linspace(tmin, tmax, data.shape[1])
            
            self.plot_raw.clear_plot()
            ax = self.plot_raw.get_axes()
            
            for i, name in enumerate(names):
                offset = i * 40
                color = '#00ff41' if 'C3' in name else '#ff0055' if 'C4' in name else 'cyan' if 'Cz' in name else 'yellow'
                ax.plot(times, data[i,:] + offset, color=color, linewidth=1, label=name)
                
            ax.axvline(0, color='gray', linestyle='--', linewidth=1.5, label='Cue')
            if len(names) <= 6: ax.legend(loc='upper right', ncol=3, fontsize='small')
            
            self.plot_raw.style_axes(ax, title=f"SIGNAL INSPECTOR - TRIAL #{trial_idx+1} [{label_str}]", 
                                     xlabel="Time (s)", ylabel="Amplitude (Stacked uV)")
            self.plot_raw.draw()
        except Exception as e:
            self.log(f"PLOT ERROR: {e}")
            
    # TAB 2: TIME-FREQUENCY (CWT)
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
        
        self.combo_cwt_channel = QComboBox()
        self.combo_cwt_channel.addItems(["C3", "Cz", "C4"])
        
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
        if self.active_viz_data is None:
            self.log("ERROR: NO ACTIVE DATA SELECTED.")
            return

        w_type = 'mexican_hat' if "Mexican" in self.combo_wavelet.currentText() else 'morlet'
        
        try:
            fmin = float(self.input_fmin.text())
            fmax = float(self.input_fmax.text())
            
            self.log(f"RUNNING CWT ({w_type.upper()}) on C3, Cz, C4...")
            
            data = self.active_viz_data["data_3ch"]
            fs = self.active_viz_data["fs"]
            events = self.active_viz_data["events"]
            
            # Find relevant trials
            valid_trials = [e for e in events if e[2] in [769, 770, 783]]
            trial_idx = self.spin_trial.value() - 1
            
            if not valid_trials or trial_idx >= len(valid_trials):
                self.log("ERROR: Invalid Trial Index or No Valid Trials.")
                return
            
            cue = valid_trials[trial_idx][0]
            label_code = valid_trials[trial_idx][2]
            label_str = "LEFT" if label_code == 769 else "RIGHT" if label_code == 770 else "UNKNOWN"
            
            start = cue - int(1.0*fs)
            end = cue + int(4.0*fs)
            
            plots = self.scroll_cwt.get_plots()
            names = ['C3', 'Cz', 'C4']
            
            for i in range(3):
                segment = data[i, start:end]
                tfr, freqs = CWT.run_cwt(segment, fs, fmin, fmax, 0.5, w_type)
                
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
                
                p.style_axes(ax, title=f"SPECTROGRAM: {names[i]} - Trial #{trial_idx+1} ({label_str})", 
                             xlabel="Time (s)", ylabel="Frequency (Hz)")
                p.draw()
                
            self.log("SPECTROGRAMS GENERATED SUCCESSFULLY.")
        except Exception as e:
            self.log(f"CWT ERROR: {e}")

    # TAB 3: FILTERING - SCROLLABLE MULTI-PLOT (With Band Selection)
    def init_tab_filter(self):
        tab = QWidget()
        layout = QVBoxLayout()
        
        g = QGroupBox("BPF PARAMETERS")
        h = QHBoxLayout()
        
        # Band Selection Radio Buttons
        self.radio_mu = QRadioButton("Mu (8-13 Hz)")
        self.radio_beta = QRadioButton("Beta (13-30 Hz)")
        self.radio_broad = QRadioButton("Broadband (8-30 Hz)")
        self.radio_broad.setChecked(True)
        
        self.bg_band = QButtonGroup()
        self.bg_band.addButton(self.radio_mu)
        self.bg_band.addButton(self.radio_beta)
        self.bg_band.addButton(self.radio_broad)
        
        # Connect to auto-update filter values
        self.radio_mu.toggled.connect(lambda: self.update_filter_vals(8.0, 13.0))
        self.radio_beta.toggled.connect(lambda: self.update_filter_vals(13.0, 30.0))
        self.radio_broad.toggled.connect(lambda: self.update_filter_vals(8.0, 30.0))
        
        self.input_lowcut = QLineEdit("8.0")
        self.input_highcut = QLineEdit("30.0")
        self.input_order = QComboBox()
        self.input_order.addItems(["2 (Standard)", "4 (Steep Cascade)"])
        
        h.addWidget(QLabel("Preset:"))
        h.addWidget(self.radio_mu); h.addWidget(self.radio_beta); h.addWidget(self.radio_broad)
        h.addWidget(QLabel(" | Low [Hz]:")); h.addWidget(self.input_lowcut)
        h.addWidget(QLabel("High [Hz]:")); h.addWidget(self.input_highcut)
        h.addWidget(QLabel("Order:")); h.addWidget(self.input_order)
        
        btn = QPushButton("[ APPLY FILTER ]")
        btn.clicked.connect(self.apply_filter)
        h.addWidget(btn)
        g.setLayout(h)
        layout.addWidget(g)
        
        self.txt_filter_desc = QTextEdit()
        self.txt_filter_desc.setReadOnly(True)
        self.txt_filter_desc.setMaximumHeight(60)
        self.txt_filter_desc.setText(filtering_BPF_EEG.get_filter_description())
        layout.addWidget(self.txt_filter_desc)
        
        self.scroll_filter = ScrollableChannelLayout()
        layout.addWidget(self.scroll_filter)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "3. FILTERING")

    def update_filter_vals(self, low, high):
        self.input_lowcut.setText(str(low))
        self.input_highcut.setText(str(high))

    def apply_filter(self):
        if self.active_viz_data is None:
            self.log("ERROR: NO DATA SELECTED.")
            return
        try:
            low = float(self.input_lowcut.text())
            high = float(self.input_highcut.text())
            order = 2 if "2" in self.input_order.currentText() else 4
            
            self.log(f"FILTERING: {low}-{high} Hz (Order {order})...")
            
            raw = self.active_viz_data["data_3ch"]
            fs = self.active_viz_data["fs"]
            
            self.filtered_data = filtering_BPF_EEG.run_filter_multi_channel(
                raw, fs, low, high, order
            )
            
            # Update description based on band
            self.txt_filter_desc.setText(filtering_BPF_EEG.get_filter_description(low, high))
            
            plots = self.scroll_filter.get_plots()
            names = ['C3', 'Cz', 'C4']
            t = np.linspace(0, 5, int(5*fs))
            
            for i in range(3):
                p = plots[i]
                p.clear_plot()
                ax = p.get_axes()
                
                # Plot Raw (Gray) vs Filtered (Cyan) for comparison
                ax.plot(t, raw[i, :len(t)], color='gray', alpha=0.5, label='Raw Signal')
                ax.plot(t, self.filtered_data[i, :len(t)], color='cyan', linewidth=1.2, label='Filtered Signal')
                
                p.style_axes(ax, title=f"FILTER CHECK: {names[i]} (First 5s)", 
                             xlabel="Time (s)", ylabel="Amplitude (uV)")
                ax.legend(loc='upper right')
                p.draw()
                
            self.log("FILTERING COMPLETE.")
        except Exception as e:
            self.log(f"FILTER ERROR: {e}")

    # TAB 4: SQUARING - SCROLLABLE MULTI-PLOT
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
        names = ['C3', 'Cz', 'C4']
        t = np.linspace(0, 5, int(5*self.active_viz_data['fs']))
        
        for i in range(3):
            p = plots[i]
            p.clear_plot()
            ax = p.get_axes()
            
            ax.plot(t, self.filtered_data[i, :len(t)], color='gray', alpha=0.5, label='Amplitude')
            ax.plot(t, self.squared_data[i, :len(t)], color='magenta', linewidth=1.0, label='Power (Squared)')
            
            p.style_axes(ax, title=f"INSTANTANEOUS POWER: {names[i]}", 
                         xlabel="Time (s)", ylabel="Power (uV^2)")
            ax.legend(loc='upper right')
            p.draw()
            
        self.log("SQUARING COMPLETE.")

    # TAB 5: AVERAGING - SCROLLABLE MULTI-PLOT
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
            
            # This returns (3, n_times)
            self.avg_left, self.avg_right, self.time_axis_epochs = \
                average_all_EEG_trials.extract_and_average_epochs(
                    self.squared_data, self.active_viz_data["events"], self.active_viz_data["fs"], tmin, tmax)
            
            plots = self.scroll_avg.get_plots()
            names = ['C3', 'Cz', 'C4']
            
            for i in range(3):
                p = plots[i]
                p.clear_plot()
                ax = p.get_axes()
                
                # Plot Class 1 (Left Hand) vs Class 2 (Right Hand)
                ax.plot(self.time_axis_epochs, self.avg_left[i, :], color='cyan', linewidth=1.5, label='Left Hand Class')
                ax.plot(self.time_axis_epochs, self.avg_right[i, :], color='red', linestyle='--', linewidth=1.5, label='Right Hand Class')
                
                ax.axvline(0, color='white', linestyle=':', label='Cue Onset')
                
                p.style_axes(ax, title=f"GRAND AVERAGE: {names[i]}", 
                             xlabel="Time relative to Cue (s)", ylabel="Mean Power")
                ax.legend(loc='upper right')
                p.draw()
                
            self.log("AVERAGING DONE.")
        except Exception as e:
            self.log(f"AVG ERROR: {e}")

    # TAB 6: SMOOTHING - SCROLLABLE MULTI-PLOT
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
            self.smoothed_left = moving_average_EEG.apply_moving_average(self.avg_left, self.active_viz_data["fs"], win)
            self.smoothed_right = moving_average_EEG.apply_moving_average(self.avg_right, self.active_viz_data["fs"], win)
            
            plots = self.scroll_smooth.get_plots()
            names = ['C3', 'Cz', 'C4']
            
            for i in range(3):
                p = plots[i]
                p.clear_plot()
                ax = p.get_axes()
                
                # Plot Noisy Background vs Smoothed Envelopes
                ax.plot(self.time_axis_epochs, self.avg_left[i, :], color='gray', alpha=0.3, label='Noisy')
                ax.plot(self.time_axis_epochs, self.smoothed_left[i, :], color='cyan', linewidth=2.0, label='Smoothed Left')
                ax.plot(self.time_axis_epochs, self.smoothed_right[i, :], color='red', linewidth=2.0, linestyle='--', label='Smoothed Right')
                
                ax.axvline(0, color='white', linestyle=':')
                
                p.style_axes(ax, title=f"SMOOTHED ENVELOPE: {names[i]}", 
                             xlabel="Time (s)", ylabel="Power")
                ax.legend(loc='upper right')
                p.draw()
                
            self.log("SMOOTHING COMPLETE.")
        except Exception as e:
            self.log(f"SMOOTH ERROR: {e}")

    # TAB 7: ERD/ERS PERCENTAGE - SCROLLABLE MULTI-PLOT
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
        
        self.txt_erd_vals = QTextEdit()
        self.txt_erd_vals.setReadOnly(True)
        self.txt_erd_vals.setMaximumHeight(80)
        self.txt_erd_vals.setPlaceholderText("Numeric Average ERD% values will appear here...")
        layout.addWidget(self.txt_erd_vals)
        
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
            fs = self.active_viz_data["fs"]
            
            self.log("COMPUTING ERD/ERS PERCENTAGES...")
            
            self.erd_left, _ = percentage_ERD_ERS.calculate_erd_percent(
                self.smoothed_left, fs, tmin, tmax, (t_start, t_end))
            self.erd_right, _ = percentage_ERD_ERS.calculate_erd_percent(
                self.smoothed_right, fs, tmin, tmax, (t_start, t_end))
            
            plots = self.scroll_erd.get_plots()
            names = ['C3', 'Cz', 'C4']
            
            # String builder for the numeric values
            val_str = "=== AVERAGE ERD% DURING TASK (0s - 3s) ===\n"
            
            for i in range(3):
                p = plots[i]
                p.clear_plot()
                ax = p.get_axes()
                
                # Plot ERD % Curves
                ax.plot(self.time_axis_epochs, self.erd_left[i, :], color='cyan', linewidth=2.0, label='Left Hand')
                ax.plot(self.time_axis_epochs, self.erd_right[i, :], color='magenta', linewidth=2.0, label='Right Hand')
                
                ax.axhline(0, color='white', linewidth=1)
                ax.axvline(0, color='gray', linestyle=':')
                
                # Fill ERD (Negative area) for visual clarity (Right Hand ERD)
                ax.fill_between(self.time_axis_epochs, self.erd_right[i, :], 0, 
                                where=(self.erd_right[i, :] < 0), color='magenta', alpha=0.15)
                
                p.style_axes(ax, title=f"ERD/ERS PERCENTAGE: {names[i]}", 
                             xlabel="Time relative to Cue (s)", ylabel="Power Change (%)")
                ax.legend(loc='upper right')
                p.draw()
                
                # Calculation of Mean ERD during active period (0 to 3s)
                mask = (self.time_axis_epochs > 0) & (self.time_axis_epochs < 3)
                mean_l = np.mean(self.erd_left[i, mask])
                mean_r = np.mean(self.erd_right[i, mask])
                val_str += f"{names[i]}: Left={mean_l:.2f}% | Right={mean_r:.2f}%\n"
                
            self.txt_erd_vals.setText(val_str)
            self.log("ERD CALCULATION DONE.")
        except Exception as e:
            self.log(f"ERD ERROR: {e}")

    # TAB 8: FEATURE EXTRACTION (Scrollable with 6 Plots)
    def init_tab_features(self):
        tab = QWidget(); l = QVBoxLayout()
        btn = QPushButton("[ EXTRACT FEATURES (FROM ALL TRAINING FILES) ]")
        btn.clicked.connect(self.run_features)
        l.addWidget(btn)
        
        self.scroll_feats = ScrollableFeatureLayout()
        l.addWidget(self.scroll_feats)
        tab.setLayout(l); self.tabs.addTab(tab, "8. FEATURES")

    def run_features(self):
        if not self.datasets["Train"]:
            QMessageBox.warning(self, "Error", "No Training Files Loaded in Tab 1!")
            return
            
        try:
            self.log("AGGREGATING TRAINING DATA...")
            all_epochs = []
            all_labels = []
            
            # Loop through all loaded training files and aggregate epochs
            for d in self.datasets["Train"]:
                raw = d["data_3ch"] # Shape: (3, n_samples) -> (C3, Cz, C4)
                ev = d["events"]
                fs = d["fs"]
                
                # Extract epochs (0.5s to 3.5s post-cue)
                ep, lbl = self.ml_pipeline.prepare_data(raw, ev, fs, tmin=0.5, tmax=3.5)
                all_epochs.append(ep)
                all_labels.append(lbl)
                
            X_train_full = np.concatenate(all_epochs, axis=0)
            y_train_full = np.concatenate(all_labels, axis=0)
            
            self.log(f"TOTAL TRAINING SAMPLES: {X_train_full.shape[0]}")
            
            # Train CSP
            self.ml_pipeline.csp = csp_scratch.CSP_Scratch(2)
            self.ml_pipeline.csp.fit(X_train_full, y_train_full)
            csp_feats = self.ml_pipeline.csp.transform(X_train_full)
            
            # Temporal Features
            temp_ext = csp_scratch.TemporalFeatureExtractor()
            temp_feats = temp_ext.transform(X_train_full)
            
            # Store for ML
            self.ml_epochs = X_train_full
            self.ml_labels = y_train_full
            
            # 1. CSP Scatter Plot
            p_csp = self.scroll_feats.get_csp_plot()
            p_csp.clear_plot()
            ax = p_csp.get_axes()
            
            # Plot Class 0 (Left) vs Class 1 (Right)
            ax.scatter(csp_feats[y_train_full==0, 0], csp_feats[y_train_full==0, 1], 
                       color='cyan', label='Left Hand', alpha=0.7, edgecolors='white', s=40)
            ax.scatter(csp_feats[y_train_full==1, 0], csp_feats[y_train_full==1, 1], 
                       color='magenta', label='Right Hand', alpha=0.7, edgecolors='white', s=40)
            
            p_csp.style_axes(ax, title="CSP FEATURE SPACE (CLUSTERING)", 
                             xlabel="Log-Var (Comp 1)", ylabel="Log-Var (Comp 2)")
            ax.legend()
            p_csp.draw()
            
            # 2. Temporal Boxplots (5 Metrics) for ALL Channels (C3, Cz, C4)
            p_temps = self.scroll_feats.get_temporal_plots()
            metrics = ["Mean", "Variance", "StdDev", "Skewness", "Kurtosis"]
            
            # Iterate through each metric (Mean, Var, etc.)
            for i, p in enumerate(p_temps):
                p.clear_plot()
                ax = p.get_axes()
                
                # Calculate column indices in the flattened feature matrix
                # Structure: [Ch0_Metrics(0-4), Ch1_Metrics(5-9), Ch2_Metrics(10-14)]
                idx_c3 = (0 * 5) + i 
                idx_cz = (1 * 5) + i 
                idx_c4 = (2 * 5) + i
                
                # Extract Data for Boxplots
                d_c3_L = temp_feats[y_train_full==0, idx_c3]
                d_c3_R = temp_feats[y_train_full==1, idx_c3]
                d_cz_L = temp_feats[y_train_full==0, idx_cz]
                d_cz_R = temp_feats[y_train_full==1, idx_cz]
                d_c4_L = temp_feats[y_train_full==0, idx_c4]
                d_c4_R = temp_feats[y_train_full==1, idx_c4]
                
                data_to_plot = [d_c3_L, d_c3_R, d_cz_L, d_cz_R, d_c4_L, d_c4_R]
                tick_labels = ['L(C3)', 'R(C3)', 'L(Cz)', 'R(Cz)', 'L(C4)', 'R(C4)']
                
                # Create Boxplot
                box = ax.boxplot(data_to_plot, 
                                 tick_labels=tick_labels, 
                                 patch_artist=True,
                                 medianprops=dict(color="white"), 
                                 whiskerprops=dict(color="white"), 
                                 capprops=dict(color="white"),
                                 flierprops=dict(marker='o', markerfacecolor='white', markersize=3))
                
                # Color Coding: Cyan (Left), Magenta (Right)
                colors = ['cyan', 'magenta', 'cyan', 'magenta', 'cyan', 'magenta']
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                
                p.style_axes(ax, title=f"METRIC DISTRIBUTION: {metrics[i].upper()}", 
                             xlabel="Class (Channel)", ylabel="Feature Value")
                
                ax.grid(True, axis='y', color='#333', linestyle='--')
                p.draw()
                
            self.log("FEATURES EXTRACTED & VISUALIZED (C3, Cz, C4).")
            
        except Exception as e:
            self.log(f"FEATURE ERROR: {e}")
            import traceback
            traceback.print_exc()

    # TAB 9: MACHINE LEARNING & INFERENCE
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
        
        btn_infer = QPushButton("[ RUN INFERENCE ON EVAL FILE ]")
        btn_infer.clicked.connect(self.run_inference)
        
        h_ctrl.addWidget(btn_compare)
        h_ctrl.addWidget(self.combo_ml_plot)
        h_ctrl.addWidget(btn_infer)
        layout.addLayout(h_ctrl)
        
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
        
        # Sub-tab 4: Detailed Results Table (Validation)
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
        self.ml_tabs.addTab(self.tab_ml_details, "TEST PREDICTIONS")
        
        # Sub-tab 5: Inference Table (Eval Data)
        self.tab_ml_eval = QWidget()
        l_eval = QVBoxLayout(self.tab_ml_eval)
        self.table_eval = QTableWidget()
        self.table_eval.setColumnCount(2)
        self.table_eval.setHorizontalHeaderLabels(["Eval Trial ID", "Predicted Class"])
        self.table_eval.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_eval.setStyleSheet("gridline-color: magenta; color: white;")
        l_eval.addWidget(self.table_eval)
        self.ml_tabs.addTab(self.tab_ml_eval, "INFERENCE (EVAL DATA)")
        
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

    '''def run_ml_comparison(self):
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
            self.txt_ml_results.setText(f"ERROR: {e}")'''
    
    def run_ml_comparison(self):
        """
        Executes the Machine Learning Pipeline with Explicit Train/Test Split.
        1. Aggregates data from ALL loaded Training Files.
        2. Prepares data from the SINGLE Test File.
        3. Runs training and evaluation using 'run_full_comparison'.
        """
        # 1. Validation: Ensure files are loaded
        if not self.datasets["Train"]:
            QMessageBox.warning(self, "Pipeline Error", "No Training Files loaded. Please add files in Tab 1.")
            return
            
        if self.datasets["Test"] is None:
            QMessageBox.warning(self, "Pipeline Error", "No Test File set. Please set a Test File in Tab 1.")
            return

        self.log("PREPARING DATA FOR EXPLICIT TRAINING/TESTING...")
        
        try:
            # 2. Extract Sampling Frequency (fs) from the first training file
            # This fixes the AttributeError by getting 'fs' directly from the data source
            current_fs = self.datasets["Train"][0]["fs"]
            
            # 3. Prepare TRAINING Data (Aggregate all files in the Train list)
            train_epochs_list = []
            train_labels_list = []
            
            for d in self.datasets["Train"]:
                # Process each training file using 'train' mode (looks for 769/770)
                ep, lbl = self.ml_pipeline.prepare_data(
                    d["data_3ch"], 
                    d["events"], 
                    current_fs, 
                    tmin=0.5, 
                    tmax=3.5, 
                    mode='train'
                )
                train_epochs_list.append(ep)
                train_labels_list.append(lbl)
                
            # Concatenate all training epochs into one large array
            X_train = np.concatenate(train_epochs_list, axis=0)
            y_train = np.concatenate(train_labels_list, axis=0)
            
            # 4. Prepare TESTING Data (Single file)
            test_d = self.datasets["Test"]
            
            # Ensure Test file uses the same FS (though usually they are the same)
            test_fs = test_d["fs"]
            
            X_test, y_test = self.ml_pipeline.prepare_data(
                test_d["data_3ch"], 
                test_d["events"], 
                test_fs, 
                tmin=0.5, 
                tmax=3.5, 
                mode='train'
            )
            
            self.log(f"DATA SPLIT -> Train: {len(y_train)} samples | Test: {len(y_test)} samples")
            
            # Safety Check for empty data
            if len(y_train) == 0 or len(y_test) == 0:
                self.log("ERROR: No valid trials found in Training or Test set. Check Event Codes.")
                return

            # Update Status text
            self.txt_ml_results.setText(">> TRAINING IN PROGRESS... PLEASE WAIT.\n")
            QApplication.processEvents() # Force GUI update to show "Please Wait"

            # 5. Run Pipeline (Explicit Comparison)
            # We pass the extracted 'current_fs' here
            self.ml_metrics = self.ml_pipeline.run_full_comparison(
                X_train, y_train, 
                X_test_raw=X_test, 
                y_test_raw=y_test,
                fs=current_fs
            )
            
            # 6. Update Bar Chart Visualization
            self.plot_ml_bar.clear_plot()
            ax = self.plot_ml_bar.get_axes()
            
            names = list(self.ml_metrics.keys())
            # Convert decimal accuracy to percentage
            vals = [m['Accuracy'] * 100 for m in self.ml_metrics.values()]
            
            bars = ax.bar(names, vals, color='#00ff41', alpha=0.8)
            
            self.plot_ml_bar.style_axes(ax, title="ACCURACY COMPARISON (TRAIN vs TEST FILE)", ylabel="Accuracy (%)")
            
            # Fix Label Overlap and Cut-off using FixedLocator
            import matplotlib.ticker as ticker
            ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(len(names))))
            ax.set_xticklabels(names, rotation=45, ha='right')
            
            # Adjust bottom margin so labels are not cut off
            self.plot_ml_bar.fig.subplots_adjust(bottom=0.30)
            
            # Add Value Labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, 
                        f'{height:.1f}%', 
                        ha='center', va='bottom', color='white', fontsize=9)
            
            self.plot_ml_bar.draw()
            
            # 7. Update Dropdown and Text Report
            self.combo_ml_plot.clear() 
            self.combo_ml_plot.addItem("Select Model to View...")
            self.combo_ml_plot.addItems(names)
            
            # Select Best Model Automatically
            if self.ml_metrics:
                best_model = max(self.ml_metrics, key=lambda k: self.ml_metrics[k]['Accuracy'])
                self.combo_ml_plot.setCurrentText(best_model)
            
            # Generate Text Report
            report = f">> TRAINING COMPLETE.\n"
            report += f">> Train Size: {len(y_train)} trials | Test Size: {len(y_test)} trials\n"
            report += "="*70 + "\n"
            report += f"{'MODEL':<25} | {'ACC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8}\n"
            report += "-"*70 + "\n"
            
            for name, m in self.ml_metrics.items():
                report += f"{name:<25} | {m['Accuracy']*100:.1f}     | {m['Precision']*100:.1f}     | {m['Recall']*100:.1f}     | {m['F1']:.2f}\n"
                
            self.txt_ml_results.setText(report)
            self.log("ML PIPELINE FINISHED SUCCESSFULLY.")
            
        except Exception as e:
            self.log(f"ML TRAINING ERROR: {e}")
            import traceback
            traceback.print_exc()

    '''def run_inference(self):
        if self.datasets["Eval"] is None:
            QMessageBox.warning(self, "Error", "No Evaluation File Loaded in Tab 1.")
            return
            
        model_name = self.combo_ml_plot.currentText()
        if "Select" in model_name: 
            QMessageBox.warning(self, "Error", "Please select a trained model from the dropdown first.")
            return
        
        self.log(f"INFERENCING using {model_name}...")
        
        # Prepare Eval Data (Mode='inference' to find 783)
        eval_d = self.datasets["Eval"]
        try:
            X_eval, _ = self.ml_pipeline.prepare_data(eval_d["data_3ch"], eval_d["events"], eval_d["fs"], mode='inference')
            
            if len(X_eval) == 0:
                self.log("WARN: No 'Unknown' trials found in Eval file.")
                return
                
            preds = self.ml_pipeline.predict_new_data(X_eval, model_name)
            
            # Fill Table
            self.table_eval.setRowCount(len(preds))
            for i, p in enumerate(preds):
                self.table_eval.setItem(i, 0, QTableWidgetItem(str(i+1)))
                self.table_eval.setItem(i, 1, QTableWidgetItem(p))
                
            self.ml_tabs.setCurrentWidget(self.tab_ml_eval)
            self.log("INFERENCE DONE.")
        except Exception as e:
            self.log(f"INFERENCE ERROR: {e}")'''
    
    def run_inference(self):
        """
        Runs inference on the Evaluation File (Unlabeled) using the selected model.
        """
        # 1. Validation
        if self.datasets["Eval"] is None:
            QMessageBox.warning(self, "Error", "No Evaluation File Loaded in Tab 1.")
            return
            
        model_name = self.combo_ml_plot.currentText()
        if "Select" in model_name or not model_name: 
            QMessageBox.warning(self, "Error", "Please select a trained model from the dropdown first.")
            return
        
        self.log(f"RUNNING INFERENCE on {self.datasets['Eval']['name']} using {model_name}...")
        
        try:
            # 2. Prepare Evaluation Data
            # CRITICAL: mode='inference' tells prepare_data to look for event 783 (Cue Unknown)
            eval_d = self.datasets["Eval"]
            X_eval, _ = self.ml_pipeline.prepare_data(
                eval_d["data_3ch"], 
                eval_d["events"], 
                eval_d["fs"], 
                tmin=0.5, 
                tmax=3.5, 
                mode='inference'
            )
            
            if len(X_eval) == 0:
                self.log("WARN: No 'Unknown' trials (Event 783) found in Eval file.")
                return
                
            # 3. Predict
            preds = self.ml_pipeline.predict_new_data(X_eval, model_name)
            
            # 4. Populate Inference Table
            self.table_eval.setRowCount(len(preds))
            for i, p in enumerate(preds):
                # Trial ID
                self.table_eval.setItem(i, 0, QTableWidgetItem(str(i+1)))
                # Predicted Class
                item_pred = QTableWidgetItem(p)
                # Color code for readability
                if p == "Left":
                    item_pred.setForeground(QColor("cyan"))
                else:
                    item_pred.setForeground(QColor("magenta"))
                self.table_eval.setItem(i, 1, item_pred)
                
            # Switch to the Inference Tab
            self.ml_tabs.setCurrentWidget(self.table_eval) # Ensure variable matches your layout (tab_ml_eval vs table_eval parent)
            self.log(f"INFERENCE COMPLETE. Predicted {len(preds)} trials.")
            
        except Exception as e:
            self.log(f"INFERENCE ERROR: {e}")
    
    '''def update_ml_visuals(self):
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
                from sklearn.model_selection import learning_curve
                model = self.ml_pipeline.trained_models[model_name]
                ax_perf = self.plot_ml_perf.get_axes()
                
                try:
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, self.ml_pipeline.X_train, self.ml_pipeline.y_train, 
                        cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5) # Increased to 0.1 to avoid errors
                    )
                    train_mean = np.mean(train_scores, axis=1) * 100
                    test_mean = np.mean(test_scores, axis=1) * 100
                    
                    ax_perf.plot(train_sizes, train_mean, 'o-', color="cyan", label="Training")
                    ax_perf.plot(train_sizes, test_mean, 'o-', color="magenta", label="Validation")
                    ax_perf.legend()
                    self.plot_ml_perf.style_axes(ax_perf, title=f"LEARNING CURVE: {model_name}", 
                                                 xlabel="Training Samples", ylabel="Accuracy (%)")
                    self.plot_ml_perf.draw()
                except ValueError as ve:
                    ax_perf.text(0.5, 0.5, "Insufficient Data for Curve", color='white', ha='center')
                    self.plot_ml_perf.draw()

            # 3. Update Detailed Table
            details = self.ml_pipeline.get_detailed_predictions(model_name)
            
            # FIX: Ensure we use the correct variable name 'table_ml_details'
            self.table_ml_details.setRowCount(len(details))
            for i, (tid, true_l, pred_l, status) in enumerate(details):
                self.table_ml_details.setItem(i, 0, QTableWidgetItem(str(tid)))
                self.table_ml_details.setItem(i, 1, QTableWidgetItem(true_l))
                self.table_ml_details.setItem(i, 2, QTableWidgetItem(pred_l))
                
                item_status = QTableWidgetItem(status)
                if status == "CORRECT":
                    item_status.setForeground(QColor("#00ff41")) # Green
                    item_status.setBackground(QColor("#003300"))
                else:
                    item_status.setForeground(QColor("#ff3333")) # Red
                    item_status.setBackground(QColor("#330000"))
                self.table_ml_details.setItem(i, 3, item_status)
                
        except Exception as e:
            self.log(f"VISUAL UPDATE ERROR: {e}")'''
            
    def update_ml_visuals(self):
        """
        Updates the ML visualizations (Confusion Matrix, Curve, Table) 
        based on the selected model in the dropdown.
        """
        model_name = self.combo_ml_plot.currentText()
        if "Select" in model_name or model_name not in self.ml_metrics:
            return
            
        try:
            self.log(f"Updating visuals for: {model_name}")
            
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
            ax.set_xticklabels(classes, color='white')
            ax.set_yticklabels(classes, color='white')
            
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
            
            if "MLP" in model_name:
                # MLP Special Case: Loss Curve
                fig_loss = self.ml_pipeline.generate_loss_curve(model_name)
                if fig_loss:
                    model = self.ml_pipeline.trained_models[model_name]
                    ax_perf = self.plot_ml_perf.get_axes()
                    ax_perf.plot(model.loss_curve_, color='#00ff41', linewidth=2)
                    self.plot_ml_perf.style_axes(ax_perf, title=f"LOSS CURVE: {model_name}", 
                                                 xlabel="Epochs", ylabel="Loss")
                    self.plot_ml_perf.draw()
            else:
                # Standard Models: Learning Curve
                from sklearn.model_selection import learning_curve
                model = self.ml_pipeline.trained_models[model_name]
                ax_perf = self.plot_ml_perf.get_axes()
                
                # Wrap Learning Curve in Try-Except for Safety
                try:
                    # FIX: Start at 0.1 (10%) to ensure enough data for CV split
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, self.ml_pipeline.X_train, self.ml_pipeline.y_train, 
                        cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
                    )
                    train_mean = np.mean(train_scores, axis=1) * 100
                    test_mean = np.mean(test_scores, axis=1) * 100
                    
                    ax_perf.plot(train_sizes, train_mean, 'o-', color="cyan", label="Training")
                    ax_perf.plot(train_sizes, test_mean, 'o-', color="magenta", label="Validation")
                    ax_perf.legend()
                    self.plot_ml_perf.style_axes(ax_perf, title=f"LEARNING CURVE: {model_name}", 
                                                 xlabel="Training Samples", ylabel="Accuracy (%)")
                    self.plot_ml_perf.draw()
                except ValueError as ve:
                    # Handle cases with very small datasets
                    ax_perf.text(0.5, 0.5, "Insufficient Data for Curve\n(Need >20 samples per class)", 
                                 color='white', ha='center', va='center')
                    self.plot_ml_perf.draw()

            # 3. Update Detailed Predictions Table
            details = self.ml_pipeline.get_detailed_predictions(model_name)
            
            self.table_ml_details.setRowCount(len(details))
            for i, (tid, true_l, pred_l, status) in enumerate(details):
                self.table_ml_details.setItem(i, 0, QTableWidgetItem(str(tid)))
                self.table_ml_details.setItem(i, 1, QTableWidgetItem(true_l))
                self.table_ml_details.setItem(i, 2, QTableWidgetItem(pred_l))
                
                item_status = QTableWidgetItem(status)
                if status == "CORRECT":
                    item_status.setForeground(QColor("#00ff41")) # Green
                else:
                    item_status.setForeground(QColor("#ff3333")) # Red
                self.table_ml_details.setItem(i, 3, item_status)
                
        except Exception as e:
            self.log(f"VISUAL UPDATE ERROR: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EEGAnalysisWindow()
    window.show()
    sys.exit(app.exec())