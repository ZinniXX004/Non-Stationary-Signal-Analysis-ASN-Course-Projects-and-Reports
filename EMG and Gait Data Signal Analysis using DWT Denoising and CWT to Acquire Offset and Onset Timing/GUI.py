import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QTabWidget, 
    QScrollArea, QLineEdit, QComboBox, QCheckBox, 
    QSpinBox, QDoubleSpinBox, QMessageBox, QFrame, 
    QGroupBox, QRadioButton, QButtonGroup, QTextEdit,
    QSplitter
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QFont, QPalette, QColor

# Matplotlib integration
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import Logic Modules
import Load_and_Plot_Raw_Data as ModulLoad
import Segmentation_Foot_Switch as ModulSeg
import Filtering_BPF as ModulFilter
import Denoising_DWT_EMG as ModulDenoise
import STFT_EMG as ModulSTFT
import CWT_EMG as ModulCWT
import Threshold as ModulThreshold
import Result_Reporting

# THEME CONFIGURATION (DARK VIOLET)
THEME_CSS = """
QMainWindow {
    background-color: #121212;
}
QWidget {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 10pt;
}
QTabWidget::pane {
    border: 1px solid #444;
    background-color: #1e1e2e;
    border-radius: 5px;
}
QTabBar::tab {
    background: #2c2c3e;
    color: #aaa;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}
QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #7e57c2, stop:1 #5e35b1);
    color: white;
    font-weight: bold;
}
QTabBar::tab:hover {
    background: #3d3d5c;
}
QPushButton {
    background-color: #6200ea;
    color: white;
    border-radius: 5px;
    padding: 8px 16px;
    font-weight: bold;
    border: none;
}
QPushButton:hover {
    background-color: #7c4dff;
}
QPushButton:pressed {
    background-color: #311b92;
}
QPushButton#Destructive {
    background-color: #c62828;
}
QPushButton#Destructive:hover {
    background-color: #e53935;
}
QGroupBox {
    border: 1px solid #7e57c2;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 15px;
    font-weight: bold;
    color: #b39ddb;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    left: 10px;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {
    background-color: #2c2c3e;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 5px;
    color: white;
}
QLineEdit:focus, QSpinBox:focus, QTextEdit:focus {
    border: 1px solid #7e57c2;
}
QScrollArea {
    border: none;
    background-color: #121212;
}
QScrollBar:vertical {
    border: none;
    background: #1e1e2e;
    width: 12px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #5e35b1;
    min-height: 20px;
    border-radius: 6px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QToolBar {
    background-color: #2c2c3e;
    border: none;
    spacing: 5px; 
}
QToolButton {
    background-color: #3d3d5c;
    border-radius: 3px;
    padding: 2px;
}
QToolButton:hover {
    background-color: #5e35b1;
}
QSplitter::handle {
    background-color: #444;
}
"""

# GRAPHICS HELPER CLASS (MATPLOTLIB CANVAS)
class MplCanvas(QWidget):
    """
    Wrapper Widget containing:
    1. Navigation Toolbar (Zoom, Pan, Save)
    2. FigureCanvas (Plot Area)
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100, is_3d=False):
        super().__init__(parent)
        
        # Initialize Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#1e1e2e') # Global figure background
        
        self.canvas = FigureCanvas(self.fig)
        self.figure = self.fig 
        
        # Initialize Axes first (default)
        self.axes = self.fig.add_subplot(111, projection='3d' if is_3d else None)
        
        # Setup Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("background-color: #2c2c3e; color: white;")
        
        # Layout Vertikal: Toolbar on top, Plot below
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Set Minimum Height
        self.setMinimumHeight(int(height * dpi) + 50) 
        
        # Apply theme initial
        self.style_axis(self.axes, is_3d)

    def style_axis(self, ax, is_3d=False):
        ax.set_facecolor('#1e1e2e')
        
        # Colors for Title and Labels
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        if is_3d: 
            ax.zaxis.label.set_color('white')
        
        # Colors for Ticks
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        if is_3d: 
            ax.tick_params(axis='z', colors='white')
        
        # Grid Lines
        ax.grid(True, linestyle='--', alpha=0.2, color='white')
        
        # Spines (Border lines) - 2D only
        if not is_3d:
            for spine in ax.spines.values():
                spine.set_color('white')
        else:
            # Pane walls for 3D
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.05))

    def format_legend(self, ax):
        """Formats legend with white text and dark background."""
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor('#2c2c3e')
            legend.get_frame().set_edgecolor('#555')
            for text in legend.get_texts():
                text.set_color("white")

    def format_colorbar(self, cbar):
        """Formats colorbar ticks and label to white."""
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label(cbar.ax.get_ylabel(), color='white')

# MAIN GUI CLASS (MAIN WINDOW)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("5023231017 - Jeremia Christ Immanuel Manalu - ASN - EMG Gait Analysis System (Physionet)")
        self.setGeometry(100, 100, 1350, 850)
        
        # Data Storage Variables
        self.raw_data = None
        self.segments = None 
        
        self.setup_ui()
        
    def setup_ui(self):
        # Apply Theme CSS
        app = QApplication.instance()
        app.setStyleSheet(THEME_CSS)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # --- Header ---
        header_layout = QHBoxLayout()
        title_label = QLabel("EMG Movement Signal Analysis Pipeline")
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #b39ddb;")
        
        btn_clear = QPushButton("Clear All Data")
        btn_clear.setObjectName("Destructive")
        btn_clear.clicked.connect(self.clear_all_data)
        
        btn_exit = QPushButton("Exit App")
        btn_exit.setObjectName("Destructive")
        btn_exit.clicked.connect(self.close)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(btn_clear)
        header_layout.addWidget(btn_exit)
        
        main_layout.addLayout(header_layout)
        
        # Tabs Container
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Initialize Tabs
        self.create_tab_load()
        self.create_tab_segmentation()
        self.create_tab_filtering()
        self.create_tab_denoising()
        self.create_tab_stft()
        self.create_tab_cwt()
        self.create_tab_threshold()
        
        # Status Bar
        self.status_label = QLabel("Status: Ready. Please load data.")
        self.statusBar().addWidget(self.status_label)

    # TAB 1: LOAD DATA
    def create_tab_load(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        group = QGroupBox("Step 1: Load Raw Data")
        form_layout = QHBoxLayout()
        self.txt_record = QLineEdit("S01")
        self.txt_record.setPlaceholderText("Record Name (e.g. S01)")
        
        btn_load = QPushButton("Load Data")
        btn_load.clicked.connect(self.process_load_data)
        
        form_layout.addWidget(QLabel("Record Name:"))
        form_layout.addWidget(self.txt_record)
        form_layout.addWidget(btn_load)
        group.setLayout(form_layout)
        
        content_layout.addWidget(group)
        
        # Area Plot
        self.plot_layout_raw = QVBoxLayout()
        content_layout.addLayout(self.plot_layout_raw)
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "1. Load Data")

    def process_load_data(self):
        rec_name = self.txt_record.text()
        self.status_label.setText(f"Loading {rec_name}...")
        QApplication.processEvents()
        
        self.raw_data = ModulLoad.load_raw_data(rec_name)
        
        if self.raw_data:
            self.status_label.setText(f"Loaded {rec_name} successfully.")
            self.plot_raw_data_ui()
            QMessageBox.information(self, "Success", "Data Loaded Successfully!\nPlease proceed to Segmentation tab.")
        else:
            self.status_label.setText("Failed to load data.")
            QMessageBox.critical(self, "Error", f"Could not load record '{rec_name}'. Check file existence.")

    def plot_raw_data_ui(self):
        # Clear previous plots
        while self.plot_layout_raw.count():
            child = self.plot_layout_raw.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            
        if not self.raw_data: return
        
        time = self.raw_data['time']
        
        # 3 Separate Canvases
        
        # 1. Canvas Foot Switch
        canvas1 = MplCanvas(self, width=5, height=4, dpi=100)
        ax1 = canvas1.axes
        ax1.plot(time, self.raw_data['signal_fs'], color='#e0e0e0', linewidth=0.8)
        ax1.set_title("Foot Switch (Channel 7)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (mV)")
        canvas1.style_axis(ax1)
        canvas1.fig.tight_layout()
        self.plot_layout_raw.addWidget(canvas1)
        
        # 2. Canvas Gastrocnemius
        canvas2 = MplCanvas(self, width=5, height=4, dpi=100)
        ax2 = canvas2.axes
        ax2.plot(time, self.raw_data['signal_gl'], color='#4fc3f7', linewidth=0.8)
        ax2.set_title("Gastrocnemius Lateralis (Channel 10)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude (mV)")
        canvas2.style_axis(ax2)
        canvas2.fig.tight_layout()
        self.plot_layout_raw.addWidget(canvas2)
        
        # 3. Canvas Vastus
        canvas3 = MplCanvas(self, width=5, height=4, dpi=100)
        ax3 = canvas3.axes
        ax3.plot(time, self.raw_data['signal_vl'], color='#ff8a65', linewidth=0.8)
        ax3.set_title("Vastus Lateralis (Channel 13)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Amplitude (mV)")
        canvas3.style_axis(ax3)
        canvas3.fig.tight_layout()
        self.plot_layout_raw.addWidget(canvas3)

    # TAB 2: SEGMENTATION
    def create_tab_segmentation(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        group = QGroupBox("Step 2: Segmentation & Cycle Selection")
        vlo = QVBoxLayout()
        
        # Row 1: Run Button
        row1 = QHBoxLayout()
        btn_seg = QPushButton("Run Segmentation")
        btn_seg.clicked.connect(self.process_segmentation)
        row1.addWidget(QLabel("Detect Toe Offs (TO-TO Cycles):"))
        row1.addWidget(btn_seg)
        row1.addStretch()
        
        # Row 2: Selector Siklus
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Select Cycle to Inspect:"))
        self.spin_cycle = QSpinBox()
        self.spin_cycle.setRange(1, 999)
        self.spin_cycle.setEnabled(False)
        self.spin_cycle.valueChanged.connect(self.plot_segmentation_ui) # Trigger on change
        row2.addWidget(self.spin_cycle)
        row2.addStretch()
        
        vlo.addLayout(row1)
        vlo.addLayout(row2)
        group.setLayout(vlo)
        
        content_layout.addWidget(group)
        self.plot_layout_seg = QVBoxLayout()
        content_layout.addLayout(self.plot_layout_seg)
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "2. Segmentation")
        
    def process_segmentation(self):
        if not self.raw_data:
            QMessageBox.warning(self, "Warning", "Please Load Data first.")
            return
            
        self.status_label.setText("Segmenting data...")
        self.segments = ModulSeg.segment_data(self.raw_data)
        
        if self.segments:
            num_cycles = len(self.segments)
            self.status_label.setText(f"Found {num_cycles} cycles.")
            self.spin_cycle.setEnabled(True)
            self.spin_cycle.setRange(1, num_cycles)
            # Set value 1 and call plot
            self.spin_cycle.setValue(1)
            if self.spin_cycle.value() == 1:
                self.plot_segmentation_ui()
        else:
            self.status_label.setText("Segmentation failed. No cycles found.")

    def plot_segmentation_ui(self):
        while self.plot_layout_seg.count():
            child = self.plot_layout_seg.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            
        if not self.segments: return
        
        selected_id = self.spin_cycle.value()
        if selected_id > len(self.segments) or selected_id < 1: return
        
        seg = self.segments[selected_id - 1]
        
        t_abs = seg['time'] 
        
        # 4 Separate Canvases
        
        # 1. Canvas Full Signal
        canvas1 = MplCanvas(self, width=5, height=4)
        ax1 = canvas1.axes
        ax1.plot(self.raw_data['time'], self.raw_data['signal_fs'], color='white', alpha=0.5, linewidth=0.5, label='Raw FS')
        ax1.axvspan(t_abs[0], t_abs[-1], color='yellow', alpha=0.3, label='Selected Cycle')
        ax1.set_title(f"Full Foot Switch Signal ({len(self.segments)} Cycles)")
        ax1.set_ylabel("Amplitude (V)")
        ax1.set_xlabel("Time (s)")
        ax1.legend(loc='upper right', fontsize='small')
        canvas1.format_legend(ax1)
        canvas1.style_axis(ax1)
        canvas1.fig.tight_layout()
        self.plot_layout_seg.addWidget(canvas1)
        
        # 2. Canvas Zoom Segment FS
        canvas2 = MplCanvas(self, width=5, height=4)
        ax2 = canvas2.axes
        
        ax2.plot(t_abs, seg['fs_segment'], color='#e0e0e0')
        ax2.set_title(f"Cycle {selected_id} - Foot Switch Segment")
        ax2.set_ylabel("Amplitude (V)")
        ax2.set_xlabel("Time (s) [Absolute]")
        
        # FIX: Logic for lines (Red=TO, Purple=Next TO, Cyan=HS)
        ax2.axvline(t_abs[0], color='#ff5252', linestyle='--', label='Toe Off (Start)')
        ax2.axvline(t_abs[-1], color='#e040fb', linestyle='--', label='Next Toe Off')
        
        if seg.get('hs_time_rel') is not None:
            # Calculate Absolute Time for Heel Strike (using offset)
            # The segmentation script passes 'hs_time_rel' which is relative to the start of the segment.
            # So we add it to t_abs[0]
            hs_time_abs = t_abs[0] + seg['hs_time_rel']
            ax2.axvline(hs_time_abs, color='#18ffff', linestyle=':', linewidth=2, label='Heel Strike')
            
        ax2.legend(loc='upper right')
        canvas2.format_legend(ax2)
        canvas2.style_axis(ax2)
        canvas2.fig.tight_layout()
        self.plot_layout_seg.addWidget(canvas2)
            
        # 3. Canvas Zoom GL
        canvas3 = MplCanvas(self, width=5, height=4)
        ax3 = canvas3.axes
        ax3.plot(t_abs, seg['gl_segment'], color='#4fc3f7')
        ax3.set_title(f"Cycle {selected_id} - Gastrocnemius (GL) Raw")
        ax3.set_ylabel("Amplitude (mV)")
        ax3.set_xlabel("Time (s) [Absolute]")
        
        ax3.axvline(t_abs[0], color='#ff5252', linestyle='--', alpha=0.5)
        ax3.axvline(t_abs[-1], color='#e040fb', linestyle='--', alpha=0.5)
        if seg.get('hs_time_rel') is not None:
             hs_time_abs = t_abs[0] + seg['hs_time_rel']
             ax3.axvline(hs_time_abs, color='#18ffff', linestyle=':', alpha=0.5)

        canvas3.style_axis(ax3)
        canvas3.fig.tight_layout()
        self.plot_layout_seg.addWidget(canvas3)
        
        # 4. Canvas Zoom VL
        canvas4 = MplCanvas(self, width=5, height=4)
        ax4 = canvas4.axes
        ax4.plot(t_abs, seg['vl_segment'], color='#ff8a65')
        ax4.set_title(f"Cycle {selected_id} - Vastus (VL) Raw")
        ax4.set_ylabel("Amplitude (mV)")
        ax4.set_xlabel("Time (s) [Absolute]")
        
        ax4.axvline(t_abs[0], color='#ff5252', linestyle='--', alpha=0.5)
        ax4.axvline(t_abs[-1], color='#e040fb', linestyle='--', alpha=0.5)
        if seg.get('hs_time_rel') is not None:
             hs_time_abs = t_abs[0] + seg['hs_time_rel']
             ax4.axvline(hs_time_abs, color='#18ffff', linestyle=':', alpha=0.5)
            
        canvas4.style_axis(ax4)
        canvas4.fig.tight_layout()
        self.plot_layout_seg.addWidget(canvas4)

    # TAB 3: FILTERING
    def create_tab_filtering(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        group = QGroupBox("Step 3: BPF Filtering (20-450 Hz)")
        vlo = QVBoxLayout()
        
        # Row 1: Method Selection
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Select Filter Method:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItem(ModulFilter.METHOD_STANDARD)
        self.combo_filter.addItem(ModulFilter.METHOD_RBJ)
        h1.addWidget(self.combo_filter)
        h1.addStretch()
        
        # Row 2: Apply Button
        btn_filt = QPushButton("Apply Selected BPF")
        btn_filt.clicked.connect(self.process_filtering)
        
        vlo.addLayout(h1)
        vlo.addWidget(btn_filt)
        group.setLayout(vlo)
        
        content_layout.addWidget(group)
        self.plot_layout_filt = QVBoxLayout()
        content_layout.addLayout(self.plot_layout_filt)
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "3. Filtering")

    def process_filtering(self):
        if not self.segments:
            QMessageBox.warning(self, "Warning", "Please run Segmentation first.")
            return
            
        method = self.combo_filter.currentText()
        self.status_label.setText(f"Filtering data using {method}...")
        QApplication.processEvents()
        
        # Apply Filter
        self.segments = ModulFilter.apply_bpf(self.segments, method=method)
        
        self.status_label.setText("Filtering Complete.")
        self.plot_filtering_ui(method)

    def plot_filtering_ui(self, current_method):
        while self.plot_layout_filt.count():
            child = self.plot_layout_filt.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            
        selected_id = self.spin_cycle.value() if self.segments else 1
        if selected_id > len(self.segments) or selected_id < 1: return
        
        seg = self.segments[selected_id - 1]
        t = seg['time'] 
        fs = seg['fs']
        
        # 3 Separate Canvases (Freq Resp + GL + VL)
        
        # 1. Frequency Response Comparison
        canvas_freq = MplCanvas(self, width=5, height=4)
        ax_f = canvas_freq.axes
        
        # Get data for both methods
        f1, mag1 = ModulFilter.get_frequency_response_data(fs, ModulFilter.METHOD_STANDARD)
        f2, mag2 = ModulFilter.get_frequency_response_data(fs, ModulFilter.METHOD_RBJ)
        
        # Plot Lines (Linear Scale X-Axis)
        line1, = ax_f.plot(f1, mag1, label='Standard BPF', color='cyan')
        line2, = ax_f.plot(f2, mag2, label='RBJ Cascade', color='magenta', linestyle='--')
        
        # Highlight selected method
        if current_method == ModulFilter.METHOD_STANDARD:
            line1.set_linewidth(2.5)
            line2.set_alpha(0.5)
        else:
            line2.set_linewidth(2.5)
            line1.set_alpha(0.5)
            
        ax_f.set_title("Frequency Response Comparison (Linear Magnitude)")
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("Magnitude (Linear)")
        ax_f.grid(True, which='both', linestyle='--', alpha=0.3)
        
        # Set X-Limit to show Passband (20-450) clearly
        ax_f.set_xlim(0, 600) 
        
        ax_f.legend(loc='lower right')
        
        canvas_freq.format_legend(ax_f)
        canvas_freq.style_axis(ax_f)
        canvas_freq.fig.tight_layout()
        self.plot_layout_filt.addWidget(canvas_freq)
        
        # 2. GL Filtered
        canvas_gl = MplCanvas(self, width=5, height=4)
        ax_gl = canvas_gl.axes
        ax_gl.plot(t, seg['gl_segment'], color='gray', alpha=0.5, label='Raw')
        ax_gl.plot(t, seg['gl_filtered'], color='#4fc3f7', label='Filtered (BPF)')
        ax_gl.set_title(f"Cycle {selected_id} - Gastrocnemius (GL) | Method: {current_method}")
        ax_gl.set_xlabel("Time (s)")
        ax_gl.set_ylabel("Amplitude (mV)")
        ax_gl.legend(loc='upper right')
        canvas_gl.format_legend(ax_gl)
        canvas_gl.style_axis(ax_gl)
        canvas_gl.fig.tight_layout()
        self.plot_layout_filt.addWidget(canvas_gl)
        
        # 3. VL Filtered
        canvas_vl = MplCanvas(self, width=5, height=4)
        ax_vl = canvas_vl.axes
        ax_vl.plot(t, seg['vl_segment'], color='gray', alpha=0.5, label='Raw')
        ax_vl.plot(t, seg['vl_filtered'], color='#ff8a65', label='Filtered (BPF)')
        ax_vl.set_title(f"Cycle {selected_id} - Vastus (VL) | Method: {current_method}")
        ax_vl.set_xlabel("Time (s)")
        ax_vl.set_ylabel("Amplitude (mV)")
        ax_vl.legend(loc='upper right')
        canvas_vl.format_legend(ax_vl)
        canvas_vl.style_axis(ax_vl)
        canvas_vl.fig.tight_layout()
        self.plot_layout_filt.addWidget(canvas_vl)

    # TAB 4: DENOISING
    def create_tab_denoising(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        group = QGroupBox("Step 4: DWT Denoising")
        vlo = QVBoxLayout()
        
        # Row 1: Method Selection
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Select Window Function:"))
        self.combo_window = QComboBox()
        self.combo_window.addItems(["Rectangular", "Hanning", "Hamming", "Blackman", "Triangular", "Kaiser"])
        h1.addWidget(self.combo_window)
        h1.addStretch()
        
        # Row 2: Buttons
        h2 = QHBoxLayout()
        btn_vis_window = QPushButton("Visualize Windows")
        btn_vis_window.clicked.connect(self.visualize_windows)
        
        btn_den = QPushButton("Apply DWT Denoising")
        btn_den.clicked.connect(self.process_denoising)
        
        h2.addWidget(btn_vis_window)
        h2.addWidget(btn_den)
        
        vlo.addLayout(h1)
        vlo.addLayout(h2)
        group.setLayout(vlo)
        
        content_layout.addWidget(group)
        self.plot_layout_den = QVBoxLayout()
        content_layout.addLayout(self.plot_layout_den)
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "4. Denoising")

    def visualize_windows(self):
        while self.plot_layout_den.count():
            child = self.plot_layout_den.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        N = 100
        # Generate dummy sine wave
        t = np.linspace(0, 1, N)
        sine_wave = np.sin(2 * np.pi * 5 * t)
        
        window_types = ["Rectangular", "Hanning", "Hamming", "Blackman", "Triangular", "Kaiser"]
        
        for w_type in window_types:
            window = ModulDenoise.ManualWindow.get_window(w_type, N)
            applied = sine_wave * window
            
            canvas = MplCanvas(self, width=5, height=3)
            ax = canvas.axes
            ax.plot(window, label='Window Shape', color='cyan')
            ax.plot(applied, label='Applied to Sine', color='yellow', linestyle='--')
            ax.set_title(f"{w_type} Window Visualization")
            ax.set_ylabel("Amplitude (mV)")
            ax.legend(loc='upper right')
            
            canvas.style_axis(ax)
            canvas.format_legend(ax)
            canvas.fig.tight_layout()
            self.plot_layout_den.addWidget(canvas)

    def process_denoising(self):
        if not self.segments: return
        if 'gl_filtered' not in self.segments[0]:
             QMessageBox.warning(self, "Warning", "Please run Filtering first.")
             return
             
        window_type = self.combo_window.currentText()
        self.status_label.setText(f"Denoising using {window_type} window...")
        QApplication.processEvents()
        
        self.segments = ModulDenoise.denoise_dwt(self.segments, window_type=window_type)
        self.status_label.setText("Denoising Complete.")
        self.plot_denoising_ui(window_type)

    def plot_denoising_ui(self, window_type):
        while self.plot_layout_den.count():
            child = self.plot_layout_den.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            
        selected_id = self.spin_cycle.value() if self.segments else 1
        if selected_id > len(self.segments) or selected_id < 1: return
        
        seg = self.segments[selected_id - 1]
        t = seg['time'] 
        
        # 2 Separate Canvases
        
        # 1. GL Denoised
        canvas1 = MplCanvas(self, width=5, height=5)
        ax1 = canvas1.axes
        ax1.plot(t, seg['gl_filtered'], color='gray', alpha=0.5, label='Filtered Input')
        ax1.plot(t, seg['gl_denoised'], color='#00e676', label=f'Denoised ({window_type})')
        ax1.set_title(f"Cycle {selected_id} - GL Denoised")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (mV)")
        ax1.legend(loc='upper right')
        canvas1.format_legend(ax1)
        canvas1.style_axis(ax1)
        canvas1.fig.tight_layout()
        self.plot_layout_den.addWidget(canvas1)
        
        # 2. VL Denoised
        canvas2 = MplCanvas(self, width=5, height=5)
        ax2 = canvas2.axes
        ax2.plot(t, seg['vl_filtered'], color='gray', alpha=0.5, label='Filtered Input')
        ax2.plot(t, seg['vl_denoised'], color='#00e676', label=f'Denoised ({window_type})')
        ax2.set_title(f"Cycle {selected_id} - VL Denoised")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude (mV)")
        ax2.legend(loc='upper right')
        canvas2.format_legend(ax2)
        canvas2.style_axis(ax2)
        canvas2.fig.tight_layout()
        self.plot_layout_den.addWidget(canvas2)

    # TAB 5: STFT (Gait Cycle Axis)
    def create_tab_stft(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        group = QGroupBox("Step 5: STFT Analysis")
        vlo = QVBoxLayout()
        
        # Row 1: Settings
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Window Size:"))
        self.spin_win_size = QSpinBox()
        self.spin_win_size.setRange(32, 2048)
        self.spin_win_size.setValue(256)
        h1.addWidget(self.spin_win_size)
        
        h1.addWidget(QLabel("Overlap:"))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 2047)
        self.spin_overlap.setValue(128)
        h1.addWidget(self.spin_overlap)
        h1.addStretch()
        
        # Row 2: Plot Settings
        h2 = QHBoxLayout()
        self.chk_db_stft = QCheckBox("Use Decibel (dB) Scale")
        self.chk_db_stft.setChecked(False) # Default False per request
        self.radio_2d_stft = QRadioButton("2D Contour")
        self.radio_3d_stft = QRadioButton("3D Surface")
        self.radio_2d_stft.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.radio_2d_stft)
        bg.addButton(self.radio_3d_stft)
        h2.addWidget(self.chk_db_stft)
        h2.addWidget(self.radio_2d_stft)
        h2.addWidget(self.radio_3d_stft)
        h2.addStretch()
        
        btn_stft = QPushButton("Compute & Plot STFT")
        btn_stft.clicked.connect(self.process_stft)
        vlo.addLayout(h1)
        vlo.addLayout(h2)
        vlo.addWidget(btn_stft)
        group.setLayout(vlo)
        
        content_layout.addWidget(group)
        self.plot_layout_stft = QVBoxLayout()
        content_layout.addLayout(self.plot_layout_stft)
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "5. STFT Analysis")

    def process_stft(self):
        if not self.segments: return
        
        win_size = self.spin_win_size.value()
        overlap = self.spin_overlap.value()
        
        # Simple validation
        if overlap >= win_size:
            QMessageBox.warning(self, "Invalid Parameter", "Overlap must be less than Window Size.")
            return

        self.status_label.setText(f"Computing STFT (Size={win_size}, Overlap={overlap})...")
        QApplication.processEvents()
        
        self.segments = ModulSTFT.compute_stft_for_segments(self.segments, window_size=win_size, overlap=overlap)
        self.status_label.setText("STFT Complete.")
        self.plot_stft_ui()

    def plot_stft_ui(self):
        while self.plot_layout_stft.count():
            child = self.plot_layout_stft.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            
        is_3d = self.radio_3d_stft.isChecked()
        use_db = self.chk_db_stft.isChecked()
        sel = self.spin_cycle.value() if self.segments else 1; seg = self.segments[sel - 1]
        
        # Loop for each muscle
        for muscle in ['GL', 'VL']:
            canvas = MplCanvas(self, width=5, height=6, is_3d=is_3d)
            ax = canvas.axes
            
            data = seg[f'stft_{muscle.lower()}']
            f, t, Z = data['f'], data['t'], data['Z']
            
            # Normalize Time to Gait Cycle (0-100%)
            if len(t) > 1:
                t_norm = (t - t[0]) / (t[-1] - t[0]) * 100 
            else:
                t_norm = t
            
            Z_plot = 20 * np.log10(Z + 1e-10) if use_db else Z
            lbl = "Magnitude (dB)" if use_db else "Magnitude"

            if is_3d:
                T, F = np.meshgrid(t_norm, f)
                # Using stride=1 to show grid resolution
                surf = ax.plot_surface(T, F, Z_plot, cmap='viridis', edgecolor='none', rstride=1, cstride=1)
                cbar = canvas.figure.colorbar(surf, ax=ax, label=lbl, pad=0.1)
                ax.view_init(elev=40, azim=-45)
            else:
                # Using pcolormesh with shading='flat'/auto to show grid blocks (pixels)
                mesh = ax.pcolormesh(t_norm, f, Z_plot, cmap='jet', shading='auto')
                cbar = canvas.figure.colorbar(mesh, ax=ax, label=lbl)
                
            canvas.style_axis(ax, is_3d)
            canvas.format_colorbar(cbar)
            
            ax.set_title(f"STFT {muscle} - Cycle {sel}")
            ax.set_xlabel("% Gait Cycle")
            ax.set_ylabel("Freq (Hz)")
            ax.set_ylim(0, 500)
            ax.set_xlim(0, 100)
            
            canvas.fig.tight_layout(pad=3.0, h_pad=5.0)
            self.plot_layout_stft.addWidget(canvas)

    # TAB 6: CWT (Gait Cycle Axis)
    def create_tab_cwt(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        group = QGroupBox("Step 6: CWT Time-Frequency Analysis")
        vlo = QVBoxLayout()
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Mother Wavelet:"))
        self.combo_wavelet = QComboBox()
        self.combo_wavelet.addItems(["Morlet (Analytic)", "db4 (Numerical)"])
        h1.addWidget(self.combo_wavelet)
        h1.addStretch()
        
        h2 = QHBoxLayout()
        self.chk_db_cwt = QCheckBox("Use dB Scale")
        self.chk_db_cwt.setChecked(True)
        self.radio_2d_cwt = QRadioButton("2D Contour")
        self.radio_3d_cwt = QRadioButton("3D Surface")
        self.radio_2d_cwt.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self.radio_2d_cwt)
        bg.addButton(self.radio_3d_cwt)
        h2.addWidget(self.chk_db_cwt)
        h2.addWidget(self.radio_2d_cwt)
        h2.addWidget(self.radio_3d_cwt)
        h2.addStretch()
        
        btn_cwt = QPushButton("Compute & Plot CWT")
        btn_cwt.clicked.connect(self.process_cwt)
        vlo.addLayout(h1)
        vlo.addLayout(h2)
        vlo.addWidget(btn_cwt)
        group.setLayout(vlo)
        
        content_layout.addWidget(group)
        self.plot_layout_cwt = QVBoxLayout()
        content_layout.addLayout(self.plot_layout_cwt)
        content_layout.addStretch()
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self.tabs.addTab(tab, "6. CWT Analysis")

    def process_cwt(self):
        if not self.segments: return
        w_type = "morlet" if "Morlet" in self.combo_wavelet.currentText() else "db4"
        self.status_label.setText(f"Computing CWT ({w_type})...")
        QApplication.processEvents()
        self.segments = ModulCWT.compute_cwt_for_segments(self.segments, wavelet_type=w_type)
        self.status_label.setText("CWT Complete.")
        self.plot_cwt_ui()

    def plot_cwt_ui(self):
        while self.plot_layout_cwt.count():
            child = self.plot_layout_cwt.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            
        is_3d = self.radio_3d_cwt.isChecked()
        use_db = self.chk_db_cwt.isChecked()
        sel = self.spin_cycle.value() if self.segments else 1; seg = self.segments[sel - 1]
        
        for muscle in ['GL', 'VL']:
            canvas = MplCanvas(self, width=5, height=6, is_3d=is_3d)
            ax = canvas.axes
            
            data = seg[f'cwt_{muscle.lower()}']
            f, t, E = data['f'], data['t'], data['E']
            
            # Normalize Time to Gait Cycle (0-100%)
            if len(t) > 1:
                t_norm = (t - t[0]) / (t[-1] - t[0]) * 100
            else:
                t_norm = t
            
            Z_plot = 10 * np.log10(E + 1e-10) if use_db else E
            lbl = "Power (dB)" if use_db else "Energy Density"

            if is_3d:
                T, F = np.meshgrid(t_norm, f)
                surf = ax.plot_surface(T, F, Z_plot, cmap='plasma', edgecolor='none')
                cbar = canvas.figure.colorbar(surf, ax=ax, label=lbl, pad=0.1)
                ax.view_init(elev=40, azim=-45)
            else:
                cf = ax.contourf(t_norm, f, Z_plot, levels=50, cmap='plasma')
                cbar = canvas.figure.colorbar(cf, ax=ax, label=lbl)
            
            canvas.style_axis(ax, is_3d)
            canvas.format_colorbar(cbar)
            
            ax.set_title(f"CWT {muscle} - Cycle {sel}")
            ax.set_xlabel("% Gait Cycle")
            ax.set_ylabel("Freq (Hz)")
            ax.set_xlim(0, 100)
            
            canvas.fig.tight_layout(pad=3.0, h_pad=5.0)
            self.plot_layout_cwt.addWidget(canvas)

    # TAB 7: THRESHOLD & REPORT (UPDATED with SPLIT LAYOUT)
    def create_tab_threshold(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Main Controls Group
        group_ctrl = QGroupBox("Step 7: Onset/Offset Detection & Reporting")
        vlo_ctrl = QVBoxLayout()
        
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Threshold (% Peak):"))
        self.spin_th = QDoubleSpinBox(); self.spin_th.setRange(0.1, 100.0); self.spin_th.setValue(1.0); self.spin_th.setSingleStep(0.1)
        h1.addWidget(self.spin_th); h1.addStretch()
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Mode:"))
        self.rad_th_2d = QRadioButton("2D + Shading"); self.rad_th_3d = QRadioButton("3D + Plane")
        self.rad_th_2d.setChecked(True)
        bg = QButtonGroup(self); bg.addButton(self.rad_th_2d); bg.addButton(self.rad_th_3d)
        h2.addWidget(self.rad_th_2d); h2.addWidget(self.rad_th_3d); h2.addStretch()
        
        btn = QPushButton("Detect & Visualize"); btn.clicked.connect(self.process_threshold)
        vlo_ctrl.addLayout(h1); vlo_ctrl.addLayout(h2); vlo_ctrl.addWidget(btn)
        group_ctrl.setLayout(vlo_ctrl)
        layout.addWidget(group_ctrl)
        
        # --- SPLIT VIEW: GRAPHS (Left) | REPORT (Right) ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top: Scroll Area for Plots
        scroll_plots = QScrollArea(); scroll_plots.setWidgetResizable(True)
        content_plots = QWidget(); self.plot_layout_th = QVBoxLayout(content_plots)
        content_plots.setLayout(self.plot_layout_th)
        scroll_plots.setWidget(content_plots)
        
        # Bottom: Text Report
        self.txt_report = QTextEdit()
        self.txt_report.setReadOnly(True)
        self.txt_report.setPlaceholderText("Analysis results will appear here...")
        self.txt_report.setStyleSheet("font-family: Consolas; font-size: 10pt; background-color: #1e1e2e; color: #00e676;")
        self.txt_report.setMinimumHeight(150) 
        
        splitter.addWidget(scroll_plots)
        splitter.addWidget(self.txt_report)
        splitter.setStretchFactor(0, 3) 
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "7. Threshold Detection")

    def process_threshold(self):
        if not self.segments: return
        if 'cwt_gl' not in self.segments[0]:
            QMessageBox.warning(self, "Warning", "Please run CWT first.")
            return
        
        th_val = self.spin_th.value() / 100.0
        self.status_label.setText(f"Detecting (TH={self.spin_th.value()}%)...")
        QApplication.processEvents()
        
        processed = []
        for seg in self.segments:
            new_seg = seg.copy(); fs = seg['fs']
            
            # GL
            prof_gl = ModulThreshold.get_envelope(seg['cwt_gl']['E'])
            acts_gl = ModulThreshold.detect_bursts(prof_gl, fs, threshold_ratio=th_val)
            res_gl = [{'start_idx': s, 'end_idx': e, 'start_t': seg['time'][s], 'end_t': seg['time'][e-1] if e<=len(seg['time']) else seg['time'][-1]} for s, e in acts_gl]
            new_seg['activations_gl'] = res_gl; new_seg['energy_profile_gl'] = prof_gl
            
            # VL
            prof_vl = ModulThreshold.get_envelope(seg['cwt_vl']['E'])
            acts_vl = ModulThreshold.detect_bursts(prof_vl, fs, threshold_ratio=th_val)
            res_vl = [{'start_idx': s, 'end_idx': e, 'start_t': seg['time'][s], 'end_t': seg['time'][e-1] if e<=len(seg['time']) else seg['time'][-1]} for s, e in acts_vl]
            new_seg['activations_vl'] = res_vl; new_seg['energy_profile_vl'] = prof_vl
            
            processed.append(new_seg)
            
        self.segments = processed
        self.status_label.setText("Detection Complete.")
        self.plot_threshold_ui()
        
        # GENERATE REPORT FOR CURRENT CYCLE
        sel = self.spin_cycle.value()
        current_seg = self.segments[sel-1]
        report_text = Result_Reporting.generate_cycle_report(current_seg)
        self.txt_report.setText(report_text)

    def plot_threshold_ui(self):
        while self.plot_layout_th.count(): self.plot_layout_th.takeAt(0).widget().deleteLater()
        
        sel = self.spin_cycle.value() if self.segments else 1; seg = self.segments[sel-1]
        th_pct = self.spin_th.value(); is_3d = self.rad_th_3d.isChecked()
        
        for muscle in ['GL', 'VL']:
            # 2 Separate Canvases per muscle (Top=CWT, Bottom=Binary)
            
            # Canvas A: CWT (Top)
            canvas_cwt = MplCanvas(self, width=5, height=5, is_3d=is_3d)
            ax_cwt = canvas_cwt.axes
            
            d = seg[f'cwt_{muscle.lower()}']
            f, t, E = d['f'], d['t'], d['E']
            
            if len(t) > 1:
                tn = (t - t[0]) / (t[-1] - t[0]) * 100
            else:
                tn = t
                
            peak = np.max(E)
            th_val = (th_pct / 100.0) * peak
            
            if is_3d:
                T, F = np.meshgrid(tn, f)
                canvas_cwt.axes.plot_surface(T, F, E, cmap='magma', alpha=0.9)
                
                # Plane
                xp, yp = np.meshgrid([0, 100], [f[0], f[-1]])
                zp = np.full_like(xp, th_val)
                canvas_cwt.axes.plot_surface(xp, yp, zp, color='cyan', alpha=0.3)
                canvas_cwt.axes.view_init(30, -45)
                
                # Add Colorbar
                cbar = canvas_cwt.figure.colorbar(canvas_cwt.axes.collections[0], ax=ax_cwt, pad=0.1)
                canvas_cwt.format_colorbar(cbar)
            else:
                cf = ax_cwt.contourf(tn, f, E, 50, cmap='magma')
                
                # Overlay Shading on CWT
                acts = seg[f'activations_{muscle.lower()}']
                for a in acts:
                    # Convert Abs time back to % for visualization
                    if len(t) > 1:
                        s = (a['start_t'] - t[0]) / (t[-1] - t[0]) * 100
                        e = (a['end_t'] - t[0]) / (t[-1] - t[0]) * 100
                    else:
                        s, e = 0, 100
                    ax_cwt.axvspan(s, e, color='cyan', alpha=0.2, hatch='//')
                    
                # Add Colorbar
                cbar = canvas_cwt.figure.colorbar(cf, ax=ax_cwt, label="Energy Density")
                canvas_cwt.format_colorbar(cbar)
            
            ax_cwt.set_title(f"{muscle} CWT & Threshold Plane")
            ax_cwt.set_ylabel("Freq (Hz)")
            if not is_3d: ax_cwt.set_xlabel("% Gait Cycle")
                
            canvas_cwt.style_axis(ax_cwt, is_3d)
            canvas_cwt.fig.tight_layout()
            self.plot_layout_th.addWidget(canvas_cwt)
            
            # Canvas B: Binary Square Wave (Bottom)
            canvas_bin = MplCanvas(self, width=5, height=3, is_3d=False)
            ax_bin = canvas_bin.axes
            
            # Construct Binary Signal (0 or 1)
            y_binary = np.zeros_like(tn)
            acts = seg[f'activations_{muscle.lower()}']
            
            # Map indices from activations to the binary array
            for a in acts:
                s_idx = a['start_idx']
                e_idx = a['end_idx']
                # Ensure indices are within bounds
                s_idx = max(0, min(s_idx, len(y_binary)-1))
                e_idx = max(0, min(e_idx, len(y_binary)))
                y_binary[s_idx:e_idx] = 1
            
            # Plot Step Function
            ax_bin.plot(tn, y_binary, color='cyan', label='Activation Status', linewidth=2)
            ax_bin.fill_between(tn, 0, y_binary, color='cyan', alpha=0.3)
            
            ax_bin.set_title(f"{muscle} Activation (Binary)")
            ax_bin.set_xlabel("% Gait Cycle")
            ax_bin.set_ylabel("State (0/1)")
            ax_bin.set_xlim(0, 100)
            ax_bin.set_ylim(-0.2, 1.2)
            ax_bin.set_yticks([0, 1])
            ax_bin.set_yticklabels(['Inactive', 'Active'])
            
            canvas_bin.style_axis(ax_bin)
            canvas_bin.fig.tight_layout()
            self.plot_layout_th.addWidget(canvas_bin)

    def clear_all_data(self):
        reply = QMessageBox.question(self, 'Confirmation', 
                                     "Are you sure you want to clear all data?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.raw_data = None
            self.segments = None
            self.spin_cycle.setEnabled(False)
            self.spin_cycle.setValue(1)
            
            for layout in [self.plot_layout_raw, self.plot_layout_seg, self.plot_layout_filt,
                           self.plot_layout_den, self.plot_layout_stft, self.plot_layout_cwt,
                           self.plot_layout_th]:
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            
            self.status_label.setText("All data cleared.")
            self.tabs.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())