import sys
import os

# Handle High DPI Scaling (Important to prevent blurry GUI on modern/Retina/4K screens)
# Must be done before importing QApplication
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from GUI import MainWindow
except ImportError as e:
    print("CRITICAL ERROR: Failed to load required libraries")
    print(f"Error Details: {e}")
    print("Make sure you have installed: PyQt6, numpy, matplotlib, wfdb")
    print("Also make sure the GUI.py file and other modules are in the same folder")
    sys.exit(1)

def main():
    print("=======================================================")
    print("   ASN - EMG Movement Signal Analysis (Physionet)      ")
    print("   Based on Wavelet-Based Assessment Method            ")
    print("=======================================================")
    print("[-] Initializing Application...")

    # 1. Creating Application Instance
    app = QApplication(sys.argv)
    
    # 2. Style Configuration
    # Using 'Fusion' as the base style because it's the most consistent across OSes (Windows/Mac/Linux)
    # before override it with the dark purple Stylesheet in GUI.py
    app.setStyle("Fusion")

    # 3. Application Attribute Configuration (High DPI Support)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    # 4. Load Main Window
    try:
        window = MainWindow()
        window.showMaximized() 
        print("[-] Application Started Successfully.")
    except Exception as e:
        print(f"[!] Error when GUI Initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. Running event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()