"""
main.py

Purpose:
    - Strict Entry Point / Launcher.
    - Initializes Logging to .txt
    - Performs pre-flight checks.
"""

import sys
import os
import logger_util # Import modul logger baru

def system_preflight_check():
    dll_name = "eeg_processing.dll"
    if not os.path.exists(dll_name):
        print(f"[CRITICAL] Missing Core Library: '{dll_name}'")
        print(">> ACTION REQUIRED: Compile 'eeg_core.cpp' using g++.")
        return False
    return True

def main():
    # 1. Aktifkan Logger ke file .txt
    logger_util.setup_logging()

    # 2. Cek DLL
    if not system_preflight_check():
        print("[SYSTEM HALTED] Pre-flight check failed.")
        input("Press Enter to exit...")
        sys.exit(1)

    print(">> SYSTEM BOOT SEQUENCE INITIATED...")

    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QPalette, QColor
        import GUI
    except ImportError as e:
        print(f"[CRITICAL ERROR] Import Failed: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

    app = QApplication(sys.argv)

    # Set Global Dark Palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(13, 13, 13))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 255, 65))
    app.setPalette(palette)

    window = GUI.EEGAnalysisWindow()
    window.show()
    
    print(">> GUI LAUNCH SUCCESSFUL.")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()