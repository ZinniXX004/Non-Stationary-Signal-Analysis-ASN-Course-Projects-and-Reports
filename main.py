"""
main.py

Purpose:
    - Entry point for the EEG Analysis System (Version 5.0).
    - Orchestrates system startup, logging, and dependency checks.
    - Launches the Main Graphical User Interface (GUI).
    - Ensures all critical components (DLLs, Libraries) are present before launch.

Dependencies:
    - sys, os
    - PyQt6
    - logger_util (Custom Logging Utility)
    - GUI (Main Application Logic)
"""

import sys
import os
import logger_util # Custom module to redirect output to .txt

def system_preflight_check():
    """
    Performs critical checks before loading the heavy GUI modules.
    
    Returns:
        bool: True if checks pass, False otherwise.
    """
    # 1. Check for the C++ Compiled Dynamic Link Library
    dll_name = "eeg_processing.dll"
    if not os.path.exists(dll_name):
        print(f"[CRITICAL ERROR] Core Library '{dll_name}' is missing!")
        print(">> ACTION REQUIRED: You must compile the C++ backend first.")
        print(">> COMMAND: g++ -O3 -shared -static -o eeg_processing.dll eeg_core.cpp")
        return False
    
    return True

def main():
    """
    Main execution function.
    """
    # 1. Initialize Logging System (Console + File)
    # This ensures that any errors occurring during import are saved to debug_output.txt
    logger_util.setup_logging()

    # 2. Run Pre-flight Checks
    if not system_preflight_check():
        print("[SYSTEM HALTED] Pre-flight checks failed.")
        print(">> Please resolve the errors above and restart the application.")
        input("Press Enter to exit...")
        sys.exit(1)

    print(">> SYSTEM BOOT SEQUENCE INITIATED...")
    print(">> Loading Python Libraries (PyQt6, NumPy, Scikit-Learn, Matplotlib)...")

    # 3. Import GUI and Qt Components
    # We wrap this in a try-except block because importing GUI.py triggers 
    # imports of all other modules (ML, CSP, etc.). If a library is missing, it fails here.
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QPalette, QColor
        import GUI
    except ImportError as e:
        print("\n[CRITICAL ERROR] Failed to import necessary modules.")
        print(f">> ERROR DETAILS: {e}")
        print(">> SUGGESTION: Ensure 'numpy', 'scipy', 'matplotlib', 'scikit-learn', and 'PyQt6' are installed.")
        input("Press Enter to exit...")
        sys.exit(1)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred during startup: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

    # 4. Initialize Qt Application
    app = QApplication(sys.argv)

    # 5. Apply Global Dark Palette (Base Theme)
    # While GUI.py handles specific styling, this ensures standard dialogs 
    # (like File Pickers) match the dark aesthetic.
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(13, 13, 13))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 255, 65))
    palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(26, 26, 26))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 255, 65))
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 255, 65))
    palette.setColor(QPalette.ColorRole.Button, QColor(13, 13, 13))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 255, 65))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 255, 65))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    
    app.setPalette(palette)

    # 6. Launch Main Window
    window = GUI.EEGAnalysisWindow()
    window.show()
    
    print(">> GUI LAUNCH SUCCESSFUL.")
    
    # 7. Start Event Loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()