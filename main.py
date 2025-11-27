import sys
import os

# Menangani High DPI Scaling (Penting agar GUI tidak buram di layar modern/Retina/4K)
# Harus dilakukan sebelum mengimport QApplication
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from GUI import MainWindow
except ImportError as e:
    print("CRITICAL ERROR: Gagal memuat library yang dibutuhkan.")
    print(f"Detail Error: {e}")
    print("Pastikan Anda telah menginstall: PyQt6, numpy, matplotlib, wfdb")
    print("Pastikan juga file: GUI.py dan modul lainnya berada dalam satu folder.")
    sys.exit(1)

def main():
    """
    Fungsi utama untuk menjalankan aplikasi ASN EMG Analysis.
    """
    print("=======================================================")
    print("   ASN - EMG Movement Signal Analysis (Physionet)      ")
    print("   Based on Wavelet-Based Assessment Method            ")
    print("=======================================================")
    print("[-] Initializing Application...")

    # 1. Membuat Instance Aplikasi
    app = QApplication(sys.argv)
    
    # 2. Konfigurasi Style
    # Menggunakan 'Fusion' sebagai base style karena paling konsisten lintas OS (Windows/Mac/Linux)
    # sebelum kita menimpanya dengan Stylesheet ungu gelap di GUI.py
    app.setStyle("Fusion")

    # 3. Konfigurasi Atribut Aplikasi (High DPI Support)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    # 4. Memuat Jendela Utama
    try:
        window = MainWindow()
        # Tampilkan secara Maximized agar grafik terlihat jelas dan besar
        window.showMaximized() 
        print("[-] Application Started Successfully.")
    except Exception as e:
        print(f"[!] Error saat inisialisasi GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 5. Menjalankan Event Loop (Infinite Loop sampai user menutup aplikasi)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()