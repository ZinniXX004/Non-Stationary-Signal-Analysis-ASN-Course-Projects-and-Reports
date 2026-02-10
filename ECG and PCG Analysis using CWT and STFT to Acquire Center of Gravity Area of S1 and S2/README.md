![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)
![Language](https://img.shields.io/badge/Language-Python-green)

# ECG and PCG Analysis using CWT and STFT

## 📝 1. Description
This folder contains Python scripts and the final report for the assignment: **"ECG and PCG Analysis using CWT and STFT to Acquire Center of Gravity Area of S1 and S2"**.

The main objective of this project is to analyze non-stationary biomedical signals (specifically Phonocardiogram/PCG and Electrocardiogram/ECG) to identify and localize the **S1 (Lub)** and **S2 (Dub)** heart sounds. The analysis compares two Time-Frequency Analysis methods:
* **Short-Time Fourier Transform (STFT)**
* **Continuous Wavelet Transform (CWT)**

By calculating the **Center of Gravity (CoG)** of the signal energy, the script attempts to precisely locate the timing of S1 and S2 events.

## 🛠️ 2. Prerequisites
To run the scripts in this folder, you will need the following dependencies:

### Environment
* **Python 3.8+**

### Libraries
* `numpy` (Numerical operations)
* `matplotlib` (Plotting signals and spectrograms/scalograms)
* `scipy` (Signal processing, STFT implementation)
* `PyWavelets` (or `pywt` for CWT implementation)
* `pandas` (Data handling)

### Dataset
The dataset used is the **PhysioNet/Computing in Cardiology Challenge 2016**.
* **Source:** [PhysioNet Challenge 2016](https://physionet.org/content/challenge-2016/1.0.0/)
* **Specific Usage:** Walking dataset (analyzing motion artifacts and heart rates).

## 🧠 3. Theory & Equations
The project relies on time-frequency analysis to handle the non-stationary nature of PCG signals.

### Short-Time Fourier Transform (STFT)
STFT divides the signal into short segments using a window function $$w(t)$$ and applies the Fourier Transform to each segment.

$$STFT \{x(t)\}(\tau, \omega) = \int_{-\infty}^{\infty} x(t) w(t-\tau) e^{-j\omega t} dt$$

### Continuous Wavelet Transform (CWT)
CWT decomposes the signal using a "mother wavelet" $$\psi(t)$$ that is scaled ($$a$$) and translated ($$b$$). It provides better time resolution for high frequencies and frequency resolution for low frequencies.

$$CWT_x(a,b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^* \left(\frac{t-b}{a}\right) dt$$

### Center of Gravity (CoG)
To find the precise location of the heart sounds (S1/S2), we calculate the center of gravity of the energy distribution in the selected time window:

$$CoG = \frac{\sum (t \cdot E(t))}{\sum E(t)}$$

*Where* $$E(t)$$ *is the energy of the signal at time* $$t$$*.*

## 🚀 4. How to Run
1.  **Clone the repository** and navigate to this folder.
2.  **Download the dataset** from the link in Prerequisites and place the relevant `.wav` or `.mat` file in a `data/` subfolder (or update the path in the script).
3.  **Run the analysis script:**

    ```bash
    python main.py
    ```
4.  **View Output:** The script will generate plots comparing the STFT spectrogram and CWT scalogram, marking the detected S1 and S2 regions.

## 📊 5. Evaluations
Based on the attached report:
* **Resolution:** CWT provided superior visualization of the S1 and S2 components compared to STFT. STFT suffered from the uncertainty principle (trade-off between time and frequency resolution), making the "smearing" effect visible.
* **CoG Accuracy:** The Center of Gravity method successfully estimated the centroids of the heart sound energies, providing a distinct timestamp for S1 and S2 even in the presence of noise.
* **Motion Artifacts:** The walking dataset introduced noise, but the CWT scales allowed for better isolation of the heart sound frequencies (typically 20-200 Hz) from high-frequency noise.

## 🤝 6. Contribution
Contributions are welcome! This project is free to use under the **MIT License**.
If you wish to contribute:
1.  Fork the repo.
2.  Create a branch (`git checkout -b feature/NewAlgorithm`).
3.  Commit your changes.
4.  Open a Pull Request.

## ⚠️ 7. Disclaimer
> **Educational & Research Purpose Only**
> This code was developed for a Biomedical Engineering course assignment at ITS. It is intended for educational demonstration and analysis of signal processing techniques. It should **not** be used for clinical diagnosis, medical decision-making, or as a medical device.