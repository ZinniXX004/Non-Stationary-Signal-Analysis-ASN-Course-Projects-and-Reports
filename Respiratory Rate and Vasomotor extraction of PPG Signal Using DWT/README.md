# !\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# !\[Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)

# !\[Language](https://img.shields.io/badge/Language-Python-green)

# 

# \# Photoplethysmography (PPG) Analysis for Stress Detection

# 

# \## 📝 1. Description

# This folder contains the Python implementation and final report for the combined \*\*Assignment 4 \& 5\*\*: \*\*"Photoplethysmography Signal Analysis for Stress Analysis Using DWT and EMD Algorithm"\*\*.

# 

# The primary objective is to extract physiological stress markers—specifically \*\*Breath Rate\*\*, \*\*Vasomotor Activity\*\*, and \*\*Heart Rate Variability (HRV)\*\*—from PPG signals collected from our own group members ("Group 5").

# 

# This project compares two advanced signal processing techniques to handle the non-stationary nature of PPG data:

# 1\.  \*\*Discrete Wavelet Transform (DWT):\*\* Used for multi-resolution analysis to denoise and decompose signal components.

# 2\.  \*\*Empirical Mode Decomposition (EMD):\*\* A data-adaptive method used to break down the signal into Intrinsic Mode Functions (IMFs).

# 

# \## 🛠️ 2. Prerequisites

# To replicate this analysis, you will need the following environment configuration:

# 

# \### Environment

# \* \*\*Python 3.8+\*\*

# 

# \### Libraries

# \* `numpy` (Numerical computing)

# \* `scipy` (Signal processing, peak detection)

# \* `matplotlib` (Visualization of IMFs and decomposition levels)

# \* `PyWavelets` (Required for \*\*DWT\*\* method)

# \* `EMD-signal` or `PyEMD` (Required for \*\*EMD\*\* method)

# \* `pandas` (Data management)

# 

# \### Dataset

# \* \*\*Source:\*\* Custom dataset located in the `data input/` folder.

# \* \*\*Content:\*\* Raw PPG recordings from group members (including yourself) and friends.

# \* \*\*Format:\*\* `.csv` or `.txt` files containing time-series amplitude data.

# 

# \## 🧠 3. Theory \& Equations

# Since this project applies two distinct methods, the theory is divided accordingly.

# 

# \### Method A: Discrete Wavelet Transform (DWT)

# DWT is used to separate the PPG signal into approximation ($cA$) and detail ($cD$) coefficients. We typically use the \*\*Daubechies (db4)\*\* or \*\*Symlet\*\* wavelet to denoise the signal and isolate the respiratory component (Breath Rate) found in lower frequency approximations.

# 

# \*\*DWT Decomposition Equation:\*\*

# 

# $$x\[n] = \\sum\_{k} cA\_{j,k} \\phi\_{j,k}\[n] + \\sum\_{j} \\sum\_{k} cD\_{j,k} \\psi\_{j,k}\[n]$$

# 

# \### Method B: Empirical Mode Decomposition (EMD)

# EMD decomposes the nonlinear and non-stationary PPG signal into a finite set of \*\*Intrinsic Mode Functions (IMFs)\*\* without a predefined basis function. High-frequency IMFs typically contain noise, while lower-order IMFs contain the cardiac and respiratory information.

# 

# \*\*Sifting Process:\*\*

# The signal $x(t)$ is represented as:

# 

# $$x(t) = \\sum\_{i=1}^{n} IMF\_i(t) + r\_n(t)$$

# 

# \*Where\* $$IMF\_i(t)$$ \*is the\* $$i$$\*-th intrinsic mode function and\* $$r\_n(t)$$ \*is the final residue.\*

# 

# \### Physiological Parameters

# \* \*\*Breath Rate:\*\* Derived from the respiratory-induced intensity variation (RIIV) in the PPG signal (typically 0.15 - 0.4 Hz).

# \* \*\*Vasomotor Activity:\*\* Low-frequency fluctuations (approx. 0.05 - 0.15 Hz) related to sympathetic nervous system activity.

# \* \*\*HRV:\*\* Calculated from the peak-to-peak intervals (PPI) of the clean PPG signal.

# 

# \## 🚀 4. How to Run

# Ensure your data is placed in the `data input/` folder before running.

# 

# \### To Run DWT Analysis

# 1\.  Navigate to the directory.

# 2\.  Run the DWT-specific script:

# 

# &nbsp;   ```bash

# &nbsp;   python main\_GUI.py

# &nbsp;   ```

# &nbsp;   \*This will output the denoised signal and the decomposed frequency bands.\*

# 

# \### To Run EMD Analysis

# 1\.  Run the EMD-specific script:

# 

# &nbsp;   ```bash

# &nbsp;   python ppg\_emd\_GUI.py

# &nbsp;   ```

# &nbsp;   \*This will generate plots showing the original signal and its constituent IMFs.\*

# 

# \## 📊 5. Evaluations

# Based on the \*\*Group 5 Final Report\*\*:

# \* \*\*Noise Reduction:\*\* \*\*DWT\*\* proved highly effective for removing high-frequency noise and baseline wander using thresholding on detail coefficients.

# \* \*\*Adaptive Decomposition:\*\* \*\*EMD\*\* excelled at isolating the respiratory component (Breath Rate) into specific IMFs without needing a fixed cutoff frequency, making it robust for subjects with varying heart rates.

# \* \*\*Stress Correlation:\*\* The extracted HRV parameters (SDNN, RMSSD) and Vasomotor activity showed a correlation with the subjects' reported stress states during data collection.

# 

# \## 🤝 6. Contribution

# Contributions are welcome! This project is free to use under the \*\*MIT License\*\*.

# 1\.  Fork the repo.

# 2\.  Create a branch (`git checkout -b feature/NewStressMetric`).

# 3\.  Commit your changes.

# 4\.  Open a Pull Request.

# 

# \## ⚠️ 7. Disclaimer

# > \*\*Educational \& Research Purpose Only\*\*

# > This code was developed for the \*\*Non-Stationary Signal Analysis (ASN)\*\* course at ITS (Biomedical Engineering). It is intended for educational demonstration and is \*\*not\*\* a clinical diagnostic tool for stress or cardiovascular health.

