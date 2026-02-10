# !\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# !\[Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)

# !\[Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B%20%7C%20Pascal-green)

# 

# \# Non-Stationary Signal Analysis (ASN) Course Repository

# 

# \## 🚩 Introduction

# This repository contains the complete collection of assignments and final projects for the \*\*Non-Stationary Signal Analysis (ASN)\*\* course at the \*\*Department of Biomedical Engineering, Institut Teknologi Sepuluh Nopember (ITS)\*\*.

# 

# The primary focus of this codebase is the advanced processing of biomedical signals that exhibit time-varying frequency content (non-stationary). The projects explore various Time-Frequency Analysis (TFA) techniques to extract meaningful physiological features from \*\*ECG\*\*, \*\*PCG\*\*, \*\*EMG\*\*, \*\*EEG\*\*, and \*\*PPG\*\* signals.

# 

# \## 📂 Assignment Modules

# 

# \### 1. 🫀 ECG and PCG Analysis (STFT vs. CWT)

# \*\*Objective:\*\* Localization of S1 (Lub) and S2 (Dub) heart sounds using center-of-gravity estimation.

# \* \*\*Techniques:\*\*

# &nbsp;   \* \*\*Short-Time Fourier Transform (STFT):\*\* Investigated the time-frequency trade-off (Heisenberg uncertainty).

# &nbsp;   \* \*\*Continuous Wavelet Transform (CWT):\*\* Utilized Morlet wavelets for superior temporal resolution of high-frequency transients.

# \* \*\*Dataset:\*\* PhysioNet Challenge 2016 (Walking/Motion Artifacts).

# 

# \### 2. 🦵 Gait Phase and EMG Analysis

# \*\*Objective:\*\* Correlation of muscle activation bursts with gait cycle phases (Heel-Strike, Toe-Off).

# \* \*\*Techniques:\*\*

# &nbsp;   \* \*\*Discrete Wavelet Transform (DWT):\*\* Employed `db4`/`sym5` wavelets for multi-resolution denoising of raw Surface EMG.

# &nbsp;   \* \*\*CWT Scalograms:\*\* Mapped the Power Spectral Density (PSD) to identify the active frequency range (30-150 Hz) during the stance phase.

# \* \*\*Dataset:\*\* PhysioNet sEMG Database.

# 

# \### 3. 🧠 BCI Motor Imagery \& EEG Feature Extraction

# \*\*Objective:\*\* Classification of Left vs. Right hand motor imagery tasks for Brain-Computer Interfaces.

# \* \*\*Techniques:\*\*

# &nbsp;   \* \*\*Hybrid Architecture:\*\* High-performance filtering core written in \*\*C++\*\* combined with Python ML pipelines.

# &nbsp;   \* \*\*Common Spatial Patterns (CSP):\*\* Maximized spatial variance discrimination between classes.

# &nbsp;   \* \*\*Machine Learning:\*\* Linear Discriminant Analysis (LDA) and SVM for classification.

# \* \*\*Dataset:\*\* BCI Competition IV (Dataset 2a/2b).

# 

# \### 4. 🫁 Photoplethysmography (PPG) Stress Analysis

# \*\*Objective:\*\* Extraction of autonomic nervous system markers that are Breath Rate, Vasomotor Activity, and Heart Rate Variability (HRV), from Group 5's custom dataset. This module is divided into two distinct algorithmic approaches:

# 

# \#### 4.1 📉 Discrete Wavelet Transform (DWT) Method

# \* \*\*Approach:\*\* Decomposed the PPG signal into approximation ($cA$) and detail ($cD$) coefficients using Daubechies wavelets.

# \* \*\*Key Result:\*\* Successfully isolated the respiratory component in the lower frequency approximation levels, allowing for robust breath rate estimation despite baseline wander.

# 

# \#### 4.2 🌊 Empirical Mode Decomposition (EMD) Method

# \* \*\*Approach:\*\* Applied a data-adaptive sifting process to decompose the nonlinear PPG signal into \*\*Intrinsic Mode Functions (IMFs)\*\*.

# \* \*\*Key Result:\*\* Demonstrated superior adaptability to inter-subject variability by isolating respiratory and cardiac components into separate IMFs without fixed cutoff frequencies.

# 

# \## 🛠️ Tech Stack \& Prerequisites

# The projects in this repository are built primarily in \*\*Python\*\*, with performance-critical sections in \*\*C++\*\* and legacy modules in \*\*Pascal\*\*.

# 

# \* \*\*Languages:\*\* Python 3.8+, C++ (GCC/MinGW), Free Pascal.

# \* \*\*Core Libraries:\*\*

# &nbsp;   \* `numpy` \& `scipy`: Numerical computing and DSP algorithms.

# &nbsp;   \* `PyWavelets`: DWT and CWT implementation.

# &nbsp;   \* `mne`: EEG data handling.

# &nbsp;   \* `scikit-learn`: Machine learning classifiers.

# &nbsp;   \* `matplotlib` \& `pandas`: Visualization and data management.

# 

# \## 🚀 Usage

# To inspect a specific assignment, navigate to its respective folder. Each directory contains its own `README.md` with detailed instructions on how to run the scripts and the theory behind them.

# 

# ```bash

# \# Clone the repository

# git clone \[https://github.com/yourusername/ASN-Course-Repo.git](https://github.com/yourusername/ASN-Course-Repo.git)

# 

# \# Navigate to a specific module (example)

# cd "BCI Motor Imagery Analysis"

# 

# \# Follow specific instructions in that folder's README

# ```

# 

# \## 📜 License

# This project is open-source and available under the \*\*MIT License\*\*.

# 

# \## ⚠️ Disclaimer

# > Educational Purpose Only

# > These algorithms were developed for academic assessment and research demonstration within the Biomedical Engineering curriculum. They are not intended for clinical use, medical diagnosis, or as certified medical software.

