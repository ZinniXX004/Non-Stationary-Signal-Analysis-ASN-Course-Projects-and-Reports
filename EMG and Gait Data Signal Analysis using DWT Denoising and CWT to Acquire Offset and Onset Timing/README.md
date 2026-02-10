![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)
![Language](https://img.shields.io/badge/Language-Python-green)

# Gait Phase and EMG Analysis using DWT and CWT

## 📝 1. Description
This folder contains the Python implementation and final report for the assignment: **"Gait Phase and EMG Analysis using DWT Denoising and CWT to Acquire Muscle-Activation Frequency Range"**.

The goal of this project is to analyze **Surface Electromyography (sEMG)** signals collected from lower limb muscles during walking. The pipeline involves:
1.  **Denoising:** Using **Discrete Wavelet Transform (DWT)** to remove noise and artifacts from raw EMG data.
2.  **Frequency Analysis:** Applying **Continuous Wavelet Transform (CWT)** to determine the active frequency range of specific muscles during different gait phases (Stance vs. Swing).
3.  **Gait Phase Detection:** Correlating muscle activation bursts with Heel-Strike and Toe-Off events.

## 🛠️ 2. Prerequisites
To replicate this analysis, you will need:

### Environment
* **Python 3.8+**

### Libraries
* `numpy` (Signal manipulation)
* `matplotlib` (Visualization of EMG bursts and Scalograms)
* `scipy` (Signal processing utilities)
* `PyWavelets` (Crucial for DWT denoising and CWT analysis)
* `pandas` (Dataset handling)

### Dataset
The dataset used is the **sEMG for Basic Hand movements & Gait Analysis** from PhysioNet.
* **Source:** [PhysioNet sEMG Database](https://physionet.org/content/semg/1.0.1/)
* **Specific Usage:** Comparison of muscle activation in healthy subjects vs. subjects with gait abnormalities (if applicable based on specific file used).

## 🧠 3. Theory & Equations

### Discrete Wavelet Transform (DWT) Denoising
DWT is used to decompose the signal into approximation ($$cA$$) and detail ($$cD$$) coefficients. Noise is often present in the detail coefficients, which are thresholded before reconstruction.

$$x(t) = \sum_{k} cA_{j,k} \phi_{j,k}(t) + \sum_{j} \sum_{k} cD_{j,k} \psi_{j,k}(t)$$

*Where* $$\phi$$ *is the scaling function and* $$\psi$$ *is the wavelet function.*

### Continuous Wavelet Transform (CWT)
CWT is applied to the denoised signal to visualize how the frequency content changes over time (scalogram).

$$CWT_x(a,b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^* \left(\frac{t-b}{a}\right) dt$$

### Muscle Activation & Gait Cycle
The analysis focuses on determining the **Power Spectral Density (PSD)** to find the dominant frequency range (typically 20-150 Hz for active muscle contraction).

## 🚀 4. How to Run
1.  **Clone the repository** and enter this directory.
2.  **Prepare Data:** Download the database from PhysioNet. Ensure the `.wav` or `.mat` files for the walking trials are in the `data/` folder.
3.  **Run the script:**

    ```bash
    python emg_gait_analysis.py
    ```
5.  **Visualize:** The script outputs:
    * Raw vs. Denoised EMG plots.
    * CWT Scalograms showing frequency intensity during muscle activation.

## 📊 5. Evaluations
Based on the analysis report:
* **Denoising Efficiency:** The DWT (often using `db4` or `sym5` wavelets) effectively removed baseline wander and high-frequency noise without compromising the sharp bursts of muscle activation.
* **Frequency Range:** The CWT analysis identified that the primary muscle activation energy lies between **30 Hz and 150 Hz** during the stance phase.
* **Gait Phase Correlation:** Distinct activation patterns were observed corresponding to the **Heel-Strike** (shock absorption) and **Toe-Off** (propulsion) phases.

## 🤝 6. Contribution
Contributions are welcome under the **MIT License**.
1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/NewGaitMetric`).
3.  Commit changes.
4.  Push to the branch and open a PR.

## ⚠️ 7. Disclaimer
> **Educational & Research Purpose Only**
> This software is designed for an academic assignment at ITS (Biomedical Engineering). It is **not** intended for clinical gait analysis, medical diagnosis, or rehabilitation planning.