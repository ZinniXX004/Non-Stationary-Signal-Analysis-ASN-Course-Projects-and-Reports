![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-blue)
![Language](https://img.shields.io/badge/Language-Python%20%7C%20C%2B%2B-green)

# BCI Motor Imagery Analysis and EEG Feature Extraction Using Machine Learning

## 📝 1. Description
This folder contains the hybrid Python/C++ implementation and final report for the assignment: **"BCI Motor Imagery Analysis and EEG Feature Extraction Using Machine Learning"**.

The objective of this project is to classify Motor Imagery (MI) tasks—specifically imagining left hand vs. right hand movements—using Electroencephalography (EEG) signals. To ensure high performance and real-time processing capabilities, the system uses a hybrid approach:
* **C++ (`eeg_core.cpp`):** Handles high-performance signal preprocessing and core feature extraction algorithms.
* **Python:** Manages data loading, Machine Learning model training (LDA/SVM), and result visualization.

## 🛠️ 2. Prerequisites
To replicate this analysis, you will need the following tools and libraries:

### Compilers & Environment
* **G++ / MinGW:** Required to compile the `eeg_core.cpp` module.
* **Python 3.8+**

### Python Libraries
* `numpy` (Data manipulation)
* `scipy` (Signal processing)
* `scikit-learn` (Machine Learning classifiers: LDA, SVM)
* `mne` (EEG data handling)
* `matplotlib` (Visualization)

### Dataset
The dataset used is the **BCI Competition IV (Dataset 2a or 2b)**.
* **Source:** [BCI Competition IV](https://www.bbci.de/competition/iv/)
* **Paradigm:** Motor Imagery (Left Hand, Right Hand, Feet, Tongue).

## 🧠 3. Theory & Equations

### Common Spatial Patterns (CSP)
CSP is used to maximize the variance of the spatially filtered signal for one class while minimizing it for the other. The objective function $J(w)$ is maximized:

$$J(w) = \frac{w^T R_1 w}{w^T R_2 w}$$

*Where* $$R_1$$ *and* $$R_2$$ *are the spatial covariance matrices of the two classes.*

### Feature Extraction (Log-Variance)
The features input into the classifier are the log-transformed variances of the CSP-filtered signals:

$$f_p = \log(\text{var}(Z_p))$$

### Linear Discriminant Analysis (LDA)
The classifier finds a hyperplane that separates the classes:

$$y(x) = w^T x + b$$

## 🚀 4. How to Run
Since this project uses a C++ core, you must compile it first.

1.  **Clone the repository** and navigate to this folder.
2.  **Download the Dataset:** Place the BCI Competition IV dataset files (e.g., `A01T.gdf` or `.mat`) in the `data/` directory.
3.  **Compile the C++ Core:**
    Open your terminal and run:

    ```bash
    g++ -o eeg_core eeg_core.cpp -O3
    ```
    *(Note: On Windows, this creates `eeg_core.exe`)*
5.  **Run the Main Python Script:**
    The Python script will call the compiled C++ executable or use its output.

    ```bash
    python main.py
    ```

## 📊 5. Evaluations
Based on the final project report:
* **Classification Accuracy:** The proposed CSP + LDA pipeline achieved significant accuracy (typically >70-80% for subject-dependent models) on the BCI Competition IV dataset.
* **Feature Separability:** The Log-Variance features extracted via CSP showed clear clustering between Left-Hand and Right-Hand imagery tasks.
* **Performance:** The C++ implementation of the filtering block reduced processing time compared to a pure Python implementation, simulating a real-time BCI scenario.

## 🤝 6. Contribution
Contributions are welcome under the **MIT License**.
1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/DeepLearningBCI`).
3.  Commit changes.
4.  Push to the branch and open a PR.

## ⚠️ 7. Disclaimer
> **Educational & Research Purpose Only**
> This software was developed for a Biomedical Engineering final project at ITS. It is intended for educational demonstration of Neural Engineering concepts and is **not** a medical device for clinical EEG analysis or diagnosis.