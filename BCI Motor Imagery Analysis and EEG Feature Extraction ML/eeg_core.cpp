/*
 * eeg_core.cpp
 * 
 * Core signal processing module for EEG Analysis.
 * Optimized for BCI Competition IV Dataset 2b.
 * 
 * Features:
 * 1. IIR Filter (Direct Form II Transposed) for Bandpass Filtering.
 * 2. Continuous Wavelet Transform (Morlet & Mexican Hat) via Time-Domain Convolution.
 * 3. Moving Average for smoothing power signals.
 * 
 * Compile with g++ (MSYS2/MinGW):
 * g++ -O3 -shared -static -o eeg_processing.dll eeg_core.cpp
 */

#include <cmath>
#include <vector>
#include <complex>
#include <algorithm>

// Macro to export functions for Windows DLL compatibility
#ifdef _WIN32
    #define DLLEXPORT extern "C" __declspec(dllexport)
#else
    #define DLLEXPORT extern "C"
#endif

// Constants
const double PI = 3.14159265358979323846;

// 1. IIR FILTER IMPLEMENTATION (Standard Difference Equation)
DLLEXPORT void apply_filter(double* input, int length, 
                            double* b, int b_len, 
                            double* a, int a_len, 
                            double* output) {
    
    // Initialize output with zeros to avoid garbage values
    for (int i = 0; i < length; i++) {
        output[i] = 0.0;
    }

    // Direct Form I Implementation (Stable and simple)
    // Formula: y[n] = (b0*x[n] + ... + bM*x[n-M]) - (a1*y[n-1] + ... + aN*y[n-N])
    
    for (int n = 0; n < length; n++) {
        // Feedforward part (b coefficients)
        for (int k = 0; k < b_len; k++) {
            if (n - k >= 0) {
                output[n] += b[k] * input[n - k];
            }
        }

        // Feedback part (a coefficients)
        for (int k = 1; k < a_len; k++) {
            if (n - k >= 0) {
                output[n] -= a[k] * output[n - k];
            }
        }
        
        // Normalize by a[0] if it's not 1.0 (Safety check)
        if (a[0] != 1.0 && a[0] != 0.0) {
            output[n] /= a[0];
        }
    }
}

// 2. MOVING AVERAGE (Smoothing)
DLLEXPORT void moving_average(double* input, int length, int window_size, double* output) {
    if (window_size <= 0) return;

    double sum = 0.0;
    
    // Initial window
    for (int i = 0; i < length; i++) {
        sum += input[i];
        
        if (i >= window_size) {
            sum -= input[i - window_size];
        }

        if (i >= window_size - 1) {
            output[i] = sum / window_size;
        } else {
            // Padding strategy: Average of available samples (ramp-up)
            output[i] = sum / (i + 1);
        }
    }
}

// 3. CONTINUOUS WAVELET TRANSFORM (Morlet)
DLLEXPORT void compute_cwt_magnitude(double* input, int length, 
                                     double* scales, int num_scales, 
                                     double fs, 
                                     double* output) {
    
    double omega0 = 6.0; // Standard central frequency for Morlet
    
    for (int s = 0; s < num_scales; s++) {
        double scale = scales[s];
        
        // Define half-width of the wavelet in samples (Support)
        int half_width = (int)(scale * 4.0); 
        int kernel_len = 2 * half_width + 1;
        
        std::vector<std::complex<double>> kernel(kernel_len);
        
        double norm_factor = pow(PI, -0.25) / sqrt(scale);

        for (int k = 0; k < kernel_len; k++) {
            double t = (k - half_width) / fs; // Time in seconds
            double t_scaled = t / (scale / fs); // Normalized time
            
            // Complex Morlet: exp(i * w0 * t_scaled) * exp(-0.5 * t_scaled^2)
            double gaussian = exp(-0.5 * t_scaled * t_scaled);
            double real_part = cos(omega0 * t_scaled) * gaussian;
            double imag_part = sin(omega0 * t_scaled) * gaussian;
            
            kernel[k] = std::complex<double>(real_part, imag_part) * norm_factor;
        }

        // Convolution loop
        for (int t = 0; t < length; t++) {
            std::complex<double> conv_sum(0.0, 0.0);
            
            for (int k = 0; k < kernel_len; k++) {
                int input_idx = t - half_width + k;
                
                if (input_idx >= 0 && input_idx < length) {
                    conv_sum += input[input_idx] * std::conj(kernel[k]);
                }
            }
            output[s * length + t] = std::abs(conv_sum);
        }
    }
}

// 4. CONTINUOUS WAVELET TRANSFORM (Mexican Hat / Ricker)
/*
 * Computes CWT using the Real-valued Mexican Hat wavelet.
 * Formula: psi(t) = (2 / (sqrt(3*sigma) * pi^0.25)) * (1 - t^2/sigma^2) * exp(-t^2 / (2*sigma^2))
 * This is widely used for peak detection and is a good alternative to Morlet.
 */
DLLEXPORT void compute_cwt_mexican_hat(double* input, int length, 
                                       double* scales, int num_scales, 
                                       double fs, 
                                       double* output) {
    
    for (int s = 0; s < num_scales; s++) {
        double scale = scales[s];
        
        // Support for Mexican Hat is typically [-5, 5] standard deviations
        int half_width = (int)(scale * 5.0);
        int kernel_len = 2 * half_width + 1;
        
        std::vector<double> kernel(kernel_len);
        
        // Normalization factor A = 2 / (sqrt(3*scale) * pi^0.25)
        double norm_factor = 2.0 / (sqrt(3.0 * scale) * pow(PI, 0.25));

        for (int k = 0; k < kernel_len; k++) {
            double t = (k - half_width) / fs;
            double t_sq = (t * t) / ((scale/fs) * (scale/fs)); // t^2 / sigma^2
            
            // Mexican Hat Function
            kernel[k] = norm_factor * (1.0 - t_sq) * exp(-0.5 * t_sq);
        }

        // Convolution Loop (Real-valued)
        for (int t = 0; t < length; t++) {
            double conv_sum = 0.0;
            
            for (int k = 0; k < kernel_len; k++) {
                int input_idx = t - half_width + k;
                if (input_idx >= 0 && input_idx < length) {
                    conv_sum += input[input_idx] * kernel[k];
                }
            }
            // Store Magnitude (Absolute value since Mexican Hat can be negative)
            output[s * length + t] = std::abs(conv_sum);
        }
    }
}