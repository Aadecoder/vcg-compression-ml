# VCG Signal Compression Using CNN–LSTM Autoencoder

This repository implements a **Vectorcardiogram (VCG) signal compression system using deep learning**, specifically a **CNN–LSTM autoencoder**, designed with eventual **Raspberry Pi deployment** in mind.

The project focuses on compressing physiological VCG signals while preserving clinically relevant morphology, and evaluates performance using standard biomedical compression metrics.

---

## Project Overview

Electrocardiogram (ECG) and Vectorcardiogram (VCG) signals contain significant **spatial and temporal redundancy**. Efficient compression is essential for:

- Long-term monitoring
- Embedded / IoT healthcare devices
- Low-bandwidth transmission
- Edge processing (e.g., Raspberry Pi)

This project:
1. Converts ECG signals to VCG (X, Y, Z components)
2. Preprocesses signals using biomedical filtering and normalization
3. Segments signals into overlapping windows
4. Compresses them using a CNN–LSTM autoencoder
5. Reconstructs signals and evaluates compression quality

---

## Methodology

### 1. ECG → VCG Conversion
- Uses **Frank’s transformation** to convert multi-lead ECG into orthogonal VCG components (X, Y, Z)
- Input source: **PTB Diagnostic ECG Database (PhysioNet)**

---

### 2. Preprocessing
Implemented in `VCGPreprocessor`:

- **Low-pass Butterworth filter**
  - Cutoff: 40 Hz
  - Order: 5
  - Zero-phase filtering using `filtfilt`
- **Z-score normalization**
  - Per-channel normalization (X, Y, Z)
  - Stabilizes training and improves reconstruction quality

---

### 3. Sliding Window Segmentation
Implemented in `VCGDataGenerator`:

- Window size: **1250 samples** (≈1.25 s at 1000 Hz)
- Overlap: **50%**
- Ensures each window contains a complete cardiac cycle
- Increases dataset size and reduces boundary artifacts

---

### 4. CNN–LSTM Autoencoder Architecture

#### Encoder
- **Conv1D layers** (32 → 64 → 128 filters)
- **MaxPooling** for temporal downsampling
- **LSTM layers** (128 → 128 → 64 units)
- **Dense bottleneck** (compressed representation)

#### Decoder
- Mirrors the encoder
- Dense expansion → LSTM layers → upsampling → Conv1DTranspose
- Reconstructs original VCG window

**Compression Ratio (CR):** 30:1

---

## Performance Metrics

Implemented in `PerformanceEvaluator`, following standard ECG/VCG compression literature:

- Compression Ratio (CR)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Normalized MSE (NMSE)
- Percentage RMS Difference (PRD)
- PRD Normalized (PRDN)
- Signal-to-Noise Ratio (SNR)
- Peak SNR (PSNR)
- Quality Score (QS)

### Example Results (Average over test samples)

- CR = 30.00
- MSE = 0.0757
- RMSE = 0.2606
- NMSE = 0.0997
- PRD = 28.08 %
- SNR = 11.23 dB
- PSNR = 25.05 dB
- QS = 1.12

These values indicate **acceptable biomedical compression quality** at a high compression ratio.

---

## Visualizations

The code generates:
- Training & validation loss curves
- Original vs reconstructed VCG waveforms
- 3D VCG trajectory plots (X–Y–Z)

---

## Raspberry Pi Deployment

- Model can be exported to **TensorFlow Lite**
- Optimized for CPU inference
- Suitable for edge deployment on Raspberry Pi–class devices

---

## Project Structure
```text
├── model
│   ├── best_vcg_autoencoder.keras      # Best trained autoencoder model
│   ├── output.txt                     # Model output and performance metrics
│   ├── training_history.png           # Training and validation loss curves
│   ├── vcg_3d_original.png            # 3D plot of original VCG signal
│   ├── vcg_3d_reconstructed.png       # 3D plot of reconstructed VCG signal
│   ├── vcg_comparison.png             # Original vs reconstructed VCG waveforms
│   ├── vcg_compression_ml.py          # Main training and evaluation pipeline
│   ├── vcg_decoder.keras              # Trained decoder model
│   └── vcg_encoder.keras              # Trained encoder model
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── vcg_compression_model(1).ipynb     # Jupyter notebook version of the pipeline
├── low-pass-butterworth-filter.md     # Explanation of Butterworth filtering
├── sliding-window.md                  # Explanation of sliding window segmentation
└── z-score-normalization.md           # Explanation of z-score normalization
```

---

## Dataset

- **PTB Diagnostic ECG Database**
- Source: PhysioNet
- Sampling frequency: 1000 Hz
- Converted to VCG using Frank’s lead system
- https://www.physionet.org/content/ptbdb/1.0.0/#files-panel

---

## Requirements (Core)

- Python 3.9+
- NumPy
- SciPy
- wfdb
- TensorFlow / Keras
- scikit-learn
- matplotlib
- PyWavelets

---

## References

This project is based on and inspired by the methodology described in the following research paper:

https://www.sciencedirect.com/science/article/pii/S1746809425013072
 
---

## Author

Developed as part of an **ECE Year Long Project** focused on  
**VCG Signal Compression using Machine Learning**.

---