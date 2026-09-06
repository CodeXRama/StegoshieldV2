# 🛡️ StegoShield — Quick Start Guide

StegoShield is an image steganography detection and forensic analysis tool built with Python, TensorFlow/Keras, and Streamlit.

---

## ⚡ Quick Start

### 1. Environment Setup

Install project dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Verify System Health

Run the diagnostic script to check data paths and model availability:

```powershell
python test_system.py
```

### 3. Launch Dashboard

Start the Streamlit web application:

```powershell
python -m streamlit run app/app.py
```

The app will open at **http://localhost:8501**.

---

## 🎯 Features

1. **Stego Detection**:
   - Convolutional neural network classification.
   - Probabilistic confidence score and risk classification (**LOW / MEDIUM / HIGH**).
   - Temperature calibration for reliable probability scaling.

2. **Forensic Analysis Tools**:
   - **Error Level Analysis (ELA)**: Visualizes JPEG compression differentials.
   - **Noise Analysis**: Extracts high-frequency spatial noise patterns.
   - **Color Histograms**: Plots Red, Green, Blue, and Luminance distributions.
   - **EXIF & Metadata**: Reads image tags, camera information, and GPS coordinates if available.

3. **Batch Directory Scanner**:
   - Scans entire folders of images within allowed directories.
   - Export scan results to CSV.

---

## 🛠️ Common Commands

| Task | Command | Description |
|------|---------|-------------|
| **Launch Dashboard** | `python -m streamlit run app/app.py` | Starts the Streamlit dashboard |
| **Run Health Check** | `python test_system.py` | Validates data directories and model loading |
| **Run Unit Tests** | `pytest tests -q` | Runs automated test suite |
| **Organize Dataset** | `python prepare_data.py` | Splits clean and stego pairs into train/val/test |
| **Train CNN Model** | `python train_cnn.py` | Trains the CNN model and saves to `model/` |
| **Evaluate Test Split**| `python evaluate_model.py` | Calculates test accuracy, precision, recall, F1, and AUC |
| **Calibrate Probabilities** | `python calibrate_temperature.py` | Optimizes temperature scaling value |

---

## 📁 Project Layout

```text
StegoShield/
├── app/
│   ├── app.py              # Main web application (Streamlit)
│   ├── forensics_tab.py    # ELA, Noise map, Histogram, and EXIF tabs
│   ├── scanner.py          # Detection viewport and batch scanner
│   └── styles.py           # Dashboard theme and styling
├── data/
│   ├── train_data/         # Training clean & stego images
│   ├── val_data/           # Validation clean & stego images
│   └── test_data/          # Held-out test images
├── model/
│   ├── model.keras         # Trained model weights
│   ├── best_model.keras    # Best checkpoint weights
│   └── calibration.json    # Temperature scaling configuration
├── tests/
│   ├── test_forensics_utils.py
│   ├── test_model_utils.py
│   └── test_scanner.py
├── utils/
│   ├── forensics_utils.py  # ELA, noise analysis, histogram, EXIF logic
│   └── model_utils.py      # Preprocessing, normalization, model inference
├── calibrate_temperature.py# Probability calibration tool
├── config.py               # Central configuration file
├── evaluate_model.py       # Test split evaluation
├── prepare_data.py         # Data organization utility
├── test_system.py          # System diagnostic tool
├── train_cnn.py            # CNN training pipeline
├── Savery.ttf              # Primary UI font
├── Savery-Outline.ttf      # Accent UI font
├── StegoShield_Training_Colab.ipynb # GPU training notebook
└── requirements.txt        # Dependencies
```
