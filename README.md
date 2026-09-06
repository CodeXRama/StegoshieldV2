# 🛡️ StegoShield — Image Steganography Detection & Forensics

StegoShield is a machine learning and digital forensics application for detecting steganography (hidden data) in images. It combines a Convolutional Neural Network (CNN) trained with TensorFlow/Keras with forensic analysis tools including Error Level Analysis (ELA), high-frequency noise maps, color channel histograms, and EXIF metadata inspection.

---
## 🌟 Key Features

1. **CNN Steganography Detection**:
   - Neural network trained on paired clean and stego image datasets.
   - Temperature scaling calibration for smoother, calibrated probability estimates.
   - Configurable decision thresholds to balance false positive and false negative rates.

2. **Digital Image Forensics**:
   - **Error Level Analysis (ELA)**: Analyzes compression differences across image regions.
   - **Noise Residual Extraction**: Highlights subtle high-frequency spatial patterns.
   - **Color Histograms**: Analyzes discrete 256-bin distributions across Red, Green, Blue, and Luminance channels.
   - **EXIF & Metadata Viewer**: Displays camera metadata, software tags, and GPS coordinates if present.

3. **Batch Directory Scanner**:
   - Scans image folders inside the configured data root.
   - Real-time progress display and exportable CSV scan history.

4. **Web Dashboard**:
   - Interactive UI built with Streamlit for inspecting single images or performing batch scans.

---

## 📁 Repository Structure

```text
StegoShield/
├── app/
│   ├── app.py              # Main Streamlit web application
│   ├── forensics_tab.py    # Forensic tabs (ELA, Noise, Histograms, Metadata)
│   ├── scanner.py          # Detection viewport, batch scanner, report export
│   └── styles.py           # Dashboard styling and typography
├── data/
│   ├── train_data/         # Training images (clean & stego)
│   ├── val_data/           # Validation images (clean & stego)
│   └── test_data/          # Test images (clean & stego)
├── model/
│   ├── model.keras         # Trained model weights
│   ├── best_model.keras    # Best checkpoint weights
│   └── calibration.json    # Probability temperature calibration
├── tests/
│   ├── conftest.py         # Pytest fixtures and mock models
│   ├── test_forensics_utils.py # Forensic tools test suite
│   ├── test_model_utils.py # Preprocessing & thresholding tests
│   └── test_scanner.py     # Scanner & safety tests
├── utils/
│   ├── forensics_utils.py  # ELA, noise maps, histograms, EXIF extraction
│   └── model_utils.py      # Preprocessing, normalization, model loader, inference
├── calibrate_temperature.py# Probability calibration tool
├── config.py               # Application configuration and thresholds
├── evaluate_model.py       # Test set evaluation (Accuracy, F1, ROC-AUC)
├── prepare_data.py         # Dataset organization and splitting utility
├── test_system.py          # System diagnostic tool
├── train_cnn.py            # CNN training pipeline
├── Savery.ttf              # Primary UI font
├── Savery-Outline.ttf      # Outline accent font
├── StegoShield_Training_Colab.ipynb # Google Colab GPU training notebook
└── requirements.txt        # Project dependencies
```

---

## 🚀 Getting Started

### 1. Installation

Ensure Python 3.10+ is installed, then install the dependencies:

```powershell
pip install -r requirements.txt
```

### 2. Run Diagnostics

Verify that data directories and model weights are ready:

```powershell
python test_system.py
```

### 3. Launch the Application

Start the Streamlit web dashboard:

```powershell
python -m streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔬 Training & Evaluation

### Data Preparation
Organize paired clean and stego images into train, validation, and test splits:
```powershell
python prepare_data.py
```

### Training
Train the CNN model with early stopping and learning rate scheduling:
```powershell
python train_cnn.py
```
Weights are saved to `model/model.keras` and `model/best_model.keras`.

### Temperature Calibration
Calibrate output probabilities against the validation set:
```powershell
python calibrate_temperature.py
```
Saves the optimal temperature parameter to `model/calibration.json`.

### Model Evaluation
Evaluate the model on the test dataset:
```powershell
python evaluate_model.py
```

---

## 🧪 Testing

Run unit tests:

```powershell
pytest tests -q
```
All 15 unit tests cover forensics utilities, model normalization, risk thresholding, calibration, and path safety validation.
