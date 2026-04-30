# StegoShield - Steganography Detection System

A deep learning-based system for detecting hidden data (steganography) in images using CNN and High-Pass Filtering.

## Project Structure

```
StegoShield/
├── train_cnn.py           # Training script
├── app/
│   └── app.py            # Streamlit web interface
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/
│   ├── clean_images/     # Raw clean images (drop here)
│   ├── stego_images/     # Raw stego images (drop here)
│   ├── train/
│   │   ├── clean/        # Training clean images
│   │   └── stego/        # Training stego images
│   └── test/
│       ├── clean/        # Test clean images
│       └── stego/        # Test stego images
├── model/
│   ├── cnn_model.keras   # Trained model
│   └── best_model.keras  # Best checkpoint
└── utils/
    └── model_utils.py    # Utility functions
```

## Installation

1. **Clone/Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   # or
   source venv/bin/activate      # Linux/Mac
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Setup Data

1. Place clean images (no hidden data) in `data/clean_images/`
2. Place images with steganography in `data/stego_images/`
3. Run `python prepare_data.py` to split into:
  - `data/train/clean/` and `data/train/stego/`
  - `data/test/clean/` and `data/test/stego/`

Example directory structure:
```
data/
├── clean_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── stego_images/
    ├── hidden1.jpg
    ├── hidden2.jpg
    └── ...
```

## Training the Model

Run the training script:
```bash
python train_cnn.py
```

**What the training script does:**
- Loads images from `data/train/clean/` and `data/train/stego/`
- Applies High-Pass Filter (HPF) preprocessing for noise extraction
- Splits training data 80/20 into training and validation sets
- Evaluates on the held-out test set in `data/test/clean/` and `data/test/stego/`
- Trains a CNN with:
  - 50 epochs maximum
  - Early stopping (patience=10)
  - Learning rate reduction on plateau
  - Model checkpointing (saves best model)
- Evaluates and saves the trained model

**Output:**
- `model/cnn_model.keras` - Final trained model
- `model/best_model.keras` - Best checkpoint
- `model/training_history.png` - Training curves

## Running the Web App

Launch the Streamlit app:
```bash
streamlit run app/app.py
```

Then open: http://localhost:8501

**Features:**
- Upload images (JPG, PNG, BMP)
- Real-time steganography detection
- Confidence scores and risk assessment
- Adjustable detection threshold
- Detailed analysis metrics

**Demo mode note:**
- The UI can optionally force a DETECTED/HIGH result for showcasing via `DEMO_FORCE_DETECTED` in [app/app.py](app/app.py).

## Model Architecture

The CNN model consists of:

```
Input (64x64x3)
  ↓
Block 1: Conv2D(32) → BN → Conv2D(32) → BN → AvgPool → Dropout
  ↓
Block 2: Conv2D(64) → BN → Conv2D(64) → BN → AvgPool → Dropout
  ↓
Block 3: Conv2D(128) → BN → Conv2D(128) → BN → AvgPool → Dropout
  ↓
Block 4: Conv2D(256) → BN → Dropout
  ↓
GlobalAveragePooling2D()
  ↓
Dense(256) → BN → Dropout(0.5)
  ↓
Dense(128) → BN → Dropout(0.4)
  ↓
Dense(1, sigmoid) → Output [0-1]
```

## Key Improvements

### Fixed Issues:
✓ Fixed data pipeline - now uses proper tf.data API instead of deprecated ImageDataGenerator
✓ Increased epochs from 4 to 50 with early stopping
✓ Added High-Pass Filter preprocessing to extract steganographic noise
✓ Integrated real model predictions in web app (no more random values)
✓ Added comprehensive error handling
✓ Created reusable utility functions
✓ Added configuration management

### Performance Enhancements:
✓ Improved CNN architecture with doubled Conv blocks
✓ Better regularization (L2 + Dropout + BatchNorm)
✓ More metrics tracked (AUC, Precision, Recall, F1)
✓ Model checkpointing to save best weights
✓ Learning rate scheduling with ReduceLROnPlateau
✓ Data augmentation for better generalization

## Configuration

Edit `config.py` to adjust:
- Image size, batch size, training epochs
- Detection thresholds
- Data directory paths
- Model save locations
- Augmentation parameters

## Expected Performance

After training on a balanced dataset:
- Accuracy: >85%
- Precision: >80%
- Recall: >80%
- AUC: >0.90

## Troubleshooting

**No images found during training:**
- Ensure images are in `data/clean_images/` and `data/stego_images/`
- Verify image formats are `.jpg`, `.jpeg`, or `.png`

**Low accuracy:**
- Increase training data (need at least 100-200 images per class)
- Adjust augmentation settings in config.py
- Increase EPOCHS in config.py
- Check image quality and diversity

**Model loading errors in app:**
- Ensure `model/cnn_model.keras` exists
- Run training script first: `python train_cnn.py`
- Check file permissions

## Dependencies

- TensorFlow 2.21.0
- Keras 3.12.1
- OpenCV 4.13.0
- Scikit-learn 1.7.2
- NumPy 2.2.6
- Streamlit 1.55.0
- Matplotlib 3.10.0

## Author
StegoShield Development Team

## Disclaimer
This tool is for **authorized security analysis and educational purposes only**. Misuse for unauthorized access or unethical purposes is prohibited.
