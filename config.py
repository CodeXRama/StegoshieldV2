# StegoShield Configuration
# Modify these settings to adjust model training and prediction behavior

# ===== TRAINING CONFIG =====
IMG_SIZE = 512             # Image size for training (512x512 pixels)
BATCH_SIZE = 32            # Training batch size
EPOCHS = 10                # Maximum number of training epochs
LEARNING_RATE = 0.0003     # Initial learning rate
RANDOM_SEED = 42           # Random seed for reproducibility

# ===== MODEL CONFIG =====
MODEL_PATH = "model/cnn_model.keras"
BEST_MODEL_PATH = "model/best_model.keras"

# Early stopping patience (epochs without improvement before stopping)
EARLY_STOPPING_PATIENCE = 10

# Learning rate reduction patience
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ===== PREDICTION CONFIG =====
DETECTION_THRESHOLD = 0.5  # Confidence threshold for steganography detection
RISK_THRESHOLDS = {
    "CLEAN": 0.5,          # <50% confidence: Safe
    "SUSPICIOUS": 0.7,     # 50-70% confidence: Suspicious
    "DETECTED": 1.0        # >70% confidence: Threat detected
}

# ===== DATA CONFIG =====
DATA_DIR = "data"

# Raw input folders (paired images live here initially)
RAW_TRAIN_CLEAN_DIR = "data/train_data/clean_images"
RAW_TRAIN_STEGO_DIR = "data/train_data/stego_images"
RAW_TEST_CLEAN_DIR = "data/test_data/clean_images"
RAW_TEST_STEGO_DIR = "data/test_data/stego_images"

# Structured folders used by training/validation/testing
TRAIN_CLEAN_DIR = "data/train_data/clean_images"
TRAIN_STEGO_DIR = "data/train_data/stego_images"
VAL_CLEAN_DIR = "data/val_data/clean_images"
VAL_STEGO_DIR = "data/val_data/stego_images"
TEST_CLEAN_DIR = "data/test_data/clean_images"
TEST_STEGO_DIR = "data/test_data/stego_images"

# Split ratios for automatic pair-safe reorganization
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Image augmentation settings
AUGMENTATION = {
    "rotation_range": 15,
    "brightness_range": 0.2,
    "contrast_range": [0.8, 1.2],
    "horizontal_flip": True,
    "vertical_flip": True
}

# ===== APP CONFIG =====
APP_TITLE = "StegoRadar | Steganography Detection"
APP_MAX_FILE_SIZE_MB = 10
STREAMLIT_THEME = "dark"
