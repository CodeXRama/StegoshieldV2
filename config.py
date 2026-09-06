# StegoShield Configuration
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0005
RANDOM_SEED = 42
MODEL_PATH = "model/model.keras"
BEST_MODEL_PATH = "model/best_model.keras"
CALIBRATION_PATH = "model/calibration.json"
EARLY_STOPPING_PATIENCE = 5

REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5

USE_RESIDUAL = False

DETECTION_THRESHOLD = 0.5
RISK_THRESHOLDS = {
    "CLEAN": 0.5,
    "SUSPICIOUS": 0.7,
    "DETECTED": 1.0,
}

DATA_DIR = "data"

TRAIN_CLEAN_DIR = "data/train_data/clean_images"
TRAIN_STEGO_DIR = "data/train_data/stego_images"
VAL_CLEAN_DIR = "data/val_data/clean_images"
VAL_STEGO_DIR = "data/val_data/stego_images"
TEST_CLEAN_DIR = "data/test_data/clean_images"
TEST_STEGO_DIR = "data/test_data/stego_images"

VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

AUGMENTATION = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rot90": True,
}

APP_TITLE = "StegoShield | Image Steganography Detection"
APP_MAX_FILE_SIZE_MB = 10
STREAMLIT_THEME = "dark"

ALLOWED_SCAN_ROOT = "data"
