"""
Test script to verify model and app functionality
"""
import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.model_utils import load_model, predict_image, load_and_preprocess_image
from config import (
    TRAIN_CLEAN_DIR,
    TRAIN_STEGO_DIR,
    VAL_CLEAN_DIR,
    VAL_STEGO_DIR,
    TEST_CLEAN_DIR,
    TEST_STEGO_DIR,
)

def test_model():
    """Test if model loads and makes predictions"""
    print("=" * 60)
    print("StegoShield Model Test")
    print("=" * 60)
    
    # Test 1: Load model
    print("\n[1/3] Testing model loading...")
    try:
        model = load_model("model/cnn_model.keras")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Test 2: Create dummy image
    print("\n[2/3] Creating test image...")
    try:
        # Create a random test image
        test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        test_img = test_img.astype(np.float32) / 255.0
        print("✓ Test image created")
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return False
    
    # Test 3: Make prediction
    print("\n[3/3] Testing prediction...")
    try:
        result = predict_image(model, test_img)
        print(f"✓ Prediction successful!")
        print(f"   Classification: {result['classification']}")
        print(f"   Confidence: {result['probability']:.2f}%")
        print(f"   Risk Level: {result['risk_level']}")
    except Exception as e:
        print(f"❌ Failed to make prediction: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    return True

def test_data():
    """Check if training data exists"""
    print("=" * 60)
    print("StegoShield Data Test")
    print("=" * 60)
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("\n❌ 'data' directory not found")
        print("Run: python prepare_data.py")
        return False
    
    train_clean_dir = Path(TRAIN_CLEAN_DIR)
    train_stego_dir = Path(TRAIN_STEGO_DIR)
    val_clean_dir = Path(VAL_CLEAN_DIR)
    val_stego_dir = Path(VAL_STEGO_DIR)
    test_clean_dir = Path(TEST_CLEAN_DIR)
    test_stego_dir = Path(TEST_STEGO_DIR)

    train_clean = list(train_clean_dir.glob("*.[jJ][pP][gG]")) + list(train_clean_dir.glob("*.[pP][nN][gG]"))
    train_stego = list(train_stego_dir.glob("*.[jJ][pP][gG]")) + list(train_stego_dir.glob("*.[pP][nN][gG]"))
    val_clean = list(val_clean_dir.glob("*.[jJ][pP][gG]")) + list(val_clean_dir.glob("*.[pP][nN][gG]"))
    val_stego = list(val_stego_dir.glob("*.[jJ][pP][gG]")) + list(val_stego_dir.glob("*.[pP][nN][gG]"))
    test_clean = list(test_clean_dir.glob("*.[jJ][pP][gG]")) + list(test_clean_dir.glob("*.[pP][nN][gG]"))
    test_stego = list(test_stego_dir.glob("*.[jJ][pP][gG]")) + list(test_stego_dir.glob("*.[pP][nN][gG]"))

    print(f"\n✓ Data directory structure found")
    print(f"   Train clean: {len(train_clean)}")
    print(f"   Train stego: {len(train_stego)}")
    print(f"   Val clean:   {len(val_clean)}")
    print(f"   Val stego:   {len(val_stego)}")
    print(f"   Test clean:  {len(test_clean)}")
    print(f"   Test stego:  {len(test_stego)}")

    if len(train_clean) == 0 or len(train_stego) == 0:
        print("\n⚠️  Missing training data!")
        print("Run: python prepare_data.py")
        return False

    if len(val_clean) == 0 or len(val_stego) == 0:
        print("\n⚠️  Missing validation data!")
        print("Run: python prepare_data.py")
        return False
    
    return True

if __name__ == "__main__":
    print("\n🔍 Running StegoShield diagnostics...\n")
    
    data_ok = test_data()
    model_ok = test_model()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Data Status: {'✓ OK' if data_ok else '❌ NEEDS ATTENTION'}")
    print(f"Model Status: {'✓ OK' if model_ok else '❌ NEEDS TRAINING'}")
    
    if data_ok and model_ok:
        print("\n🚀 System ready! Run: streamlit run app/app.py")
    else:
        print("\n📝 Follow the instructions above to fix issues")
