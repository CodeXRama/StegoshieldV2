"""
Data preparation script - Organize images for training
Run this before training to ensure data is properly structured
"""
import os
import shutil
from pathlib import Path
import sys
import random

from config import (
    RAW_TRAIN_CLEAN_DIR,
    RAW_TRAIN_STEGO_DIR,
    RAW_TEST_CLEAN_DIR,
    RAW_TEST_STEGO_DIR,
    TRAIN_CLEAN_DIR,
    TRAIN_STEGO_DIR,
    VAL_CLEAN_DIR,
    VAL_STEGO_DIR,
    TEST_CLEAN_DIR,
    TEST_STEGO_DIR,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)

def _list_images(folder):
    return list(folder.glob("*.[jJ][pP][gG]")) + list(folder.glob("*.[pP][nN][gG]"))

def _collect_pairs(clean_dir, stego_dir):
    clean_files = _list_images(clean_dir)
    stego_files = _list_images(stego_dir)

    clean_map = {f.stem: f for f in clean_files}
    stego_map = {f.stem: f for f in stego_files}

    common_keys = sorted(set(clean_map.keys()) & set(stego_map.keys()))
    pairs = [(clean_map[k], stego_map[k]) for k in common_keys]

    missing_clean = sorted(set(stego_map.keys()) - set(clean_map.keys()))
    missing_stego = sorted(set(clean_map.keys()) - set(stego_map.keys()))

    return pairs, missing_clean, missing_stego

def _split_pairs(pairs, val_split, test_split):
    if not pairs:
        return [], [], []

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)

    test_count = int(len(pairs) * test_split)
    val_count = int(len(pairs) * val_split)

    test_pairs = pairs[:test_count]
    val_pairs = pairs[test_count:test_count + val_count]
    train_pairs = pairs[test_count + val_count:]

    return train_pairs, val_pairs, test_pairs

def _move_pairs(pairs, clean_dir, stego_dir):
    moved = 0
    for clean_path, stego_path in pairs:
        clean_target = clean_dir / clean_path.name
        stego_target = stego_dir / stego_path.name

        if not clean_target.exists():
            shutil.move(str(clean_path), str(clean_target))
        if not stego_target.exists():
            shutil.move(str(stego_path), str(stego_target))
        moved += 1
    return moved

def check_and_prepare_data():
    """Check data structure and prepare for training"""
    
    print("=" * 60)
    print("StegoShield Data Preparation")
    print("=" * 60)
    
    data_dir = Path("data")
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"\n❌ 'data' directory not found!")
        print(f"Creating directory structure...")
        data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (data_dir / "clean_images").mkdir(exist_ok=True)
        (data_dir / "stego_images").mkdir(exist_ok=True)
        
        print(f"✓ Created 'data/clean_images/' - Place clean images here")
        print(f"✓ Created 'data/stego_images/' - Place images with hidden data here")
        return False
    
    # Raw input folders
    raw_train_clean_dir = Path(RAW_TRAIN_CLEAN_DIR)
    raw_train_stego_dir = Path(RAW_TRAIN_STEGO_DIR)
    raw_test_clean_dir = Path(RAW_TEST_CLEAN_DIR)
    raw_test_stego_dir = Path(RAW_TEST_STEGO_DIR)

    raw_train_clean_dir.mkdir(parents=True, exist_ok=True)
    raw_train_stego_dir.mkdir(parents=True, exist_ok=True)
    raw_test_clean_dir.mkdir(parents=True, exist_ok=True)
    raw_test_stego_dir.mkdir(parents=True, exist_ok=True)

    # Structured folders (same as raw for train/test, plus val)
    train_clean_dir = Path(TRAIN_CLEAN_DIR)
    train_stego_dir = Path(TRAIN_STEGO_DIR)
    val_clean_dir = Path(VAL_CLEAN_DIR)
    val_stego_dir = Path(VAL_STEGO_DIR)
    test_clean_dir = Path(TEST_CLEAN_DIR)
    test_stego_dir = Path(TEST_STEGO_DIR)

    train_clean_dir.mkdir(parents=True, exist_ok=True)
    train_stego_dir.mkdir(parents=True, exist_ok=True)
    val_clean_dir.mkdir(parents=True, exist_ok=True)
    val_stego_dir.mkdir(parents=True, exist_ok=True)
    test_clean_dir.mkdir(parents=True, exist_ok=True)
    test_stego_dir.mkdir(parents=True, exist_ok=True)

    # Pair-safe split from train_data if test_data is empty
    raw_train_pairs, missing_clean, missing_stego = _collect_pairs(
        raw_train_clean_dir, raw_train_stego_dir
    )

    if missing_clean or missing_stego:
        if missing_clean:
            print(f"\n⚠️  Missing clean pairs for {len(missing_clean)} stego files")
        if missing_stego:
            print(f"⚠️  Missing stego pairs for {len(missing_stego)} clean files")

    raw_test_pairs, test_missing_clean, test_missing_stego = _collect_pairs(
        raw_test_clean_dir, raw_test_stego_dir
    )

    if test_missing_clean or test_missing_stego:
        if test_missing_clean:
            print(f"\n⚠️  Missing clean pairs in test_data for {len(test_missing_clean)} stego files")
        if test_missing_stego:
            print(f"⚠️  Missing stego pairs in test_data for {len(test_missing_stego)} clean files")

    if not raw_test_pairs and raw_train_pairs:
        train_pairs, val_pairs, test_pairs = _split_pairs(
            raw_train_pairs, VAL_SPLIT, TEST_SPLIT
        )
        moved_train = _move_pairs(train_pairs, train_clean_dir, train_stego_dir)
        moved_val = _move_pairs(val_pairs, val_clean_dir, val_stego_dir)
        moved_test = _move_pairs(test_pairs, test_clean_dir, test_stego_dir)

        print(f"\n📦 Moved paired images -> train: {moved_train}, val: {moved_val}, test: {moved_test}")
    else:
        # If test_data already exists, only create val split from remaining train_data
        train_pairs, val_pairs, _ = _split_pairs(raw_train_pairs, VAL_SPLIT, 0.0)
        moved_train = _move_pairs(train_pairs, train_clean_dir, train_stego_dir)
        moved_val = _move_pairs(val_pairs, val_clean_dir, val_stego_dir)

        print(f"\n📦 Moved paired images -> train: {moved_train}, val: {moved_val}")

    # Count images in structured folders
    train_clean = _list_images(train_clean_dir)
    train_stego = _list_images(train_stego_dir)
    val_clean = _list_images(val_clean_dir)
    val_stego = _list_images(val_stego_dir)
    test_clean = _list_images(test_clean_dir)
    test_stego = _list_images(test_stego_dir)
    
    print(f"\n📊 Data Status:")
    print(f"   Train clean: {len(train_clean)}")
    print(f"   Train stego: {len(train_stego)}")
    print(f"   Val clean:   {len(val_clean)}")
    print(f"   Val stego:   {len(val_stego)}")
    print(f"   Test clean:  {len(test_clean)}")
    print(f"   Test stego:  {len(test_stego)}")
    print(f"   Total: {len(train_clean) + len(train_stego) + len(val_clean) + len(val_stego) + len(test_clean) + len(test_stego)}")
    
    min_required = 50
    if len(train_clean) < min_required:
        print(f"\n⚠️  Low train clean count ({len(train_clean)}/{min_required} minimum)")
    if len(train_stego) < min_required:
        print(f"⚠️  Low train stego count ({len(train_stego)}/{min_required} minimum)")
    if len(val_clean) == 0 or len(val_stego) == 0:
        print(f"⚠️  Validation set is empty or incomplete")
    if len(test_clean) == 0 or len(test_stego) == 0:
        print(f"⚠️  Test set is empty or incomplete")
    
    # Check class balance
    if len(train_clean) > 0 and len(train_stego) > 0:
        ratio = len(train_clean) / len(train_stego)
        if ratio < 0.8 or ratio > 1.2:
            print(f"\n⚠️  Imbalanced dataset! Ratio: {ratio:.2f} (aim for 1.0)")
    
    # Ask user if data is ready
    print("\n" + "=" * 60)
    if len(train_clean) >= min_required and len(train_stego) >= min_required:
        print("✓ Data structure looks good!")
        return True
    else:
        print("\n📝 Next steps:")
        print(f"   1. Add paired images to 'data/train_data/clean_images/' and 'data/train_data/stego_images/'")
        print(f"   2. Ensure filenames match across clean/stego pairs")
        print(f"   3. Aim for at least {min_required} images in each folder")
        print(f"   4. Run this script again to verify")
        return False

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    ready = check_and_prepare_data()
    
    if ready:
        print("\n🚀 Ready to run: python train_cnn.py")
    else:
        print("\n📁 Please organize your data and try again.")
