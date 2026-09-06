import os
import shutil
from pathlib import Path
import random

from config import (
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
    p = Path(folder)
    return list(p.glob("*.[jJ][pP][gG]")) + list(p.glob("*.[pP][nN][gG]")) + list(p.glob("*.[jJ][pP][eE][gG]"))

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
    shuffled = list(pairs)
    rng.shuffle(shuffled)

    test_count = int(len(shuffled) * test_split)
    val_count = int(len(shuffled) * val_split)

    test_pairs = shuffled[:test_count]
    val_pairs = shuffled[test_count:test_count + val_count]
    train_pairs = shuffled[test_count + val_count:]

    return train_pairs, val_pairs, test_pairs

def check_and_prepare_data():
    print("=" * 60)
    print("StegoShield Data Preparation")
    print("=" * 60)

    train_clean_dir = Path(TRAIN_CLEAN_DIR)
    train_stego_dir = Path(TRAIN_STEGO_DIR)
    val_clean_dir = Path(VAL_CLEAN_DIR)
    val_stego_dir = Path(VAL_STEGO_DIR)
    test_clean_dir = Path(TEST_CLEAN_DIR)
    test_stego_dir = Path(TEST_STEGO_DIR)

    for d in [train_clean_dir, train_stego_dir, val_clean_dir, val_stego_dir, test_clean_dir, test_stego_dir]:
        d.mkdir(parents=True, exist_ok=True)

    raw_train_pairs, missing_clean, missing_stego = _collect_pairs(train_clean_dir, train_stego_dir)
    if missing_clean or missing_stego:
        if missing_clean:
            print(f"\n⚠️  Missing clean pairs for {len(missing_clean)} stego files")
        if missing_stego:
            print(f"⚠️  Missing stego pairs for {len(missing_stego)} clean files")

    val_pairs, _, _ = _collect_pairs(val_clean_dir, val_stego_dir)
    test_pairs, _, _ = _collect_pairs(test_clean_dir, test_stego_dir)

    if raw_train_pairs and (not val_pairs or not test_pairs):
        train_p, val_p, test_p = _split_pairs(raw_train_pairs, VAL_SPLIT, TEST_SPLIT)

        # Copy to val and test
        copied_val = 0
        for clean_p, stego_p in val_p:
            c_target = val_clean_dir / clean_p.name
            s_target = val_stego_dir / stego_p.name
            if not c_target.exists():
                shutil.copy2(str(clean_p), str(c_target))
            if not s_target.exists():
                shutil.copy2(str(stego_p), str(s_target))
            copied_val += 1

        copied_test = 0
        for clean_p, stego_p in test_p:
            c_target = test_clean_dir / clean_p.name
            s_target = test_stego_dir / stego_p.name
            if not c_target.exists():
                shutil.copy2(str(clean_p), str(c_target))
            if not s_target.exists():
                shutil.copy2(str(stego_p), str(s_target))
            copied_test += 1

        print(f"\n[INFO] Safe copy complete -> Val pairs: {copied_val}, Test pairs: {copied_test}")

    train_clean = _list_images(train_clean_dir)
    train_stego = _list_images(train_stego_dir)
    val_clean = _list_images(val_clean_dir)
    val_stego = _list_images(val_stego_dir)
    test_clean = _list_images(test_clean_dir)
    test_stego = _list_images(test_stego_dir)

    print(f"\n[DATA] Current Status:")
    print(f"   Train clean: {len(train_clean)}")
    print(f"   Train stego: {len(train_stego)}")
    print(f"   Val clean:   {len(val_clean)}")
    print(f"   Val stego:   {len(val_stego)}")
    print(f"   Test clean:  {len(test_clean)}")
    print(f"   Test stego:  {len(test_stego)}")
    total = sum([len(train_clean), len(train_stego), len(val_clean), len(val_stego), len(test_clean), len(test_stego)])
    print(f"   Total: {total}")

    min_required = 50
    if len(train_clean) < min_required or len(train_stego) < min_required:
        print(f"\n[WARN] Low dataset count (< {min_required} pairs in train). Add more samples for robust training.")

    return True

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    ready = check_and_prepare_data()
    if ready:
        print("\nReady to run: python train_cnn.py")
