"""
Evaluate the trained model on the held-out test set.
Run after training: python evaluate_model.py
"""
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from config import MODEL_PATH, TEST_CLEAN_DIR, TEST_STEGO_DIR, IMG_SIZE
from utils.model_utils import normalize_rgb


def _list_images(folder):
    return list(folder.glob("*.[jJ][pP][gG]")) + list(folder.glob("*.[pP][nN][gG]"))


def _load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img_np = img.numpy().astype(np.uint8)
    img_norm = normalize_rgb(img_np)
    img = tf.convert_to_tensor(img_norm, dtype=tf.float32)
    return img


def main():
    test_clean_dir = Path(TEST_CLEAN_DIR)
    test_stego_dir = Path(TEST_STEGO_DIR)

    test_clean = _list_images(test_clean_dir)
    test_stego = _list_images(test_stego_dir)

    if not test_clean or not test_stego:
        print("Error: Test set is empty. Run python prepare_data.py first.")
        sys.exit(1)

    paths = [str(p) for p in test_clean] + [str(p) for p in test_stego]
    labels = [0] * len(test_clean) + [1] * len(test_stego)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda p, y: tf.py_function(
            lambda fp, lab: (_load_and_preprocess(fp), lab),
            [p, y],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    def _set_shapes(img, label):
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        label.set_shape(())
        return img, label

    dataset = dataset.map(_set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(MODEL_PATH)

    y_true, y_pred, y_proba = [], [], []
    for images, batch_labels in dataset:
        probs = model.predict(images, verbose=0).reshape(-1)
        preds = (probs > 0.5).astype(int)
        y_true.extend(batch_labels.numpy())
        y_pred.extend(preds)
        y_proba.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION METRICS")
    print("=" * 60)
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"AUC-ROC:      {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
