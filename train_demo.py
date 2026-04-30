"""
Train a tiny demo model for quick Streamlit showcasing.
This uses a small subset and 1-2 epochs to finish fast.
"""
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

# Demo settings
DEMO_IMG_SIZE = 128
DEMO_EPOCHS = 2
DEMO_BATCH_SIZE = 16
SAMPLES_PER_CLASS = 200
RANDOM_SEED = 42

TRAIN_CLEAN_DIR = "data/train_data/clean_images"
TRAIN_STEGO_DIR = "data/train_data/stego_images"
MODEL_PATH = "model/cnn_model.keras"


def _list_images(folder: Path):
    return list(folder.glob("*.[jJ][pP][gG]")) + list(folder.glob("*.[jJ][pP][eE][gG]")) + list(folder.glob("*.[pP][nN][gG]"))


def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [DEMO_IMG_SIZE, DEMO_IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def create_dataset(paths, labels, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, l: tf.py_function(lambda fp, lab: load_and_preprocess(fp, lab), [p, l], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    def _set_shapes(img, label):
        img.set_shape((DEMO_IMG_SIZE, DEMO_IMG_SIZE, 3))
        label.set_shape(())
        return img, label

    ds = ds.map(_set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    rng = random.Random(RANDOM_SEED)

    clean_dir = Path(TRAIN_CLEAN_DIR)
    stego_dir = Path(TRAIN_STEGO_DIR)

    clean_images = _list_images(clean_dir)
    stego_images = _list_images(stego_dir)

    if not clean_images or not stego_images:
        print("Error: Missing training data. Run prepare_data.py first.")
        return

    rng.shuffle(clean_images)
    rng.shuffle(stego_images)

    clean_images = clean_images[:SAMPLES_PER_CLASS]
    stego_images = stego_images[:SAMPLES_PER_CLASS]

    paths = [str(p) for p in clean_images] + [str(p) for p in stego_images]
    labels = [0] * len(clean_images) + [1] * len(stego_images)

    dataset = create_dataset(paths, labels, DEMO_BATCH_SIZE, shuffle=True)

    model = Sequential([
        keras.layers.Input(shape=(DEMO_IMG_SIZE, DEMO_IMG_SIZE, 3)),
        Conv2D(16, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2)),
        GlobalAveragePooling2D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(dataset, epochs=DEMO_EPOCHS, verbose=1)

    Path("model").mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved demo model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
