import os
from pathlib import Path
import random
import numpy as np
import tensorflow as tf  # type: ignore

from config import (
    BEST_MODEL_PATH,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    IMG_SIZE,
    LEARNING_RATE,
    MODEL_PATH,
    RANDOM_SEED,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    TRAIN_CLEAN_DIR,
    TRAIN_STEGO_DIR,
    VAL_CLEAN_DIR,
    VAL_STEGO_DIR,
)


def load_paths(folder):
    p = Path(folder)
    return list(p.glob("*.[jJ][pP][gG]")) + list(p.glob("*.[pP][nN][gG]")) + list(p.glob("*.[jJ][pP][eE][gG]"))


def preprocess_image_train(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    return img, label


def preprocess_image_val(path, label):
    """Read, decode, resize, and normalize validation image."""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def create_dataset(data, is_training=True):
    """Build high-performance tf.data pipeline."""
    paths = [x[0] for x in data]
    labels = [x[1] for x in data]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(data), 4096), seed=RANDOM_SEED)
        ds = ds.map(preprocess_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(preprocess_image_val, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_steganalysis_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="spatial_input")

    x = tf.keras.layers.Conv2D(32, (5, 5), padding="same", use_bias=False, name="hpf_conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="hpf_bn1")(x)
    x = tf.keras.layers.Activation(tf.nn.leaky_relu, name="hpf_act1")(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), name="hpf_pool1")(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="block2_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="block2_bn")(x)
    x = tf.keras.layers.Activation(tf.nn.leaky_relu, name="block2_act")(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), name="block2_pool")(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="block3_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="block3_bn")(x)
    x = tf.keras.layers.Activation(tf.nn.leaky_relu, name="block3_act")(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), name="block3_pool")(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="block4_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="block4_bn")(x)
    x = tf.keras.layers.Activation(tf.nn.leaky_relu, name="block4_act")(x)
    x = tf.keras.layers.AveragePooling2D((2, 2), name="block4_pool")(x)

    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", use_bias=False, name="block5_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="block5_bn")(x)
    x = tf.keras.layers.Activation(tf.nn.leaky_relu, name="block5_act")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    x = tf.keras.layers.Dense(256, use_bias=False, name="dense_1")(x)
    x = tf.keras.layers.BatchNormalization(name="dense_bn1")(x)
    x = tf.keras.layers.Activation(tf.nn.leaky_relu, name="dense_act1")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout_1")(x)

    x = tf.keras.layers.Dense(64, use_bias=False, name="dense_2")(x)
    x = tf.keras.layers.BatchNormalization(name="dense_bn2")(x)
    x = tf.keras.layers.Activation(tf.nn.leaky_relu, name="dense_act2")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="threat_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="StegoShield_CNN")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.03),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    """Train StegoShield CNN."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    train_clean = load_paths(TRAIN_CLEAN_DIR)
    train_stego = load_paths(TRAIN_STEGO_DIR)
    val_clean = load_paths(VAL_CLEAN_DIR)
    val_stego = load_paths(VAL_STEGO_DIR)

    print("=" * 60)
    print("STEGOSHIELD CNN TRAINING PIPELINE")
    print("=" * 60)
    print(f"Training Artifacts:   Clean={len(train_clean)} | Stego={len(train_stego)} (Total: {len(train_clean)+len(train_stego)})")
    print(f"Validation Artifacts: Clean={len(val_clean)} | Stego={len(val_stego)} (Total: {len(val_clean)+len(val_stego)})")
    print("=" * 60)

    if not train_clean or not train_stego:
        print("[ERROR] Dataset empty. Run python prepare_data.py first.")
        return

    train_data = [(str(p), 0) for p in train_clean] + [(str(p), 1) for p in train_stego]
    val_data = [(str(p), 0) for p in val_clean] + [(str(p), 1) for p in val_stego]

    random.shuffle(train_data)
    random.shuffle(val_data)

    train_dataset = create_dataset(train_data, is_training=True)
    val_dataset = create_dataset(val_data, is_training=False)

    model = build_steganalysis_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("\n[INFO] Starting Neural Training Pass...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save final model
    model.save(MODEL_PATH)
    print(f"\n[OK] Model successfully trained and saved to: {MODEL_PATH}")
    print(f"[OK] Best Checkpoint saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
