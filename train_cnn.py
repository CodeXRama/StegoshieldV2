import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    TRAIN_CLEAN_DIR,
    TRAIN_STEGO_DIR,
    VAL_CLEAN_DIR,
    VAL_STEGO_DIR,
    TEST_CLEAN_DIR,
    TEST_STEGO_DIR,
)

def normalize_rgb(img):
    """Normalize RGB image to [0, 1]"""
    return img.astype(np.float32) / 255.0

def load_and_preprocess_image(path, label):
    """Load and preprocess image"""
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)

    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Load image paths
train_clean_dir = Path(TRAIN_CLEAN_DIR)
train_stego_dir = Path(TRAIN_STEGO_DIR)
val_clean_dir = Path(VAL_CLEAN_DIR)
val_stego_dir = Path(VAL_STEGO_DIR)
test_clean_dir = Path(TEST_CLEAN_DIR)
test_stego_dir = Path(TEST_STEGO_DIR)

train_clean_images = sorted(list(train_clean_dir.glob("*.[jJ][pP][gG]")) + list(train_clean_dir.glob("*.[pP][nN][gG]")))
train_stego_images = sorted(list(train_stego_dir.glob("*.[jJ][pP][gG]")) + list(train_stego_dir.glob("*.[pP][nN][gG]")))
val_clean_images = sorted(list(val_clean_dir.glob("*.[jJ][pP][gG]")) + list(val_clean_dir.glob("*.[pP][nN][gG]")))
val_stego_images = sorted(list(val_stego_dir.glob("*.[jJ][pP][gG]")) + list(val_stego_dir.glob("*.[pP][nN][gG]")))
test_clean_images = sorted(list(test_clean_dir.glob("*.[jJ][pP][gG]")) + list(test_clean_dir.glob("*.[pP][nN][gG]")))
test_stego_images = sorted(list(test_stego_dir.glob("*.[jJ][pP][gG]")) + list(test_stego_dir.glob("*.[pP][nN][gG]")))

print(f"Found train clean: {len(train_clean_images)} | train stego: {len(train_stego_images)}")
print(f"Found val clean:   {len(val_clean_images)} | val stego:   {len(val_stego_images)}")
print(f"Found test clean:  {len(test_clean_images)} | test stego:  {len(test_stego_images)}")

if len(train_clean_images) == 0 or len(train_stego_images) == 0:
    print("Error: No images found in training directories!")
    sys.exit(1)

if len(val_clean_images) == 0 or len(val_stego_images) == 0:
    print("Error: No images found in validation directories!")
    print("Run: python prepare_data.py")
    sys.exit(1)

# Create datasets using tf.data (modern approach)
print("Creating tf.data pipelines...")
train_images = [(str(p), 0) for p in train_clean_images] + [(str(p), 1) for p in train_stego_images]
val_images = [(str(p), 0) for p in val_clean_images] + [(str(p), 1) for p in val_stego_images]

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(train_images)
rng.shuffle(val_images)

print(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")

# Create tf.data datasets
def create_dataset(image_list, batch_size=32, shuffle=True, augment=False):
    paths = [x[0] for x in image_list]
    labels = [x[1] for x in image_list]
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_list), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    
    dataset = dataset.map(
        lambda path, label: tf.py_function(
            lambda p, l: load_and_preprocess_image(p, l),
            [path, label],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    def _set_shapes(img, label):
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        label.set_shape(())
        return img, label

    dataset = dataset.map(_set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        pass
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_images, batch_size=BATCH_SIZE, shuffle=True, augment=False)
val_dataset = create_dataset(val_images, batch_size=BATCH_SIZE, shuffle=False, augment=False)

test_images = [(str(p), 0) for p in test_clean_images] + [(str(p), 1) for p in test_stego_images]
test_dataset = None
if test_images:
    test_dataset = create_dataset(test_images, batch_size=BATCH_SIZE, shuffle=False, augment=False)


# ===== BASIC MODEL ARCHITECTURE =====
print("\nBuilding CNN model...")
model = Sequential([
    keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    AveragePooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    AveragePooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    AveragePooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    AveragePooling2D(pool_size=(2, 2)),

    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Compile with better learning rate
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc'), 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall')]
)

print(model.summary())

# ===== TRAINING =====
print("\nStarting training...")

# Callbacks
early_stop = EarlyStopping(
    monitor='val_auc',
    patience=10,
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'model/best_model.keras',
    monitor='val_auc',
    save_best_only=True,
    verbose=1,
    mode='max'
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stop, reduce_lr, model_checkpoint],
    verbose=1
)

# ===== EVALUATION =====
print("\nEvaluating on validation set...")
val_loss, val_acc, val_auc, val_prec, val_rec = model.evaluate(val_dataset, verbose=0)

def evaluate_dataset(dataset, title):
    print(f"\nGenerating predictions for {title} metrics...")
    y_true_list = []
    y_pred_list = []
    y_pred_proba_list = []

    for images, labels in dataset:
        y_true_list.extend(labels.numpy())
        proba = model.predict(images, verbose=0)
        y_pred_proba_list.extend(proba)
        y_pred_list.extend((proba > 0.5).astype(int).flatten())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_pred_proba = np.array(y_pred_proba_list).flatten()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "="*60)
    print(f"DETAILED MODEL EVALUATION METRICS ({title})")
    print("="*60)
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"AUC-ROC:      {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    print("="*60)

evaluate_dataset(val_dataset, "VALIDATION")
if test_dataset is not None:
    evaluate_dataset(test_dataset, "TEST")
else:
    print("\n⚠️  No test data found. Add files to data/test/clean and data/test/stego.")

# Save model
Path("model").mkdir(exist_ok=True)
model.save("model/cnn_model.keras")
print("\n✓ Model successfully saved to model/cnn_model.keras")
print("✓ Best model saved to model/best_model.keras")

# Plot training history
if history:
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title('Model AUC')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=100)
    print("✓ Training history saved to model/training_history.png")
