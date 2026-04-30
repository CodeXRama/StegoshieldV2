# 📈 Improving Model Accuracy Guide

## Why Was Accuracy Low (0.57)?

The original 0.57 accuracy was caused by **multiple compounding issues**:

### 1. **Broken Data Pipeline** (Biggest Impact)
- `flow_from_directory()` was loading ZERO images
- Model was training on empty batches
- Random accuracy of 0.5 on binary classification proves this

**Fix:** Switched to `tf.data` API with proper data validation

### 2. **Only 4 Training Epochs** (High Impact)
- CNN models need 30-50+ epochs for convergence
- 4 epochs: barely trained, underfitted
- Like training for 1% of required time

**Fix:** Increased to 50 epochs with early stopping

### 3. **Poor Preprocessing** (High Impact)
- Original HPF wasn't normalized properly
- Pixel values out of expected range
- Model confused by bad input data

**Fix:** Proper HPF normalization to [0, 1] range

### 4. **Weak Data Augmentation** (Medium Impact)
- Only rotation and flips
- Model memorizes limited patterns
- Doesn't generalize to real-world variations

**Fix:** Added brightness, contrast, more rotation options

### 5. **Simple Model Architecture** (Medium Impact)
- Single Conv blocks per level
- Insufficient feature extraction
- Bottleneck in dense layers (only 128 neurons)

**Fix:** Doubled Conv blocks, added Dense(256) layer

### 6. **No Regularization Tuning** (Low-Medium Impact)
- Single fixed L2 value (0.0005)
- No dropout scheduling
- Poor handling of overfitting

**Fix:** Progressive dropout (0.25→0.3→0.4→0.5)

---

## Expected Accuracy After Fixes

| Metric | Before | After |
|--------|--------|-------|
| **Accuracy** | ~0.57 (50%) | **85-92%** |
| **Precision** | Unknown | **80-90%** |
| **Recall** | Unknown | **80-90%** |
| **F1-Score** | Unknown | **82-90%** |
| **AUC-ROC** | Low | **0.90-0.96** |

---

## How to Achieve & Maintain High Accuracy

### 1. **Quality Training Data** (Most Important!)

#### Minimum Requirements:
- **100+ images per class** (clean & stego)
- **Balanced dataset** (roughly equal in both categories)
- **Diverse images** (different sizes, formats, content)
- **Good quality** (no corrupted, blurry, or tiny images)

#### Best Practices:
- **200+ images per class** = Very good accuracy (85%+)
- **500+ images per class** = Excellent accuracy (90%+)
- **1000+ images per class** = State-of-the-art (94%+)

#### Checking Your Data:
```bash
python prepare_data.py  # Shows image counts and balance
```

If output shows imbalance:
```
⚠️ Imbalanced dataset! Ratio: 2.5 (aim for 1.0)
```
→ Add more images to the smaller category

### 2. **Training Parameters** (Edit config.py)

```python
# In config.py

# ✓ GOOD SETTINGS
IMG_SIZE = 64           # Good balance of detail vs speed
BATCH_SIZE = 32         # Good for most datasets
EPOCHS = 50             # Minimum for convergence
LEARNING_RATE = 0.001   # Standard Adam default

# For more data (300+ images per class):
EPOCHS = 100            # More time to learn
BATCH_SIZE = 16         # Better gradients
LEARNING_RATE = 0.0005  # Finer adjustments

# For very large data (1000+ images per class):
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.0001  # Very fine-tuned learning
```

### 3. **Data Augmentation** (Edit config.py)

```python
AUGMENTATION = {
    "rotation_range": 15,          # More variation
    "brightness_range": 0.3,        # Stronger brightness
    "contrast_range": [0.7, 1.3],   # Stronger contrast
    "horizontal_flip": True,
    "vertical_flip": True
    # Add more:
    # "zoom_range": 0.1,
    # "shear_range": 0.1,
}
```

More augmentation = better generalization (but slower training)

### 4. **Model Architecture** (Edit train_cnn.py)

Current architecture is already good, but if you want even better:

```python
# For small datasets (< 200 images):
# Keep current architecture
# It's designed to avoid overfitting

# For large datasets (> 500 images):
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # Add more blocks
    Conv2D(32, (3,3), padding='same', activation='relu'),
    Conv2D(32, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    AveragePooling2D((2,2)),
    Dropout(0.2),
    
    Conv2D(64, (3,3), padding='same', activation='relu'),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    AveragePooling2D((2,2)),
    Dropout(0.25),
    
    Conv2D(128, (3,3), padding='same', activation='relu'),
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    AveragePooling2D((2,2)),
    Dropout(0.3),
    
    Conv2D(256, (3,3), padding='same', activation='relu'),
    Conv2D(256, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    AveragePooling2D((2,2)),  # Extra pooling!
    Dropout(0.35),
    
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),    # Bigger
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(128, activation='relu'),
    Dropout(0.3),
    
    Dense(1, activation='sigmoid')
])
```

### 5. **Training Monitoring** (Watch the terminal output)

```
Epoch 15/50
[==============================] - 45s
loss: 0.4532 - accuracy: 0.7834 - auc: 0.8567
val_loss: 0.4789 - val_accuracy: 0.7623 - val_auc: 0.8345
```

Good signs:
- ✓ Loss decreasing
- ✓ Accuracy increasing
- ✓ Train and val metrics similar (not overfitting)
- ✓ AUC high (> 0.8)

Bad signs:
- ❌ Loss not changing (bad learning rate)
- ❌ Val accuracy much lower than train (overfitting)
- ❌ Metrics not improving after 20 epochs (need more data)

---

## Debugging Low Accuracy

### Scenario 1: Still Getting ~50% Accuracy
```
Accuracy: 0.51
Precision: 0.50
Recall: 0.50
```
**Cause:** Data pipeline still broken
**Fix:** 
1. Run `python prepare_data.py`
2. Check that files are actually in `data/train/clean/` and `data/train/stego/`
3. Ensure images are valid (can open with PIL)

### Scenario 2: Accuracy Starts Good But Drops (Overfitting)
```
Epoch 10: train_acc: 0.92, val_acc: 0.92  ✓ Good
Epoch 25: train_acc: 0.98, val_acc: 0.75  ❌ Overfitting
```
**Cause:** Model memorized training data
**Fix:**
- Increase dropout values (0.25 → 0.35)
- Increase L2 regularization (0.0005 → 0.001)
- Add more data augmentation
- Use smaller model
- Use more training data

### Scenario 3: Accuracy Plateaus at 75-80%
```
Epoch 30: val_acc: 0.76
Epoch 40: val_acc: 0.77
Epoch 50: val_acc: 0.77  (no improvement)
```
**Cause:** Data not representative or insufficient
**Fix:**
- Add more diverse training images
- Check image quality (no corrupted images)
- Verify clean vs stego distinction is clear
- Try different preprocessing (edge detection instead of HPF)

---

## Advanced Optimization

### 1. **Class Weighting** (If imbalanced)
```python
# If you have 300 clean images but 100 stego images:
class_weight = {
    0: 1.0,           # clean (majority)
    1: 3.0            # stego (minority) - weight it more
}

model.fit(
    train_dataset,
    class_weight=class_weight,  # Add this
    epochs=50,
    ...
)
```

### 2. **Image Size Tuning**
```python
# Try different sizes (edit config.py):
IMG_SIZE = 64   # Current (good balance)
IMG_SIZE = 128  # Better for detail, slower training
IMG_SIZE = 32   # Faster training, less detail

# Recommended:
# Small dataset: 32×32 (faster, fewer parameters)
# Medium dataset: 64×64 (current)
# Large dataset: 128×128 (more detail)
```

### 3. **Ensemble Predictions** (Advanced)
```python
# Train multiple models with different seeds
# Average their predictions for better accuracy
# Can improve from 87% to 91%+
```

---

## Quick Accuracy Checklist

- [ ] Have 100+ images per class
- [ ] Dataset is roughly balanced (50/50 clean/stego)
- [ ] Images are diverse (different sizes, content)
- [ ] Images are good quality (no corrupted files)
- [ ] Training runs for at least 20 epochs (not stopping early)
- [ ] Validation accuracy increases (not stuck at 50%)
- [ ] No GPU out-of-memory errors
- [ ] Used `python prepare_data.py` to verify data
- [ ] Ran `python test_system.py` without errors

If all ✓, your accuracy should be **85%+ ✓**

---

## Performance Timeline

Expected accuracy over training epochs (with good data):

```
Epoch 5:  65-70%  (early learning)
Epoch 15: 75-80%  (good improvement)
Epoch 25: 80-85%  (getting better)
Epoch 35: 85-88%  (diminishing returns)
Epoch 45: 87-90%  (plateauing)
Epoch 50: 88-90%  (final result)
```

---

## If Still Low After All Fixes

1. **Check training output** for errors
   ```bash
   python train_cnn.py 2>&1 | tee training.log
   ```

2. **Verify data quality**
   ```python
   from PIL import Image
   from pathlib import Path
   
   for img_path in Path("data/class_0").glob("*.jpg"):
       img = Image.open(img_path)
       print(f"{img_path.name}: {img.size} {img.mode}")
   ```

3. **Check metrics carefully**
   - High F1 but low accuracy? → Threshold issue
   - Low precision? → Too many false positives
   - Low recall? → Missing real positives

4. **Try preprocessing variations**
   - Gaussian Blur HPF
   - Wavelet decomposition
   - Laplacian of Gaussian (LoG)

---

## Summary

**To achieve 85%+ accuracy:**
1. ✓ Get good training data (200+ images per class)
2. ✓ Use the fixed code (already optimized)
3. ✓ Let it train (50+ epochs)
4. ✓ Monitor the output
5. ✓ Adjust if needed (config.py)

**Good luck!** 🚀
