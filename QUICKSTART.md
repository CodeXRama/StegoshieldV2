# ⚡ StegoShield Quick Start Guide

## 5-Minute Setup

### Step 1: Install (1 minute)
```bash
# Activate virtual environment
source venv/Scripts/activate   # Windows
# or
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data (2 minutes)
```bash
# Create data directory structure
mkdir -p data/clean_images
mkdir -p data/stego_images

# Copy your images:
# - Normal images → data/clean_images/
# - Images with hidden data → data/stego_images/

# Verify structure
python prepare_data.py

# This will create:
# - data/train/clean, data/train/stego
# - data/test/clean, data/test/stego
```

**Recommended:** At least 100 images per category

### Step 3: Train Model (30-60 minutes depending on data size)
```bash
python train_cnn.py
```

Watch the output:
```
Starting training...
Epoch 1/50
...
Epoch 15/50 - val_auc: 0.8234 (best improvement)
...
Model successfully saved to model/cnn_model.keras
✓ Training history saved to model/training_history.png
```

### Step 4: Launch Web App (1 minute)
```bash
streamlit run app/app.py
```

Open: **http://localhost:8501** 🎉

---

## 📤 Using the Web App

1. **Upload Image**
   - Click "Upload target image" button
   - Select JPG/PNG file

2. **Run Detection**
   - Click "🔍 INITIALIZE NEURAL SCAN" button
   - Wait 2-5 seconds for analysis

3. **Read Results**
   - **STATUS**: SAFE ✅ | SUSPICIOUS ⚠️ | THREAT 🚨
   - **CONFIDENCE**: Percentage of detected steganography
   - **RISK LEVEL**: LOW 🟢 | MEDIUM 🟠 | HIGH 🔴

4. **Adjust Sensitivity** (Optional)
   - Use "Detection Threshold" slider
   - Lower = more sensitive
   - Higher = fewer false positives

---

## 🎯 Interpreting Results

### SAFE ✅ (< 50% confidence)
- No steganography detected
- Image appears clean
- Normal pixel distribution

### SUSPICIOUS ⚠️ (50-70% confidence)
- Possible hidden data
- Minor anomalies detected
- Manual review recommended

### THREAT 🚨 (> 70% confidence)
- High probability of steganography
- Suspicious pixel patterns
- Further investigation needed

---

## 📊 Example Output

```
STATUS: ✅ SAFE
CONFIDENCE SCORE: 42.37%
RISK LEVEL: 🟢 LOW

🧠 AI DIAGNOSTIC: Standard pixel distribution detected. 
No steganographic payload identified.

Raw Confidence Score: 0.423678
Classification: CLEAN
Steganography Detected: No

Model: CNN (64×64 HPF-preprocessed)
Input Processing: High-Pass Filter (Laplacian)
Sensitivity: 50.0%

[████████░░░░░░░░░░░] 42% Steganography Probability
```

---

## 🔧 Common Commands

| Task | Command |
|------|---------|
| Check data | `python prepare_data.py` |
| Train model | `python train_cnn.py` |
| Launch app | `streamlit run app/app.py` |
| Test system | `python test_system.py` |
| View logs | Check terminal output |

---

## ⚠️ Troubleshooting Quick Links

| Problem | Solution |
|---------|----------|
| "Found 0 images" | Run `python prepare_data.py` |
| Low accuracy | See [ACCURACY_GUIDE.md](ACCURACY_GUIDE.md) |
| Model not found | Run `python train_cnn.py` first |
| Streamlit errors | Check Python version >= 3.8 |
| Memory issues | Reduce `BATCH_SIZE` in config.py |

---

## 📈 Performance Tips

1. **Use high-quality images** (min 256x256)
2. **Balance datasets** (equal clean/stego images)
3. **Large training set** (300+ images per class for best results)
4. **Let it train** (aim for 30-50 epochs)
5. **Monitor metrics** (watch for overfitting)

---

## 🔐 Security Notes

⚠️ **This tool is for authorized forensic analysis only**
- Use on your own systems only
- Don't use for unauthorized analysis
- Results are not 100% accurate - they're probabilistic
- False positives/negatives can occur

---

## 📞 Quick Reference

**Main Files:**
- `train_cnn.py` - Training script
- `app/app.py` - Web interface
- `utils/model_utils.py` - Core functions
- `config.py` - Settings
- `model/cnn_model.keras` - Trained model

**Quick Test:**
```bash
python test_system.py
```

This checks:
- ✓ Model loads correctly
- ✓ Training data exists
- ✓ Predictions work
- ✓ System is ready

---

## 🎓 Learning More

- See [README.md](README.md) for detailed documentation
- See [FIXES.md](FIXES.md) for technical details
- See [ACCURACY_GUIDE.md](ACCURACY_GUIDE.md) for improving results

---

**Ready?** Start with:
```bash
python prepare_data.py
python train_cnn.py
streamlit run app/app.py
```
