# StegoShield — Complete Project Explanation

> A plain-English guide to every part of this project: what it does, how it works, and why each piece exists.

---

## What Does This Project Do?

**StegoShield** detects **steganography** in images.

Steganography is the act of hiding secret data (messages, files) inside a normal-looking image by slightly changing pixel values — usually the **least significant bits (LSB)**. To the human eye, the image looks identical. But a trained neural network can detect the subtle statistical anomalies left behind.

**The user flow is simple:**
1. Open the web app in a browser
2. Upload a suspicious image
3. The system runs the image through a trained CNN model
4. It returns a verdict: **STEGO DETECTED** (red) or **CLEAN PASS** (green)

---

## Project Structure at a Glance

```
StegoShield/
├── config.py              ← All settings (image size, paths, thresholds)
├── prepare_data.py        ← Organizes raw images into train/val/test folders
├── train_cnn.py           ← Trains the CNN model
├── requirements.txt       ← Python packages needed
├── README.md              ← Project overview
│
├── app/
│   ├── app.py             ← Streamlit launcher (entry point)
│   ├── web_server.py      ← FastAPI backend (all the real logic)
│   └── static/
│       └── index.html     ← The entire web UI (single file)
│
├── utils/
│   ├── model_utils.py     ← Loading model, preprocessing images, running inference
│   ├── forensics_utils.py ← ELA, noise maps, EXIF extraction, histograms
│   └── db_utils.py        ← SQLite database read/write
│
├── model/
│   ├── model.keras        ← The trained neural network weights (43 MB)
│   ├── calibration.json   ← Probability calibration settings
│   └── model_metadata.json← Model accuracy, precision, recall stats
│
├── data/
│   ├── train_data/        ← Images used to train the model
│   ├── val_data/          ← Images used during training to check progress
│   └── test_data/         ← Images used to evaluate final model accuracy
│
└── outputs/
    └── stegoshield_audit.db ← SQLite database of every scan ever performed
```

---

## Part 1 — The Dataset (data/)

The model needs to learn what a "clean" image looks like vs a "stego" image.

### Folder Structure
```
data/
├── train_data/
│   ├── clean_images/   ← Normal unmodified photos
│   └── stego_images/   ← Same photos but with hidden data embedded in LSBs
├── val_data/
│   ├── clean_images/
│   └── stego_images/
└── test_data/
    ├── clean_images/
    └── stego_images/
```

- **Train** (80%): Images the model learns from
- **Val** (10%): Images checked during training to see if the model is improving
- **Test** (10%): Images only used at the very end to measure real performance

### Dataset Stats (from model_metadata.json)
| Split | Count |
|-------|-------|
| Training samples | 10,368 |
| Validation samples | 1,280 |
| Test samples | 1,152 |

### prepare_data.py — How Data Gets Organized

Before training, you run this script. It:
1. Reads all images from train_data/clean_images/ and train_data/stego_images/
2. Finds **matching pairs** — each clean image must have a corresponding stego version with the same filename
3. Randomly splits them: 80% train, 10% val, 10% test
4. **Copies** (not moves) the split images to the correct folders — safe, no data loss

---

## Part 2 — Configuration (config.py)

One central file that controls all settings. Every other script reads from here.

```python
IMG_SIZE = 256           # Images resized to 256x256 for training
                         # (512x512 during inference for better accuracy)
BATCH_SIZE = 32          # Train 32 images at a time
EPOCHS = 15              # Maximum training rounds
LEARNING_RATE = 0.0005   # How fast the model adjusts
RANDOM_SEED = 42         # Ensures reproducible results

DETECTION_THRESHOLD = 0.5   # Default: probability > 0.5 = stego
USE_RESIDUAL = False         # Whether to use residual preprocessing
ALLOWED_SCAN_ROOT = "data"   # Batch scan restricted to this folder (security)
APP_MAX_FILE_SIZE_MB = 10    # Maximum upload size
```

---

## Part 3 — The CNN Model (train_cnn.py)

This is the brain of the project. A **Convolutional Neural Network (CNN)** trained to classify images as stego or clean.

### Why CNN?
CNNs look at small regions (patches) of the image rather than individual pixels. This lets them detect subtle spatial patterns — exactly what steganography leaves behind.

### The Architecture

```
Input Image (256x256x3)
    ↓
[Block 1] Conv2D(32 filters, 5x5) → BatchNorm → LeakyReLU → AveragePool(2x2)
    ↓ Output: 128x128x32
[Block 2] Conv2D(64 filters, 3x3) → BatchNorm → LeakyReLU → AveragePool(2x2)
    ↓ Output: 64x64x64
[Block 3] Conv2D(128 filters, 3x3) → BatchNorm → LeakyReLU → AveragePool(2x2)
    ↓ Output: 32x32x128
[Block 4] Conv2D(256 filters, 3x3) → BatchNorm → LeakyReLU → AveragePool(2x2)
    ↓ Output: 16x16x256
[Block 5] Conv2D(512 filters, 3x3) → BatchNorm → LeakyReLU → GlobalAveragePool
    ↓ Output: 512 (single vector)
[Dense 1] Dense(256) → BatchNorm → LeakyReLU → Dropout(0.4)
[Dense 2] Dense(64)  → BatchNorm → LeakyReLU → Dropout(0.2)
[Output]  Dense(1, sigmoid) → probability between 0 and 1
```

### Why These Specific Choices?

| Choice | Reason |
|--------|--------|
| 5x5 kernel in Block 1 | Captures larger spatial patterns at first layer |
| AveragePooling (not MaxPooling) | Preserves average noise energy — MaxPool discards subtle signals |
| LeakyReLU (not ReLU) | Keeps small negative values — steganographic residuals can be negative |
| BatchNormalization | Stabilizes training, prevents overfitting to image brightness |
| Dropout(0.4, 0.2) | Forces model to not memorize individual images |
| Sigmoid output | Produces a probability (0 to 1) |
| label_smoothing=0.03 | Prevents model from being overconfident (generalizes better) |

### Training Callbacks
- **ModelCheckpoint**: Saves best model when val_accuracy improves
- **ReduceLROnPlateau**: Halves learning rate if val_loss stalls for 2 epochs
- **EarlyStopping**: Stops training if val_loss doesnt improve for 5 epochs

### Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 72% |
| Precision | 86.67% |
| Recall | 52% |
| F1 Score | 0.65 |
| ROC-AUC | 0.7988 |
| Avg Inference Time | ~305 ms (CPU) |

> Precision 87% means: when it says stego, its right 87% of the time.
> Recall 52% means: it catches 52% of all stego images.

---

## Part 4 — Utility Functions (utils/)

### utils/model_utils.py — The Inference Pipeline

#### load_and_preprocess_image()
1. Open image with PIL
2. Convert to RGB (handles grayscale, RGBA, etc.)
3. Resize to 512x512
4. Divide pixel values by 255.0 → range [0, 1]

#### apply_temperature(probability, temperature=0.8522)
Temperature Scaling — post-processing that makes probabilities more reliable.
Temperature < 1 sharpens predictions (pushes away from 0.5).
Found by grid search on validation set.

#### predict_image()
1. Add batch dimension: (H,W,3) → (1,H,W,3)
2. model.predict() → raw probability
3. apply_temperature() → calibrated probability
4. Compare to threshold → CLEAN / SUSPICIOUS / DETECTED

#### load_model()
Loads the .keras file. Registers custom objects for EfficientNet compatibility.

---

### utils/forensics_utils.py — Image Analysis Tools

#### compute_ela() — Error Level Analysis
1. Take original image
2. Re-compress as JPEG at quality=90
3. Compute pixel difference: original - recompressed, amplified 10x
Why: Areas with hidden data compress differently and appear as bright spots.

#### compute_noise_map() — Laplacian Noise Map
Applies Laplacian filter to grayscale image. Highlights rapid pixel changes.
Steganography adds systematic noise that becomes visible here.

#### compute_histograms()
Returns frequency distribution of pixel values for R, G, B, and Gray channels.
Steganography distorts these distributions in subtle ways.

#### extract_metadata()
Reads EXIF tags using ExifRead library. Decodes GPS coordinates from DMS format.

---

### utils/db_utils.py — SQLite Database

Every scan is logged to outputs/stegoshield_audit.db.

#### Table: scan_records
```
id              - Auto-incrementing row ID
timestamp       - When the scan happened
filename        - Name of uploaded file
sha256          - Unique fingerprint of the file
file_size_bytes - Raw file size
dimensions      - "1024 x 768 PX"
probability     - Model output (0.0 to 1.0)
classification  - "STEGO DETECTED" or "CLEAN PASS"
threshold       - Threshold used
entropy         - Shannon entropy of file bytes
runtime_ms      - How long inference took
status          - "SUCCESS" or "FAILED"
error_message   - Error details if failed
```

Uses threading.Lock() to prevent data corruption from concurrent requests.

---

## Part 5 — The Backend API (app/web_server.py)

Built with **FastAPI** running on port 8000.

### Why FastAPI?
- Handles file uploads natively (UploadFile)
- Async support — doesnt freeze while processing large files
- Much faster than Flask for API workloads

### Model Loading
The model is loaded ONCE at startup into memory (MODEL_INSTANCE global variable).
Not reloaded on every request — avoids loading 43MB from disk each time.

---

### All API Endpoints

#### GET /
Serves index.html to the browser. The entire web UI.

---

#### POST /api/analyze
Analyzes a single uploaded image. The main endpoint.

Full pipeline:
```
Step 1 — Validate: empty file? too large? wrong format?
Step 2 — Decode: open with PIL, force-load to catch corrupted files
Step 3 — Preprocess: convert to RGB, resize to 512x512, normalize
Step 4 — Inference: model.predict() → apply_temperature() → verdict
Step 5 — Post-process: SHA-256, Shannon entropy, residual image, base64 encode
Step 6 — Log to SQLite
Step 7 — Return JSON response
```

Returns:
- verdict (STEGO DETECTED / CLEAN PASS)
- confidence and probability scores
- file metadata (dimensions, format, size, entropy, SHA-256)
- base64-encoded source and residual images
- timing breakdown (decode, preprocess, inference, postprocess)
- plain-English interpretation

---

#### POST /api/batch
Scans an entire folder of images.
Security: folder must be inside data/ — prevents scanning system paths.
Resilient: if one file fails, logs it and continues with the rest.

---

#### GET /api/history
Returns last 100 scan records from SQLite for the History tab.

#### POST /api/clear-history
Deletes all scan records. Used by the Clear button in History tab.

#### GET /api/export-history-csv
Downloads all history as a CSV file attachment.

#### GET /api/export-history-json
Downloads all history as JSON.

#### GET /api/model-intel
Returns model specs shown in the Model Intel tab:
architecture name, version, input size, calibration temperature, accuracy metrics.

---

### Helper Functions in web_server.py

- calculate_shannon_entropy(): Measures randomness of file bytes. Encrypted/compressed payloads have high entropy.
- compute_sha256(): Cryptographic hash for audit trail.
- generate_srm_residual(): 5x5 high-pass spatial filter revealing micro-pixel variations.
- image_to_base64(): Encodes PIL images as base64 data URLs for immediate frontend rendering.

---

## Part 6 — The Frontend (app/static/index.html)

A sleek, instrument-grade workstation frontend using Tailwind CSS and vanilla JavaScript.

### Navigation Directory
- ANALYZE: Primary target analysis console with interactive forensic exploration
- BATCH SCAN: Scan entire directories in bulk
- HISTORY: View scan audit database and export CSV / JSON
- MODEL INTEL: Neural architecture specs and calibrated metrics

### Analyze Section Workflow & Sub-Sections

The Analyze section consists of:
1. **Intake & Staging Area**: File dropzone, target reticle, format detection, decision threshold slider ($\tau$), and quick-access buttons for direct forensic exploration.
2. **Execution Phases Tracker**: 4-phase in-flight animation with live millisecond timer and terminal logs.
3. **Forensic Result Dossier with 4 Interactive Sub-Sections**:
   - **01 // AI Detection & Residual**: Primary model verdict, calibrated confidence, probability vector gauge with threshold tick, side-by-side Source vs SRM-30 High-Pass Residual, technical telemetry matrix, forensic assessment statement, and dossier export (PDF/JSON).
   - **02 // Error Level Analysis (ELA)**: Recompression delta analysis with interactive **JPEG Quality Slider** (10% to 100%, with 75%, 90%, 95% presets), **Error Amplification Gain Slider** (1x to 40x), Contrast Invert toggle, False-Color Heatmap toggle, reset button, and live MAE / peak delta telemetry.
   - **03 // Noise & Histogram**: High-pass Laplacian noise inspection paired with an interactive **256-Bin HTML5 Canvas Histogram**. Features **Noise Gain Slider** (0.5x to 10.0x), **Histogram Equalization Toggle**, **Channel Filter Selectors** (`ALL`, `RED`, `GREEN`, `BLUE`, `GRAY`), **Histogram Resolution Slider** (32 to 256 bins), interactive crosshair hover coordinates, noise variance ($\sigma^2$), and LSB bit-parity indicators.
   - **04 // Image Metadata & EXIF**: Full structural and EXIF inspection with interactive **Live Search Filter Input**, category filter tabs (`ALL`, `CONTAINER`, `EXIF & HARDWARE`, `GPS`, `INTEGRITY`), **One-Click JSON Copy**, EXIF header verification badge, software footprint detector, and GPS geolocation with direct Google Maps link.

### Key JavaScript

```
startScanSequence()
  1. Read file from input
  2. Switch to SCANNING state
  3. Animate pipeline steps (4 phases with timer)
  4. POST to /api/analyze with image + threshold
  5. When response arrives → displayResults()

displayResults(data)
  Fill result fields from API response
  Red colors for stego, green for clean
  Switch to RESULT state

switchState(state)
  Toggle visibility of intake / scanning / result panels
```

### Threshold Slider
Adjustable tau (τ) from 0.0 to 1.0. Default: 0.50.
Higher = fewer false alarms but misses more stego.
Lower = catches more stego but more false positives.

---

## Part 7 — The Launcher (app/app.py)

Entry point: python -m streamlit run app/app.py

```python
ensure_backend_server()   # Check if port 8000 in use
                          # If not → start FastAPI in background thread
st.set_page_config(...)   # Full-screen browser tab
st.markdown(hide_styles)  # Hide ALL Streamlit UI elements
components.iframe("http://127.0.0.1:8000")  # Embed FastAPI UI
```

Streamlit is only a shell. The real UI runs in FastAPI.
You can also go directly to http://127.0.0.1:8000 — same thing.

---

## Part 8 — Complete Data Flow

```
User uploads image in browser
         │
         ▼
index.html (JavaScript)
  └── fetch POST /api/analyze
         │
         ▼
web_server.py (FastAPI)
  ├── Validate → Decode (PIL) → Preprocess
  ├── model_utils.py → predict_image()
  │     ├── model.predict() → raw probability
  │     └── apply_temperature() → calibrated probability
  ├── Compare probability vs threshold → verdict
  ├── SHA-256 + Shannon entropy
  ├── Generate residual image
  ├── db_utils.py → log_scan_record() → SQLite
  └── Return JSON
         │
         ▼
index.html (JavaScript)
  └── displayResults() → show red/green verdict
```

---

## Part 9 — Why Each Library Was Used

| Library | Used For | Where |
|---------|---------|-------|
| TensorFlow / Keras | Building and running the CNN | train_cnn.py, model_utils.py |
| NumPy | Array operations on image data | model_utils.py, forensics_utils.py |
| Pillow (PIL) | Opening, resizing, converting images | web_server.py, model_utils.py |
| OpenCV (cv2) | Laplacian filter, Sobel, color conversion | forensics_utils.py, model_utils.py |
| scikit-image | Histogram equalization | forensics_utils.py |
| ExifRead | Reading EXIF metadata | forensics_utils.py |
| FastAPI | REST API framework with async support | web_server.py |
| Uvicorn | ASGI server that runs FastAPI | web_server.py, app.py |
| Streamlit | Auto-launcher + iframe wrapper | app.py |
| SQLite3 (stdlib) | Built-in database for scan history | db_utils.py |
| hashlib (stdlib) | SHA-256 fingerprinting | web_server.py |
| base64 (stdlib) | Encoding images for JSON transport | web_server.py |
| Tailwind CSS (CDN) | Styling the entire web UI | index.html |

---

*StegoShield v2.0.0*
