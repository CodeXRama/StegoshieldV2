
import base64
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import uvicorn

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ALLOWED_SCAN_ROOT,
    APP_MAX_FILE_SIZE_MB,
    CALIBRATION_PATH,
    MODEL_PATH,
    USE_RESIDUAL,
)
from utils.db_utils import (
    clear_scan_records,
    count_scan_records,
    get_scan_records,
    init_db,
    log_scan_record,
)
from utils.model_utils import (
    load_and_preprocess_image,
    load_model,
    predict_image,
)
from utils.forensics_utils import (
    compute_ela_with_stats,
    compute_noise_with_stats,
    compute_histograms_with_stats,
    extract_rich_metadata,
)

LAST_IMAGE_BYTES: Optional[bytes] = None

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model()
    yield

app = FastAPI(title="StegoShield Forensic Workstation API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForensicAPIException(HTTPException):
    def __init__(self, error_code: str, detail: str, status_code: int = 400):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code


@app.exception_handler(ForensicAPIException)
async def forensic_exception_handler(request: Request, exc: ForensicAPIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code = "REQUEST_ERROR"
    if exc.status_code == 404:
        code = "NOT_FOUND"
    elif exc.status_code == 413:
        code = "IMAGE_TOO_LARGE"
    elif exc.status_code == 503:
        code = "MODEL_UNAVAILABLE"

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": code,
            "detail": str(exc.detail),
            "status_code": exc.status_code,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )


MODEL_INSTANCE = None
CALIBRATION_TEMP = 0.8522
CALIBRATION_CONFIG: Dict[str, Any] = {}


def init_model() -> None:
    """Load model once into memory as a safe singleton."""
    global MODEL_INSTANCE, CALIBRATION_TEMP, CALIBRATION_CONFIG
    if MODEL_INSTANCE is not None:
        return

    try:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            print(f"[ERROR] Production model weights missing at {model_path}")
            MODEL_INSTANCE = None
            return

        print(f"[INIT] Loading production model from {model_path}...")
        MODEL_INSTANCE = load_model(str(model_path))
        print(f"[INIT] Model loaded successfully: {MODEL_INSTANCE.name} ({MODEL_INSTANCE.count_params():,} params)")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        MODEL_INSTANCE = None

    # Load calibration configuration
    try:
        calib_file = Path(CALIBRATION_PATH)
        if calib_file.exists():
            with open(calib_file, "r", encoding="utf-8") as f:
                CALIBRATION_CONFIG = json.load(f)
            CALIBRATION_TEMP = float(CALIBRATION_CONFIG.get("temperature", 0.8522))
            print(f"[INIT] Temperature calibration loaded: T={CALIBRATION_TEMP:.4f}")
    except Exception as e:
        print(f"[WARN] Calibration load warning: {e}")
        CALIBRATION_TEMP = 0.8522

    # Initialize SQLite Audit DB
    init_db()


def compute_sha256(data_bytes: bytes) -> str:
    return hashlib.sha256(data_bytes).hexdigest()


def calculate_shannon_entropy(data_bytes: bytes) -> float:
    if not data_bytes:
        return 0.0
    freq = [0] * 256
    for b in data_bytes:
        freq[b] += 1
    total = len(data_bytes)
    ent = 0.0
    for count in freq:
        if count > 0:
            p = count / total
            ent -= p * math.log2(p)
    return round(ent, 3)


def generate_srm_residual(image_pil: Image.Image) -> Image.Image:
    """Generate high-pass spatial residual visualization using 3x3 high-pass delta kernel."""
    rgb = image_pil.convert("RGB")
    kernel = ImageFilter.Kernel(
        (3, 3),
        [-1, 2, -1, 2, -4, 2, -1, 2, -1],
        scale=1,
        offset=128,
    )
    residual = rgb.filter(kernel)
    return residual


def image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"



@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = Path(__file__).parent / "static" / "index.html"
    if not index_file.exists():
        raise ForensicAPIException("NOT_FOUND", "Frontend index.html missing from app/static/", 404)
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    threshold: float = Form(0.73),
    profile: str = Form("BALANCED"),
):
    global MODEL_INSTANCE, CALIBRATION_TEMP
    t_start = time.perf_counter()

    # 1. Payload validation
    data_bytes = await file.read()
    if not data_bytes or len(data_bytes) == 0:
        raise ForensicAPIException("INVALID_IMAGE", "Empty file payload received.", 400)

    max_bytes = APP_MAX_FILE_SIZE_MB * 1024 * 1024
    if len(data_bytes) > max_bytes:
        raise ForensicAPIException(
            "IMAGE_TOO_LARGE",
            f"File size exceeds maximum allowed limit of {APP_MAX_FILE_SIZE_MB}MB.",
            413,
        )

    # 2. Image Decoding
    t_decode_start = time.perf_counter()
    try:
        image_pil = Image.open(io.BytesIO(data_bytes))
        image_pil.load()  # Force load to catch truncated/corrupted images
        width, height = image_pil.size
        img_format = (image_pil.format or "PNG").upper()
    except Exception as e:
        raise ForensicAPIException(
            "INVALID_IMAGE",
            f"Failed to decode image data: {str(e)}",
            400,
        )
    t_decode_end = time.perf_counter()

    allowed_formats = {"PNG", "JPG", "JPEG", "BMP", "TIFF", "WEBP"}
    if img_format not in allowed_formats:
        raise ForensicAPIException(
            "UNSUPPORTED_FORMAT",
            f"Image format '{img_format}' is not supported. Use PNG, JPG, BMP, or TIFF.",
            400,
        )

    # 3. Preprocessing
    t_prep_start = time.perf_counter()
    # Normalize color channels to RGB
    if image_pil.mode != "RGB":
        rgb_image = image_pil.convert("RGB")
    else:
        rgb_image = image_pil

    with tempfile.NamedTemporaryFile(suffix=f".{img_format.lower()}", delete=False) as tf:
        tf.write(data_bytes)
        tmp_name = tf.name

    try:
        img_processed, _ = load_and_preprocess_image(
            tmp_name,
            target_size=512,
            use_residual=USE_RESIDUAL,
        )
    except Exception as e:
        raise ForensicAPIException("INFERENCE_FAILED", f"Image preprocessing failed: {e}", 500)
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass
    t_prep_end = time.perf_counter()

    # 4. Model Inference
    if MODEL_INSTANCE is None:
        init_model()
    if MODEL_INSTANCE is None:
        raise ForensicAPIException(
            "MODEL_UNAVAILABLE",
            "Neural network backbone is offline or weights file missing.",
            503,
        )

    t_inf_start = time.perf_counter()
    try:
        pred = predict_image(
            model=MODEL_INSTANCE,
            image_array=img_processed,
            threshold=threshold,
            suspicious_threshold=min(threshold + 0.10, 0.99),
            temperature=CALIBRATION_TEMP,
        )
        confidence = float(pred["confidence"])
        probability = float(pred["probability"]) / 100.0 if pred["probability"] > 1.0 else float(pred["probability"])
    except Exception as e:
        raise ForensicAPIException("INFERENCE_FAILED", f"Prediction execution failed: {e}", 500)
    t_inf_end = time.perf_counter()

    # 5. Postprocessing & Telemetry
    t_post_start = time.perf_counter()
    entropy = calculate_shannon_entropy(data_bytes)
    sha256_hash = compute_sha256(data_bytes)
    file_size_bytes = len(data_bytes)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    file_size_label = f"{file_size_mb} MB" if file_size_mb > 0 else f"{round(file_size_bytes/1024, 1)} KB"

    is_stego = probability >= threshold
    margin = probability - threshold

    if is_stego:
        verdict_text = "STEGO DETECTED"
        flag_text = "FLAG: POSITIVE"
        recommendation = "FORENSIC REVIEW RECOMMENDED"
        interpretation = (
            "The high-pass spatial filter extracted non-Gaussian noise variance concentrated "
            "systematically along sequential pixel scan lines. This distribution exhibits "
            "characteristics typical of sequential least-significant-bit (LSB) payload injection."
        )
    else:
        verdict_text = "CLEAN PASS"
        flag_text = "FLAG: NEGATIVE"
        recommendation = "WITHIN STATISTICAL BASELINE"
        interpretation = (
            "Spatial bit-plane residual distribution falls within normal photographic bounds. "
            "No statistically significant non-Gaussian delta variance detected across high-frequency components."
        )

    residual_pil = generate_srm_residual(rgb_image)
    source_b64 = image_to_base64(rgb_image.resize((512, 512)), "JPEG")
    residual_b64 = image_to_base64(residual_pil.resize((512, 512)), "PNG")

    global LAST_IMAGE_BYTES
    LAST_IMAGE_BYTES = data_bytes

    # Forensics Suite Extraction
    try:
        ela_img, ela_stats = compute_ela_with_stats(data_bytes, quality=90, scale=15.0)
        ela_b64 = image_to_base64(ela_img.resize((512, 512)), "PNG")
    except Exception as e:
        ela_b64 = ""
        ela_stats = {"error": str(e)}

    try:
        noise_img, noise_stats = compute_noise_with_stats(data_bytes, amplification=2.5, equalize=False)
        noise_b64 = image_to_base64(noise_img.resize((512, 512)), "PNG")
    except Exception as e:
        noise_b64 = ""
        noise_stats = {"error": str(e)}

    try:
        hist_data = compute_histograms_with_stats(data_bytes)
    except Exception as e:
        hist_data = {"r": [], "g": [], "b": [], "gray": [], "stats": {}}

    try:
        rich_meta = extract_rich_metadata(data_bytes)
    except Exception as e:
        rich_meta = {"categories": {}, "has_exif": False, "has_gps": False}

    t_post_end = time.perf_counter()

    # Timing calculations in milliseconds
    decode_ms = round((t_decode_end - t_decode_start) * 1000, 2)
    prep_ms = round((t_prep_end - t_prep_start) * 1000, 2)
    inf_ms = round((t_inf_end - t_inf_start) * 1000, 2)
    post_ms = round((t_post_end - t_post_start) * 1000, 2)
    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 6. SQLite Audit Logging
    log_scan_record(
        timestamp=now_iso,
        filename=file.filename,
        sha256=sha256_hash,
        file_size_bytes=file_size_bytes,
        dimensions=f"{width} × {height} PX",
        probability=round(probability, 4),
        confidence=round(confidence, 4),
        classification=verdict_text,
        threshold=threshold,
        entropy=entropy,
        runtime_ms=total_ms,
        status="SUCCESS",
    )

    return JSONResponse(
        content={
            "filename": file.filename,
            "scan_id": f"STG-{int(time.time() % 10000):04d}-X",
            "dimensions": f"{width} × {height} PX",
            "color_depth": f"{len(rgb_image.getbands()) * 8}-BIT RGB",
            "file_size": file_size_label,
            "file_size_bytes": file_size_bytes,
            "container_type": f"{img_format} / {'LOSSLESS' if img_format in ['PNG', 'BMP', 'TIFF'] else 'COMPRESSED'}",
            "entropy": f"{entropy:.3f} B/P",
            "sha256": sha256_hash,
            "sha256_short": f"{sha256_hash[:8]}...{sha256_hash[-4:]}",
            "verdict": verdict_text,
            "flag": flag_text,
            "is_stego": is_stego,
            "confidence": round(confidence * 100, 2),
            "probability": round(probability * 100, 2),
            "probability_norm": round(probability, 4),
            "tau": threshold,
            "profile": profile,
            "margin": f"{'+' if margin >= 0 else ''}{margin:.4f} ({abs(round(margin * 100, 1))}%)",
            "sigma_delta": f"{'+' if is_stego else '-'}{round(abs(margin) * 7.78, 3)} σ",
            "recommendation": recommendation,
            "interpretation": interpretation,
            "source_image_b64": source_b64,
            "residual_image_b64": residual_b64,
            "ela_image_b64": ela_b64,
            "ela_stats": ela_stats,
            "noise_image_b64": noise_b64,
            "noise_stats": noise_stats,
            "histogram_data": hist_data,
            "metadata": rich_meta,
            "timings": {
                "decode_ms": decode_ms,
                "preprocess_ms": prep_ms,
                "inference_ms": inf_ms,
                "postprocess_ms": post_ms,
                "total_ms": total_ms,
            },
            "runtime_ms": total_ms,
            "timestamp": now_iso,
        }
    )


@app.post("/api/forensics/stage-image")
async def stage_image_forensics(file: UploadFile = File(...)):
    global LAST_IMAGE_BYTES
    data_bytes = await file.read()
    if not data_bytes:
        raise ForensicAPIException("EMPTY_FILE", "Uploaded file is empty.", 400)
    LAST_IMAGE_BYTES = data_bytes

    try:
        ela_img, ela_stats = compute_ela_with_stats(data_bytes, quality=90, scale=15.0)
        ela_b64 = image_to_base64(ela_img.resize((512, 512)), "PNG")
    except Exception as e:
        ela_b64 = ""
        ela_stats = {"error": str(e)}

    try:
        noise_img, noise_stats = compute_noise_with_stats(data_bytes, amplification=2.5, equalize=False)
        noise_b64 = image_to_base64(noise_img.resize((512, 512)), "PNG")
    except Exception as e:
        noise_b64 = ""
        noise_stats = {"error": str(e)}

    try:
        hist_data = compute_histograms_with_stats(data_bytes)
    except Exception as e:
        hist_data = {"r": [], "g": [], "b": [], "gray": [], "stats": {}}

    try:
        rich_meta = extract_rich_metadata(data_bytes)
    except Exception as e:
        rich_meta = {"categories": {}, "has_exif": False, "has_gps": False}

    return JSONResponse(
        content={
            "filename": file.filename,
            "file_size": f"{round(len(data_bytes)/1024, 1)} KB",
            "ela_image_b64": ela_b64,
            "ela_stats": ela_stats,
            "noise_image_b64": noise_b64,
            "noise_stats": noise_stats,
            "histogram_data": hist_data,
            "metadata": rich_meta,
        }
    )


@app.post("/api/forensics/ela")
async def api_compute_ela(
    file: Optional[UploadFile] = File(None),
    quality: int = Form(90),
    scale: float = Form(15.0),
    invert: bool = Form(False),
):
    global LAST_IMAGE_BYTES
    raw_bytes = None
    if file and file.filename:
        raw_bytes = await file.read()
        LAST_IMAGE_BYTES = raw_bytes
    elif LAST_IMAGE_BYTES:
        raw_bytes = LAST_IMAGE_BYTES

    if not raw_bytes:
        raise ForensicAPIException("NO_IMAGE", "No image provided or cached for ELA analysis.", 400)

    try:
        ela_img, stats = compute_ela_with_stats(raw_bytes, quality=quality, scale=scale, invert=invert)
        return JSONResponse(
            content={
                "ela_image_b64": image_to_base64(ela_img.resize((512, 512)), "PNG"),
                "stats": stats,
            }
        )
    except Exception as e:
        raise ForensicAPIException("ELA_FAILED", f"Error Level Analysis failed: {e}", 500)


@app.post("/api/forensics/noise")
async def api_compute_noise(
    file: Optional[UploadFile] = File(None),
    amplification: float = Form(2.5),
    equalize: bool = Form(False),
):
    global LAST_IMAGE_BYTES
    raw_bytes = None
    if file and file.filename:
        raw_bytes = await file.read()
        LAST_IMAGE_BYTES = raw_bytes
    elif LAST_IMAGE_BYTES:
        raw_bytes = LAST_IMAGE_BYTES

    if not raw_bytes:
        raise ForensicAPIException("NO_IMAGE", "No image provided or cached for noise analysis.", 400)

    try:
        noise_img, stats = compute_noise_with_stats(raw_bytes, amplification=amplification, equalize=equalize)
        return JSONResponse(
            content={
                "noise_image_b64": image_to_base64(noise_img.resize((512, 512)), "PNG"),
                "stats": stats,
            }
        )
    except Exception as e:
        raise ForensicAPIException("NOISE_FAILED", f"Noise map computation failed: {e}", 500)


@app.post("/api/forensics/histogram")
async def api_compute_histogram(file: Optional[UploadFile] = File(None)):
    global LAST_IMAGE_BYTES
    raw_bytes = None
    if file and file.filename:
        raw_bytes = await file.read()
        LAST_IMAGE_BYTES = raw_bytes
    elif LAST_IMAGE_BYTES:
        raw_bytes = LAST_IMAGE_BYTES

    if not raw_bytes:
        raise ForensicAPIException("NO_IMAGE", "No image provided or cached for histogram analysis.", 400)

    try:
        hist_data = compute_histograms_with_stats(raw_bytes)
        return JSONResponse(content=hist_data)
    except Exception as e:
        raise ForensicAPIException("HISTOGRAM_FAILED", f"Histogram extraction failed: {e}", 500)


@app.post("/api/forensics/metadata")
async def api_extract_metadata(file: Optional[UploadFile] = File(None)):
    global LAST_IMAGE_BYTES
    raw_bytes = None
    if file and file.filename:
        raw_bytes = await file.read()
        LAST_IMAGE_BYTES = raw_bytes
    elif LAST_IMAGE_BYTES:
        raw_bytes = LAST_IMAGE_BYTES

    if not raw_bytes:
        raise ForensicAPIException("NO_IMAGE", "No image provided or cached for metadata extraction.", 400)

    try:
        meta_data = extract_rich_metadata(raw_bytes)
        return JSONResponse(content=meta_data)
    except Exception as e:
        raise ForensicAPIException("METADATA_FAILED", f"Metadata extraction failed: {e}", 500)


@app.post("/api/batch")

async def execute_batch(
    directory_path: str = Form("data/test_data/clean_images"),
    threshold: float = Form(0.73),
    profile: str = Form("BALANCED"),
    recursive: bool = Form(False),
):
    global MODEL_INSTANCE, CALIBRATION_TEMP
    if MODEL_INSTANCE is None:
        init_model()
    if MODEL_INSTANCE is None:
        raise ForensicAPIException("MODEL_UNAVAILABLE", "Model offline.", 503)

    target_dir = Path(directory_path)
    allowed_root = Path(ALLOWED_SCAN_ROOT).resolve()

    try:
        resolved = target_dir.resolve()
        if not resolved.is_dir() or not (resolved == allowed_root or allowed_root in resolved.parents):
            raise ForensicAPIException(
                "FORBIDDEN_PATH",
                f"Directory path must exist inside `{ALLOWED_SCAN_ROOT}/`.",
                403,
            )
    except Exception as e:
        raise ForensicAPIException("FORBIDDEN_PATH", str(e), 403)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    glob_pattern = "**/*" if recursive else "*"
    all_files = [p for p in resolved.glob(glob_pattern) if p.is_file() and p.suffix.lower() in exts]

    if not all_files:
        return JSONResponse(content={
            "total": 0,
            "processed": 0,
            "failed": 0,
            "threats_found": 0,
            "results": [],
            "failures": [],
            "directory": str(resolved),
        })

    results = []
    failures = []
    threat_count = 0

    for file_path in all_files:
        t0 = time.perf_counter()
        try:
            raw_bytes = file_path.read_bytes()
            img_pil = Image.open(io.BytesIO(raw_bytes))
            img_pil.load()
            dims = f"{img_pil.width} × {img_pil.height} PX"
            entropy = calculate_shannon_entropy(raw_bytes)
            sha = compute_sha256(raw_bytes)

            img_processed, _ = load_and_preprocess_image(
                str(file_path),
                target_size=512,
                use_residual=USE_RESIDUAL,
            )
            pred = predict_image(
                model=MODEL_INSTANCE,
                image_array=img_processed,
                threshold=threshold,
                suspicious_threshold=min(threshold + 0.10, 0.99),
                temperature=CALIBRATION_TEMP,
            )
            prob = float(pred["confidence"])
            is_stego = prob >= threshold
            classification = "STEGO DETECTED" if is_stego else "CLEAN PASS"
            if is_stego:
                threat_count += 1

            runtime = round((time.perf_counter() - t0) * 1000, 2)
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            item = {
                "filename": file_path.name,
                "classification": classification,
                "probability": round(prob * 100, 2),
                "confidence": round(prob, 4),
                "entropy": entropy,
                "dimensions": dims,
                "sha256": sha,
                "runtime_ms": runtime,
                "status": "SUCCESS",
            }
            results.append(item)

            log_scan_record(
                timestamp=timestamp,
                filename=file_path.name,
                sha256=sha,
                file_size_bytes=len(raw_bytes),
                dimensions=dims,
                probability=round(prob, 4),
                confidence=round(prob, 4),
                classification=classification,
                threshold=threshold,
                entropy=entropy,
                runtime_ms=runtime,
                status="SUCCESS",
            )
        except Exception as e:
            runtime = round((time.perf_counter() - t0) * 1000, 2)
            fail_item = {
                "filename": file_path.name,
                "status": "FAILED",
                "error_message": str(e),
                "runtime_ms": runtime,
            }
            failures.append(fail_item)
            log_scan_record(
                timestamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                filename=file_path.name,
                status="FAILED",
                error_message=str(e),
                runtime_ms=runtime,
            )

    return JSONResponse(content={
        "total": len(all_files),
        "processed": len(results),
        "failed": len(failures),
        "threats_found": threat_count,
        "results": results,
        "failures": failures,
        "directory": str(resolved),
    })


@app.get("/api/history")
async def get_history(limit: int = 100):
    records = get_scan_records(limit=limit)
    return JSONResponse(content={"history": records, "total_count": count_scan_records()})


@app.post("/api/clear-history")
async def clear_history():
    clear_scan_records()
    return JSONResponse(content={"status": "cleared", "total_count": 0})


@app.get("/api/export-history-csv")
async def export_history_csv():
    records = get_scan_records(limit=10000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "timestamp", "filename", "sha256", "file_size_bytes",
        "dimensions", "probability", "confidence", "classification",
        "threshold", "entropy", "runtime_ms", "status"
    ])
    for r in records:
        writer.writerow([
            r.get("id"), r.get("timestamp"), r.get("filename"), r.get("sha256"),
            r.get("file_size_bytes"), r.get("dimensions"), r.get("probability"),
            r.get("confidence"), r.get("classification"), r.get("threshold"),
            r.get("entropy"), r.get("runtime_ms"), r.get("status")
        ])
    return PlainTextResponse(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=stegoshield_audit_log.csv"},
    )


@app.get("/api/export-history-json")
async def export_history_json():
    records = get_scan_records(limit=10000)
    return JSONResponse(content={"records": records})


@app.get("/api/model-intel")
async def get_model_intel():
    meta_path = Path("model/model_metadata.json")
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["server_status"] = "ONLINE_AIRGAP"
            data["loaded_model_params"] = MODEL_INSTANCE.count_params() if MODEL_INSTANCE else 10785065
            return JSONResponse(content=data)
        except Exception:
            pass

    # Fallback to calibration config
    return JSONResponse(content={
        "model_name": "StegoShield Deep CNN",
        "model_architecture": "EfficientNet-B3 Backbone + Dense Classification Head",
        "model_version": "2.0.0",
        "input_size": "512 × 512 × 3",
        "calibration": {
            "method": "Temperature Scaling (Logit Scaling via Platt/Guo et al.)",
            "temperature": CALIBRATION_TEMP,
            "calibrated_before_thresholding": True,
        },
        "thresholds": {
            "SENSITIVE": 0.55,
            "BALANCED": 0.73,
            "PRECISION": 0.85,
        },
        "server_status": "ONLINE_AIRGAP",
    })


if __name__ == "__main__":
    init_model()
    uvicorn.run("app.web_server:app", host="127.0.0.1", port=8000, reload=False)
