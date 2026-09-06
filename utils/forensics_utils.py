"""Forensic analysis utilities for image inspection."""

import io
from typing import Dict

import cv2
import exifread
import numpy as np
from PIL import Image, ImageChops
from skimage import exposure


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Open *image_bytes* as an RGB PIL Image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def compute_ela(
    image_bytes: bytes, quality: int = 90, scale: float = 10.0
) -> Image.Image:
    """Error Level Analysis — highlights compression inconsistencies."""
    image = load_image_from_bytes(image_bytes)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    recompressed = Image.open(buffer)
    diff = ImageChops.difference(image, recompressed)
    diff_np = np.array(diff).astype(np.float32) * float(scale)
    diff_np = np.clip(diff_np, 0, 255).astype(np.uint8)
    return Image.fromarray(diff_np)


def compute_noise_map(
    image_bytes: bytes, amplification: float = 2.0, equalize: bool = False
) -> Image.Image:
    """Compute a Laplacian-based noise map of the image."""
    image = load_image_from_bytes(image_bytes)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    noise = np.abs(lap) * float(amplification)
    noise = noise - noise.min()
    if noise.max() > 0:
        noise = noise / noise.max()
    if equalize:
        noise = exposure.equalize_hist(noise)
    noise = np.clip(noise * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(noise, mode="L")


def compute_histograms(image_bytes: bytes) -> Dict[str, np.ndarray]:
    """Return per-channel histograms (R, G, B, gray) with 256 bins."""
    image = load_image_from_bytes(image_bytes)
    arr = np.array(image)
    hist_r, _ = np.histogram(arr[:, :, 0], bins=256, range=(0, 255))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=256, range=(0, 255))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=256, range=(0, 255))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hist_gray, _ = np.histogram(gray, bins=256, range=(0, 255))
    return {
        "r": hist_r,
        "g": hist_g,
        "b": hist_b,
        "gray": hist_gray,
    }


def extract_metadata(image_bytes: bytes) -> Dict[str, str]:
    """Extract EXIF and basic metadata from *image_bytes*."""
    image = Image.open(io.BytesIO(image_bytes))
    metadata: Dict[str, str] = {
        "Format": image.format or "Unknown",
        "Mode": image.mode,
        "Dimensions": f"{image.size[0]} x {image.size[1]}",
    }

    tags: dict = {}
    try:
        tags = exifread.process_file(io.BytesIO(image_bytes), details=False)
    except Exception:
        tags = {}

    for key, value in tags.items():
        if key.startswith("EXIF") or key.startswith("Image"):
            metadata[key] = str(value)

    gps_data = extract_gps(tags)
    if gps_data:
        metadata.update(gps_data)

    return metadata


def extract_gps(tags: Dict) -> Dict[str, str]:
    """Extract GPS coordinates from EXIF *tags*.

    Returns an empty dict if GPS data is missing or malformed.
    """

    def _convert_to_degrees(value):
        d = value.values[0]
        m = value.values[1]
        s = value.values[2]
        return (
            float(d.num) / float(d.den)
            + (float(m.num) / float(m.den) / 60.0)
            + (float(s.num) / float(s.den) / 3600.0)
        )
    try:
        lat = tags.get("GPS GPSLatitude")
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon = tags.get("GPS GPSLongitude")
        lon_ref = tags.get("GPS GPSLongitudeRef")
        if not (lat and lat_ref and lon and lon_ref):
            return {}

        latitude = _convert_to_degrees(lat)
        if str(lat_ref) == "S":
            latitude = -latitude

        longitude = _convert_to_degrees(lon)
        if str(lon_ref) == "W":
            longitude = -longitude

        return {
            "GPS Latitude": f"{latitude:.6f}",
            "GPS Longitude": f"{longitude:.6f}",
        }
    except Exception:
        return {}


def compute_ela_with_stats(
    image_bytes: bytes, quality: int = 90, scale: float = 10.0, invert: bool = False
):
    """Compute Error Level Analysis with statistical diagnostics."""
    image = load_image_from_bytes(image_bytes)
    buffer = io.BytesIO()
    clamped_q = max(1, min(100, int(quality)))
    image.save(buffer, format="JPEG", quality=clamped_q)
    buffer.seek(0)
    recompressed = Image.open(buffer)
    diff = ImageChops.difference(image, recompressed)
    diff_np = np.array(diff).astype(np.float32)

    raw_mae = float(np.mean(diff_np))
    raw_max = float(np.max(diff_np))
    anomaly_pct = float(np.sum(diff_np > 15.0) / diff_np.size * 100.0)

    scaled_diff = diff_np * float(scale)
    if invert:
        scaled_diff = 255.0 - scaled_diff
    scaled_diff = np.clip(scaled_diff, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(scaled_diff)

    stats = {
        "mae": round(raw_mae, 2),
        "max_diff": int(raw_max),
        "anomaly_pct": round(anomaly_pct, 2),
        "quality": clamped_q,
        "scale": float(scale),
        "invert": invert,
        "status": "SUSPICIOUS ERROR GRADIENT" if anomaly_pct > 6.0 or raw_max > 45 else "HOMOGENEOUS COMPRESSION PATTERN",
    }
    return result_img, stats


def compute_noise_with_stats(
    image_bytes: bytes, amplification: float = 2.0, equalize: bool = False
):
    """Compute high-pass Laplacian noise residual with distribution statistics."""
    image = load_image_from_bytes(image_bytes)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    noise_raw = np.abs(lap)
    noise_mean = float(np.mean(noise_raw))
    noise_variance = float(np.var(noise_raw))
    high_freq_density = float(np.sum(noise_raw > 12.0) / noise_raw.size * 100.0)

    noise = noise_raw * float(amplification)
    noise = noise - noise.min()
    if noise.max() > 0:
        noise = noise / noise.max()
    if equalize:
        noise = exposure.equalize_hist(noise)
    noise = np.clip(noise * 255.0, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(noise, mode="L")

    stats = {
        "noise_mean": round(noise_mean, 2),
        "noise_variance": round(noise_variance, 2),
        "high_freq_density": round(high_freq_density, 2),
        "amplification": round(float(amplification), 1),
        "equalize": equalize,
        "status": "HIGH VARIANCE (MICRO-PERTURBATIONS)" if noise_variance > 180.0 else "UNIFORM NOISE ENVELOPE",
    }
    return result_img, stats


def compute_histograms_with_stats(image_bytes: bytes):
    """Compute per-channel histograms, parity distribution, and intensity stats."""
    image = load_image_from_bytes(image_bytes)
    arr = np.array(image)
    hist_r, _ = np.histogram(arr[:, :, 0], bins=256, range=(0, 255))
    hist_g, _ = np.histogram(arr[:, :, 1], bins=256, range=(0, 255))
    hist_b, _ = np.histogram(arr[:, :, 2], bins=256, range=(0, 255))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hist_gray, _ = np.histogram(gray, bins=256, range=(0, 255))

    lsb = arr[:, :, 0] & 1
    even_pct = float(np.sum(lsb == 0) / lsb.size * 100.0)
    odd_pct = float(100.0 - even_pct)

    return {
        "r": hist_r.tolist(),
        "g": hist_g.tolist(),
        "b": hist_b.tolist(),
        "gray": hist_gray.tolist(),
        "stats": {
            "mean_intensity": round(float(np.mean(gray)), 1),
            "std_intensity": round(float(np.std(gray)), 1),
            "peak_intensity": int(np.argmax(hist_gray)),
            "lsb_even_pct": round(even_pct, 1),
            "lsb_odd_pct": round(odd_pct, 1),
            "parity_status": "BALANCED PARITY (NATURAL)" if abs(even_pct - 50.0) < 3.0 else "UNEVEN LSB DISTRIBUTION (POSSIBLE EMBEDDING)",
        },
    }


def extract_rich_metadata(image_bytes: bytes):
    """Extract categorized metadata with EXIF tampering and integrity checks."""
    raw_meta = extract_metadata(image_bytes)
    image = Image.open(io.BytesIO(image_bytes))
    w, h = image.size
    mp = round((w * h) / 1000000.0, 2)

    import math
    divisor = math.gcd(w, h)
    aspect = f"{w // divisor}:{h // divisor}" if divisor > 0 else f"{w}:{h}"

    categories = {
        "Container & Geometry": {
            "Format": (image.format or "Unknown").upper(),
            "Dimensions": f"{w} × {h} PX",
            "Resolution (MP)": f"{mp} MP",
            "Aspect Ratio": aspect,
            "Color Mode": image.mode,
            "Color Depth": f"{len(image.getbands()) * 8}-Bit",
            "Channels": len(image.getbands()),
            "Payload Size": f"{len(image_bytes):,} Bytes ({round(len(image_bytes)/1024, 1)} KB)",
        },
        "EXIF & Camera Hardware": {},
        "GPS Geolocation": {},
        "Forensic Integrity": {},
    }

    for k, v in raw_meta.items():
        if k.startswith("GPS"):
            categories["GPS Geolocation"][k] = v
        elif k not in ["Format", "Mode", "Dimensions"]:
            categories["EXIF & Camera Hardware"][k] = v

    has_exif = len(categories["EXIF & Camera Hardware"]) > 0
    has_gps = len(categories["GPS Geolocation"]) > 0

    software_tag = None
    for k, v in categories["EXIF & Camera Hardware"].items():
        if any(s in k.lower() for s in ["software", "processing", "tool", "editor"]):
            software_tag = str(v)
            break

    categories["Forensic Integrity"]["EXIF Header Present"] = "PRESENT" if has_exif else "STRIPPED / ABSENT"
    categories["Forensic Integrity"]["GPS Metadata Present"] = "PRESENT" if has_gps else "ABSENT"
    categories["Forensic Integrity"]["Software Signature"] = software_tag if software_tag else "Pure Camera Hardware / Clean"
    categories["Forensic Integrity"]["Header Anomaly Level"] = "ELEVATED (Header Cleared)" if not has_exif else "NOMINAL"

    google_maps_url = None
    if has_gps and "GPS Latitude" in categories["GPS Geolocation"] and "GPS Longitude" in categories["GPS Geolocation"]:
        lat = categories["GPS Geolocation"]["GPS Latitude"]
        lon = categories["GPS Geolocation"]["GPS Longitude"]
        google_maps_url = f"https://www.google.com/maps?q={lat},{lon}"

    return {
        "categories": categories,
        "has_exif": has_exif,
        "has_gps": has_gps,
        "google_maps_url": google_maps_url,
        "software": software_tag,
        "raw_count": len(raw_meta),
    }



