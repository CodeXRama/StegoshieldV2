import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from config import IMG_SIZE
except ImportError:
    IMG_SIZE = 256
def normalize_rgb(img):
    """Normalize RGB image to [0, 1]."""
    return img.astype(np.float32) / 255.0

def apply_temperature(probability, temperature):
    """Apply temperature scaling to a sigmoid probability.
    Temperature > 1 softens predictions (pushes toward 0.5).
    Temperature < 1 sharpens predictions (pushes toward 0 or 1).
    Temperature == 1 or None/0 returns the original probability.
    """
    if temperature is None or temperature <= 0:
        return float(probability)
    if abs(temperature - 1.0) < 1e-6:
        return float(probability)

    prob = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(prob / (1.0 - prob))
    scaled = 1.0 / (1.0 + np.exp(-logit / temperature))
    return float(scaled)


def format_confidence_for_display(probability, demo_mode=False):
    """Return UI-facing confidence percent.

    Always return the real model probability as a percentage.  ``demo_mode``
    is retained only for backward compatibility and is deliberately ignored:
    Version 1 never fabricates confidence values for a demonstration.
    """
    prob = float(probability)
    return prob * 100.0

get_display_confidence = format_confidence_for_display


def multi_residual_np(img):
    """Compute a 3-channel residual feature map (Gray / HPF / Sobel)."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV is required for residual preprocessing") from e

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel = np.array(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32
    )
    hpf = cv2.filter2D(gray, -1, kernel)
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)

    return np.stack([gray, hpf, sobel], axis=-1)


def load_and_preprocess_image(image_path, target_size=None, use_residual=False):
    """Load and preprocess image for prediction.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.
    target_size : int, optional
        Resize dimension (square).  Falls back to ``IMG_SIZE`` from config.
    use_residual : bool
        If True, applies multi-residual preprocessing.

    Returns
    -------
    img_normalized : np.ndarray
        Preprocessed image array, shape (H, W, 3), dtype float32, range [0, 1].
    img_original : np.ndarray
        Original RGB image as uint8 array.
    """
    try:
        size = target_size or IMG_SIZE
        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size))
        img = np.array(img)

        img_original = img.copy()
        if use_residual:
            img = multi_residual_np(img)
        img_normalized = img.astype(np.float32) / 255.0

        return img_normalized, img_original
    except Exception as e:
        raise RuntimeError(f"Error preprocessing image: {e}") from e


def load_model(model_path="model/model.keras"):
    """Load a trained Keras model."""
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        class FixedDropout(tf.keras.layers.Dropout):
            def call(self, inputs, training=None):
                return super().call(inputs, training=training)

        custom_objects = {
            "FixedDropout": FixedDropout,
            "swish": tf.nn.swish,
            "function": tf.nn.swish,
        }
        try:
            import efficientnet.tfkeras as efn

            custom_objects.update(efn.get_custom_objects())
            custom_objects.update(
                {
                    "EfficientNetB3": efn.EfficientNetB3,
                    "FixedDropout": efn.FixedDropout,
                }
            )
        except Exception:
            pass

        try:
            model = tf.keras.models.load_model(
                str(model_path),
                custom_objects=custom_objects,
                compile=False,
            )
        except TypeError:
            model = tf.keras.models.load_model(
                str(model_path),
                custom_objects=custom_objects,
                compile=False,
            )
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}") from e


def predict_image(
    model,
    image_array,
    threshold=0.5,
    suspicious_threshold=0.7,
    temperature=1.0,
):
    """Predict whether an image contains steganography.

    Returns a dict with keys: confidence, classification, risk_level,
    probability (percentage).
    """
    try:
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array, verbose=0)
        confidence = float(prediction[0][0])
        confidence = apply_temperature(confidence, temperature)

        if confidence < threshold:
            classification = "CLEAN"
            risk_level = "LOW"
        elif confidence < suspicious_threshold:
            classification = "SUSPICIOUS"
            risk_level = "MEDIUM"
        else:
            classification = "DETECTED"
            risk_level = "HIGH"

        return {
            "confidence": confidence,
            "classification": classification,
            "risk_level": risk_level,
            "probability": confidence * 100,
        }
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}") from e


def get_risk_explanation(confidence, classification):
    """Get a human-readable explanation for a detection result."""
    explanations = {
        "CLEAN": "Standard pixel distribution detected. No steganographic payload identified.",
        "SUSPICIOUS": "Minor frequency anomalies detected in color channels. Manual review recommended.",
        "DETECTED": "High-probability steganographic payload detected. Severe pixel noise anomaly identified.",
    }
    return explanations.get(classification, "Unknown classification")
