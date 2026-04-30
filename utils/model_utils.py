"""Utility functions for model loading and prediction"""
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

def normalize_rgb(img):
    """Normalize RGB image to [0, 1]"""
    return img.astype(np.float32) / 255.0

def load_and_preprocess_image(image_path, img_size=64):
    """Load and preprocess image for prediction"""
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_normalized = normalize_rgb(img)

        return img_normalized, img  # Return both processed and original
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def load_model(model_path="model/cnn_model.keras"):
    """Load the trained CNN model"""
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = tf.keras.models.load_model(str(model_path))
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict_image(model, image_array, threshold=0.5):
    """Predict if image contains steganography"""
    try:
        # Add batch dimension if needed
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Get prediction
        prediction = model.predict(image_array, verbose=0)
        confidence = float(prediction[0][0])
        
        # Determine risk level
        if confidence < threshold:
            classification = "CLEAN"
            risk_level = "LOW"
        elif confidence < 0.7:
            classification = "SUSPICIOUS"
            risk_level = "MEDIUM"
        else:
            classification = "DETECTED"
            risk_level = "HIGH"
        
        return {
            "confidence": confidence,
            "classification": classification,
            "risk_level": risk_level,
            "probability": confidence * 100
        }
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

def get_risk_explanation(confidence, classification):
    """Get detailed explanation for detection result"""
    explanations = {
        "CLEAN": "Standard pixel distribution detected. No steganographic payload identified.",
        "SUSPICIOUS": "Minor frequency anomalies detected in color channels. Manual review recommended.",
        "DETECTED": "High-probability steganographic payload detected. Severe pixel noise anomaly identified."
    }
    return explanations.get(classification, "Unknown classification")
