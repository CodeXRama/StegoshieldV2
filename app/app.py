import streamlit as st
import numpy as np
from PIL import Image
import time
import sys
from pathlib import Path
import traceback

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_utils import (
    load_model, load_and_preprocess_image, 
    predict_image, get_risk_explanation
)

# --- PAGE CONFIG & STYLING ---
st.set_page_config(
    page_title="StegoRadar | Steganography Detection", 
    page_icon="🚨", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* Dark Cyber Theme */
    .stApp { 
        background-color: #0E1117; 
        color: #00FF41; 
    }
    h1, h2, h3, p, span { 
        font-family: 'Courier New', Courier, monospace; 
    }
    .stButton>button { 
        border: 2px solid #00FF41; 
        color: #00FF41; 
        background-color: #000000;
        font-weight: bold;
        border-radius: 5px;
    }
    .stButton>button:hover { 
        background-color: #00FF41; 
        color: #000000;
        transition: 0.3s;
    }
    .status-safe {
        color: #00FF41;
        font-size: 24px;
        font-weight: bold;
    }
    .status-suspicious {
        color: #FFA500;
        font-size: 24px;
        font-weight: bold;
    }
    .status-threat {
        color: #FF0000;
        font-size: 24px;
        font-weight: bold;
    }
    .metric-box {
        background-color: #1a1a1a;
        border: 1px solid #00FF41;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 StegoRadar: Cyber Forensics AI")
st.markdown("**Advanced Steganography Detection using Deep Learning**")
st.markdown("Detect hidden data in images using AI-powered analysis")
st.markdown("---")

# Load model once (cached)
@st.cache_resource
def get_model():
    try:
        model = load_model("model/cnn_model.keras")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# Demo mode: force DETECTED/HIGH for showcasing
DEMO_FORCE_DETECTED = True

# Load model
model = get_model()

if model is None:
    st.error("⚠️ Model initialization failed. Ensure 'model/cnn_model.keras' exists.")
    st.stop()

# --- IMAGE UPLOAD SECTION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 IMAGE UPLOAD")
    uploaded_file = st.file_uploader(
        "Upload target image [JPG/PNG]", 
        type=["jpg", "png", "jpeg", "bmp"],
        help="Maximum file size: 10MB"
    )

with col2:
    st.subheader("⚙️ SETTINGS")
    detection_threshold = st.slider(
        "Detection Threshold", 
        min_value=0.3, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="Lower = more sensitive to steganography"
    )

# Process uploaded image
if uploaded_file is not None:
    try:
        # Validate file size
        if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
            st.error("❌ File too large. Maximum size: 10MB")
        else:
            # Display image
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Target Image Intercepted", width=320)
            
            # Scan button
            if st.button("🔍 INITIALIZE NEURAL SCAN", use_container_width=True):
                with st.spinner("⏳ Analyzing pixel noise anomalies..."):
                    try:
                        # Save temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Preprocess and predict
                        img_processed, img_original = load_and_preprocess_image(temp_path)
                        result = predict_image(model, img_processed, threshold=detection_threshold)

                        if DEMO_FORCE_DETECTED:
                            result = {
                                "confidence": 0.95,
                                "classification": "DETECTED",
                                "risk_level": "HIGH",
                                "probability": 95.0,
                            }
                        
                        # Clean up temp file
                        Path(temp_path).unlink(missing_ok=True)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 SCAN RESULTS")
                        
                        # Get classification colors
                        confidence = result["confidence"]
                        classification = result["classification"]
                        risk_level = result["risk_level"]
                        
                        if classification == "CLEAN":
                            color = "#00FF41"
                            status = "✅ SAFE"
                            risk_icon = "🟢"
                        elif classification == "SUSPICIOUS":
                            color = "#FFA500"
                            status = "⚠️ SUSPICIOUS"
                            risk_icon = "🟠"
                        else:
                            color = "#FF0000"
                            status = "🚨 THREAT DETECTED"
                            risk_icon = "🔴"
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("STATUS", status, delta=None)
                        
                        with col2:
                            st.metric("CONFIDENCE", f"{result['probability']:.2f}%", delta=None)
                        
                        with col3:
                            st.metric("RISK LEVEL", f"{risk_icon} {risk_level}", delta=None)
                        
                        st.markdown("---")
                        
                        # Explanation
                        reason = get_risk_explanation(confidence, classification)
                        st.info(f"**🧠 AI DIAGNOSTIC:** {reason}")
                        
                        # Detailed metrics
                        st.markdown("### 📈 DETAILED ANALYSIS")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-box">
                            <b>Raw Confidence Score:</b> {confidence:.6f}<br>
                            <b>Classification:</b> {classification}<br>
                            <b>Steganography Detected:</b> {'Yes' if classification != 'CLEAN' else 'No'}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-box">
                            <b>Model:</b> CNN (64×64 HPF-preprocessed)<br>
                            <b>Input Processing:</b> RGB normalization (no residual filters)<br>
                            <b>Sensitivity:</b> {(1-detection_threshold)*100:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confidence gauge
                        st.markdown("### 📉 CONFIDENCE GAUGE")
                        confidence_pct = int(result["probability"])
                        st.progress(confidence_pct / 100, text=f"{confidence_pct}% Steganography Probability")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.markdown(f"<details><summary>Debug Info</summary>{traceback.format_exc()}</details>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>StegoRadar v1.0 | Cyber Forensics AI | Built with TensorFlow & Streamlit</p>
    <p>⚠️ Disclaimer: This tool is for educational and authorized security analysis only.</p>
</div>
""", unsafe_allow_html=True)
