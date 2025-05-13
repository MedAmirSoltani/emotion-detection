import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import subprocess
import sys
import os
from huggingface_hub import HfApi

# Config
st.set_page_config(page_title="🧠 Emotion Detector", page_icon="🧠", layout="centered")
# Styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton > button {
        width: 100%;
        padding: 0.8em 1.2em;
        font-size: 1.1em;
        border-radius: 10px;
    }
    .option-btn {
        background-color: #0066cc;
        color: white;
    }
    .upload-btn {
        background-color: #28a745;
        color: white;
    }
    .title {
        font-size: 2.4em;
        font-weight: 700;
        color: #333;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 1.2em;
        color: #666;
        text-align: center;
        margin-bottom: 2em;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🧠 Real-Time Facial Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Choose an option below to predict emotions from facial images</div>', unsafe_allow_html=True)

# Hugging Face upload logic
def upload_model_to_huggingface():
    token = "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"  # ⚠️ TEMP use only!
    try:
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=".",
            repo_id="Amirsoltani21/emotion-detection",
            repo_type="model",
            allow_patterns=["model.h5"]
        )
        st.success("✅ Model uploaded to Hugging Face successfully!")
    except Exception as e:
        st.error(f"Upload failed: {e}")


# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Labels
emotion_label_to_text = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}

# Preprocessing
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((48, 48))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Options layout
st.markdown("### 🔍 Choose Your Detection Mode")

col1, col2 = st.columns(2)

# 🖼️ Upload Image
with col1:
    st.markdown("#### 🖼️ Upload a Face Image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0]
        predicted_label = emotion_label_to_text[np.argmax(prediction)]

        st.success(f"🎯 **Predicted Emotion:** {predicted_label.capitalize()}")

        st.markdown("#### 📊 Prediction Probabilities")
        for i, prob in enumerate(prediction):
            st.progress(float(prob), text=f"{emotion_label_to_text[i].capitalize()} — {prob:.2%}")

# 📸 Live Webcam
with col2:
    st.markdown("#### 📸 Launch Real-Time Detection")
    if st.button("Start Webcam Detection 🚀"):
        subprocess.Popen([sys.executable, "webcam_emotion.py"])
        st.info("Live webcam launched in a new window. Press **Q** in the OpenCV window to close.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color: #aaa;'>Built by Amir Soltani · Powered by TensorFlow & Hugging Face</p>", unsafe_allow_html=True)
