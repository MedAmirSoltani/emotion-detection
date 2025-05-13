import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download
import subprocess
import sys

# 🛠️ PAGE CONFIG
st.set_page_config(page_title="EmotionLens™", page_icon="🧠", layout="wide")

# 🎨 CUSTOM CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(to right, #e0f2fe, #f8fafc);
    }

    .title {
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        color: #111827;
        margin-bottom: 0.2em;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2em;
        margin-bottom: 2.5em;
    }

    .glass-box {
        background: rgba(255, 255, 255, 0.65);
        border-radius: 20px;
        padding: 2em;
        backdrop-filter: blur(15px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.08);
    }

    .stButton>button {
        background-color: #3b82f6;
        color: white;
        padding: 0.7em 1.5em;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1em;
        transition: 0.2s;
    }

    .stButton>button:hover {
        background-color: #2563eb;
    }

    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #999;
        margin-top: 4em;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 TITLE
st.markdown('<div class="title">EmotionLens™</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a face image and discover what your expression says</div>', unsafe_allow_html=True)

# 🧬 LOAD MODEL
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Amirsoltani21/emotion-detection",
        filename="model.h5",
        repo_type="model"
    )
    return tf.keras.models.load_model(model_path)

model = load_model()

# 🔤 LABELS
emotion_label_to_text = {
    0: 'Anger',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Surprise',
    6: 'Neutral'
}

# 🧹 PREPROCESS
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((48, 48))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# 📤 UPLOAD & DISPLAY
st.markdown("### 📤 Upload Your Image")
col1, col2 = st.columns([1.2, 2], gap="large")

with col1:
    uploaded_file = st.file_uploader("Choose a facial image (jpg, png, jpeg)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼 Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file:
        st.markdown('<div class="glass-box">', unsafe_allow_html=True)
        processed = preprocess_image(img)
        prediction = model.predict(processed)[0]
        top_emotion = emotion_label_to_text[np.argmax(prediction)]

        st.success(f"🎯 **Detected Emotion:** {top_emotion}")
        st.subheader("📊 Detailed Probabilities")
        for i, prob in enumerate(prediction):
            st.markdown(f"**{emotion_label_to_text[i]}**")
            st.progress(float(prob))
        st.markdown('</div>', unsafe_allow_html=True)

# 🎥 WEBCAM LAUNCH
st.markdown("### 🎥 Try Real-Time Detection")
if st.button("Launch Webcam"):
    subprocess.Popen([sys.executable, "webcam_emotion.py"])
    st.info("Your webcam has started in a new window. Press Q to quit.")

# 👣 FOOTER
st.markdown('<div class="footer">Made with 💙 by Amir Soltani – EmotionLens™ 2025</div>', unsafe_allow_html=True)
