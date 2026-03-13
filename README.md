<div align="center">

<img src="https://img.shields.io/badge/dataset-FER2013-4F46E5?style=for-the-badge"/>
<img src="https://img.shields.io/badge/model-VGG19-4F46E5?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/emotions-7-4F46E5?style=for-the-badge"/>
<img src="https://img.shields.io/badge/deployed-HuggingFace-4F46E5?style=for-the-badge&logo=huggingface&logoColor=white"/>

# 🧠 EmotionLens™
### Facial Emotion Recognition — VGG19 Transfer Learning + Streamlit App

> *Upload a face. Discover what it says.*

</div>

---

## 📖 Overview

End-to-end facial emotion recognition pipeline: from raw pixel data to a deployed web app. Trained on the **FER2013** benchmark dataset using **VGG19 transfer learning**, the model classifies facial expressions into 7 emotion categories. The trained model is hosted on HuggingFace and served through a Streamlit interface with real-time webcam support.

---

## 🎭 Emotions Detected

`Anger` · `Disgust` · `Fear` · `Happiness` · `Sadness` · `Surprise` · `Neutral`

---

## 📊 Dataset — FER2013

| | |
|---|---|
| **Source** | Kaggle — FER2013 Challenge |
| **Format** | CSV with pixel strings (48×48 grayscale) |
| **Classes** | 7 emotions |
| **Split** | Train / Validation (stratified) |
| **Note** | Class imbalance — Happiness overrepresented vs. Disgust/Fear |

---

## 🧠 Model Pipeline

### 1 — Preprocessing
- Pixel strings → `48×48` numpy arrays
- Grayscale → RGB conversion via OpenCV (`cv2.COLOR_GRAY2RGB`)
- Normalization: pixel values scaled to `[0, 1]`
- Label encoding + one-hot encoding

### 2 — Data Augmentation
`ImageDataGenerator` with rotation (±15°), width/height shift (15%), shear, zoom, and horizontal flip — to compensate for limited data per class.

### 3 — Architecture — VGG19 Transfer Learning
- **Base:** `VGG19` pretrained on ImageNet (`include_top=False`, input `48×48×3`)
- **Head:** `GlobalAveragePooling2D` → `Dense(7, activation='softmax')`
- **Optimizer:** Adam (`lr=0.0001`, `β1=0.9`, `β2=0.999`)
- **Loss:** Categorical Crossentropy
- **Batch size:** 32 · **Epochs:** 25

### 4 — Training Callbacks
- `EarlyStopping` — monitors `val_accuracy`, patience=11, restores best weights
- `ReduceLROnPlateau` — adapts learning rate on plateau

### 5 — Evaluation
- Confusion matrix + classification report
- Strong performance on **Happiness**, lower on **Disgust** and **Fear** (class imbalance + ambiguous images)
- Model saved as `model.h5` and `model.yaml`

---

## 🚀 Streamlit App — EmotionLens™

The deployed app loads the model directly from HuggingFace Hub:

```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="Amirsoltani21/emotion-detection", filename="model.h5")
```

**Features:**
- 📤 Upload a face image (JPG/PNG) → instant emotion prediction with probability bars
- 🎥 Real-time webcam detection via `webcam_emotion.py`
- Glass-morphism UI with blue/white gradient theme

---

## 🛠️ Stack

| Layer | Tool |
|---|---|
| Data | FER2013 (Kaggle) |
| Preprocessing | NumPy, Pandas, OpenCV |
| Model | TensorFlow / Keras — VGG19 |
| Evaluation | Scikit-learn, Scikitplot |
| Model hosting | HuggingFace Hub |
| App | Streamlit |
| Visualization | Matplotlib, Seaborn |

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/emotion-lens.git
cd emotion-lens

pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app pulls `model.h5` automatically from HuggingFace — no manual download needed.

---

## 📁 Structure

```
├── facial-emotion-recognition.ipynb   # Training pipeline
├── app.py                             # Streamlit app (EmotionLens™)
├── webcam_emotion.py                  # Real-time webcam detection
├── requirements.txt
└── README.md
```

---

## 👤 Author

**Amir Soltani** — Data Science & NLP  
Master's student · Alternance @ Benman Partners  
ESPRIT · PST&B · UTT Mastère Spécialisé 2026

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Amirsoltani21-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/Amirsoltani21)

---

<div align="center">
<sub>EmotionLens™ 2025 · Built with 💙</sub>
</div>
