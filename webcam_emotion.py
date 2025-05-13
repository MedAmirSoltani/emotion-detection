import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# ✅ Load model from Hugging Face
model_path = hf_hub_download(
    repo_id="Amirsoltani21/emotion-detection",
    filename="model.h5",
    repo_type="model"
)
model = tf.keras.models.load_model(model_path)

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

# OpenCV face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_image = Image.fromarray(face).convert("RGB").resize((48, 48))
        face_array = np.array(face_image) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        prediction = model.predict(face_array)[0]
        predicted_label = emotion_label_to_text[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Real-Time Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
