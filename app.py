import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')  # Ensure this file exists in the same directory
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data.get('image')

    if not img_data:
        return jsonify({'emotion': '--'})

    img_data = img_data.split(',')[1]
    img = base64.b64decode(img_data)
    nparr = np.frombuffer(img, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    emotion_label = '--'

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            emotion = model.predict(face)
            predicted_class = np.argmax(emotion)
            confidence = float(np.max(emotion))  # Add confidence score
            emotion_label = f"{emotion_labels[predicted_class]} ({confidence:.2f})"

    return jsonify({'emotion': emotion_label})

if __name__ == '__main__':
    app.run(debug=True)
