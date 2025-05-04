# train_model.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path to the dataset (update the paths accordingly)
train_folder = '/Users/manikantapadira/Desktop/MS/ Facial Emotion Recognition (FER) using CNN/archive/train'
test_folder = '/Users/manikantapadira/Desktop/MS/ Facial Emotion Recognition (FER) using CNN/archive/test'

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load images from folders and preprocess them
def load_images_from_folder(folder_path, label, img_size=(48, 48)):
    images = []
    labels = []
    for emotion in os.listdir(folder_path):
        emotion_path = os.path.join(folder_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            img = img / 255.0
            images.append(img)
            labels.append(label)
        label += 1
    return np.array(images), np.array(labels)

train_images, train_labels = load_images_from_folder(train_folder, 0)
test_images, test_labels = load_images_from_folder(test_folder, 0)

# Reshaping the images
train_images = train_images.reshape(-1, 48, 48, 1)
test_images = test_images.reshape(-1, 48, 48, 1)

# One-hot encoding the labels
train_labels = to_categorical(train_labels, num_classes=7)
test_labels = to_categorical(test_labels, num_classes=7)

# Splitting training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Build the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # 7 emotions
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('model.h5')

print("Model training complete. Model saved as 'model.h5'.")
