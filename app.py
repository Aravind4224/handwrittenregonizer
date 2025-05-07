

import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import os

# ----------------------------
# Load or train CNN model
# ----------------------------
@st.cache_resource
def load_or_train_model():
    model_path = "mnist_cnn_model_v2.h5"
    
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)
    model.save(model_path)
    return model

model = load_or_train_model()

# ----------------------------
# Prediction Function
# ----------------------------
def predict_digit(image):
    # Convert to grayscale and invert
    image = image.convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)[0]
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return digit, confidence

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Handwritten Digit Recognizer")

uploaded_file = st.file_uploader("Upload a digit image (28x28 or larger)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    digit, confidence = predict_digit(image)
    st.success(f"Predicted Digit: {digit} ({confidence:.2f}% confidence)")