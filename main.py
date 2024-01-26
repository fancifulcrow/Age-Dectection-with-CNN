import cv2
import tensorflow as tf
import numpy as np
import streamlit as st


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

age_classes = ['0 - 5', '6 - 18', '19 - 30', '31 - 45', '46 - 64', '65+']
age_model = tf.keras.models.load_model('age_model')


def detect_face_and_age(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160))

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        
        n = cv2.resize(face, (48, 48))
        n = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
        n = np.expand_dims(n, axis=0)

        prediction = age_model.predict(n / 255.0)

        predicted_class = np.argmax(prediction)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f'Age: {age_classes[predicted_class]}', (x, y-10), font, 1, (0, 255, 0), 2)

    return image


# Streamlit App
st.title("Age Detection with CNN")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(detect_face_and_age(image=image), channels="BGR")