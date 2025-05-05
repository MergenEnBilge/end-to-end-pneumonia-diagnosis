import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

model = load_model("model/pneumonia_diagnosis_model.h5")

img_size = 256
labels = ['NORMAL', 'PNEUMONIA']

def preprocess_image(image_bytes):
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

st.title("Pneumonia Diagnosis from Chest X-ray")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)
    
    if st.button("Diagnose"):
        with st.spinner("Diagnosing..."):
            input_tensor = preprocess_image(uploaded_file.read())
            prediction = model.predict(input_tensor)
            
            diagnosis = labels[int(prediction[0][0] > 0.5)]
            confidence = float(prediction[0][0])
            confidence = confidence if diagnosis == 'PNEUMONIA' else 1 - confidence
            st.success(f"Diagnosis: **{diagnosis}** with confidence **{confidence:.2f}**")
