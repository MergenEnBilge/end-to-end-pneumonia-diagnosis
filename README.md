# Pneumonia Diagnosis Model

A deep learning–powered web application to detect **pneumonia** from chest X-ray images. Built using a custom-trained Keras model and deployed via a **Streamlit** frontend. This project demonstrates a practical application of computer vision in healthcare diagnostics.

---

## 🧠 Model Overview

The classification model was trained on labeled chest X-ray images from two categories:

- `NORMAL`
- `PNEUMONIA`

### 🔍 Preprocessing Steps

- Images were loaded using OpenCV in **grayscale**
- Resized to **256×256** pixels
- Pixel values normalized to the range `[0, 1]`
- Final input shape for the model: `(256, 256, 1)`

### 🏷️ Labels

- `0` → NORMAL
- `1` → PNEUMONIA

The model was trained using Convolutional Neural Networks (Architecture given in the notebook attached), optimized for binary classification with appropriate validation and test sets.

---

## 🚀 Web App Features

- Upload chest X-ray images in `.jpg`, `.jpeg`, or `.png` format
- Automatic preprocessing to match the training pipeline
- Outputs:
  - Diagnosis: **Pneumonia** or **Normal**
  - Model confidence score

Frontend is implemented using **Streamlit** for rapid prototyping and user interaction.

---

## 🛠️ Setup Instructions

### 1. Clone the repository

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```
