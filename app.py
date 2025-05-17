# app.py

import streamlit as st
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# URL pública direta do seu modelo .h5 no Hugging Face
MODEL_URL  = "https://huggingface.co/Gallorafael2222/Cnn/resolve/main/flower_classifier.h5"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "flower_classifier.h5")

@st.cache_resource
def load_flower_model():
    # Se não existir localmente, baixa por HTTP
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        resp = requests.get(MODEL_URL, stream=True)
        resp.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)
    return load_model(MODEL_PATH)

# Carrega o modelo (e faz download na primeira vez)
model = load_flower_model()

# Título da app
st.title("🌸 Classificador de Flores")

# Exibe qual input_shape o modelo espera
st.write("⚙️ O modelo espera input com shape:", model.input_shape)

# Upload de imagem pelo usuário
uploaded_file = st.file_uploader("Envie uma foto de flor", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Abre e exibe a imagem
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_column_width=True)

    # Preprocessamento dinâmico
    _, h, w, c = model.input_shape  # batch, height, width, channels
    img = img.resize((w, h))
    if c == 1:
        img = img.convert("L")  # grayscale

    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predição
    preds = model.predict(x)[0]
    labels = ["Não é flor", "É flor"]
    idx = np.argmax(preds)
    conf = preds[idx] * 100

    # Resultado na tela
    st.write(f"**Resultado:** {labels[idx]}")
    st.write(f"**Confiança:** {conf:.2f}%")
