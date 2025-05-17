# app.py

import streamlit as st
import os
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Identificação do seu repositório e arquivo no HF Hub
HF_REPO_ID  = "Gallorafael2222/Cnn"
HF_FILENAME = "flower_classifier.h5"
MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, HF_FILENAME)

@st.cache_resource
def load_flower_model():
    # Se não existir localmente, baixa com huggingface_hub
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        local_fp = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            cache_dir=MODEL_DIR,
            library_name="streamlit_app"
        )
        # O hf_hub_download já salva no cache_dir, então só garantimos o nome
        if local_fp != MODEL_PATH:
            os.replace(local_fp, MODEL_PATH)
    return load_model(MODEL_PATH)

# Carrega o modelo (download na primeira execução)
model = load_flower_model()

# Título
st.title("🌸 Classificador de Flores")

# Exibe o input_shape esperado
st.write("⚙️ Modelo espera input com shape:", model.input_shape)

# Upload da imagem
uploaded_file = st.file_uploader("Envie uma foto de flor", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_column_width=True)

    # Preprocessamento dinâmico
    _, h, w, c = model.input_shape
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

    st.write(f"**Resultado:** {labels[idx]}")
    st.write(f"**Confiança:** {conf:.2f}%")
