# app.py

import streamlit as st
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Identificação do seu repositório e arquivo no HF Hub
HF_REPO_ID  = "Gallorafael2222/Cnn"
HF_FILENAME = "flower_classifier.h5"

@st.cache_resource
def load_flower_model():
    # Baixa (ou pega do cache) e retorna o caminho do arquivo .h5
    model_fp = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        library_name="streamlit_app"
    )
    # Carrega e retorna o modelo
    return load_model(model_fp)

# Carrega o modelo (download/cache na primeira execução)
model = load_flower_model()

# Título da aplicação
st.title("🌸 Classificador de Flores")

# Mostra qual input_shape o modelo espera
st.write("⚙️ Modelo espera input com shape:", model.input_shape)

# Upload de imagem
uploaded_file = st.file_uploader("Envie uma foto de flor", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_column_width=True)

    # Preprocessamento dinâmico
    _, h, w, c = model.input_shape
    img = img.resize((w, h))
    if c == 1:
        img = img.convert("L")  # grayscale somente se necessário

    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predição
    preds = model.predict(x)[0]
    labels = ["Não é flor", "É flor"]
    idx = np.argmax(preds)
    conf = preds[idx] * 100

    st.write(f"**Resultado:** {labels[idx]}")
    st.write(f"**Confiança:** {conf:.2f}%")
