# src/app.py

import os
import logging

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Configuração de logging ===
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    filename="app.log",
)

# === Contador simples de requisições ===
if "requests" not in st.session_state:
    st.session_state.requests = 0

# === Sidebar: instruções e métricas ===
st.sidebar.title("🌸 Flower Classifier")
st.sidebar.markdown("""
**Como usar**  
1. Faça upload de JPG/PNG.  
2. Aguarde a classificação.

**Sobre**  
Classificador de 5 flores: daisy, dandelion, rose, sunflower, tulip.
""")
st.sidebar.metric("Requisições totais", st.session_state.requests)

# === Carregamento do modelo ===
HF_REPO_ID  = os.getenv("HF_REPO_ID", "Gallorafael2222/Cnn")
HF_FILENAME = os.getenv("HF_FILENAME", "flower_classifier.h5")
CLASS_LABELS = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

@st.cache_resource
def load_flower_model():
    model_fp = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        library_name="streamlit_app",
    )
    logging.info(f"Modelo carregado de {model_fp}")
    return load_model(model_fp)

model = load_flower_model()

# === Título principal ===
st.title("🌸 Deploy Rede Neural CNN – Classificador de Flores")

uploaded_file = st.file_uploader("Envie uma foto de flor", type=["jpg","jpeg","png"])
if uploaded_file:
    # Incrementa contagem de requisições
    st.session_state.requests += 1

    try:
        # Exibe imagem
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_container_width=True)

        # Preprocessamento dinâmico
        _, h, w, c = model.input_shape
        img = img.resize((w, h))
        if c == 1:
            img = img.convert("L")
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # Predição
        preds = model.predict(x)[0]
        if preds.ndim != 1 or preds.size == 0:
            raise ValueError("Saída do modelo inválida.")

    except Exception as e:
        st.error(f"❌ Erro ao processar a imagem: {e}")
        logging.error(f"Erro no prediction: {e}")
        st.stop()

    # Identifica melhor classe e confiança
    idx  = int(np.argmax(preds))
    conf = float(preds[idx] * 100)

    # Verificação de consistência
    if len(CLASS_LABELS) != preds.size:
        st.error(f"Erro: modelo retornou {preds.size} classes, mas CLASS_LABELS tem {len(CLASS_LABELS)} itens.")
    else:
        # Mostra resultado principal
        st.subheader(f"🔍 Resultado: **{CLASS_LABELS[idx].capitalize()}**")
        st.write(f"Confiança: **{conf:.2f}%**")

        # Exibe probabilidades de todas as classes
        probs = {label.capitalize(): float(p * 100) for label, p in zip(CLASS_LABELS, preds)}
        df = pd.DataFrame.from_dict(probs, orient="index", columns=["Confiança (%)"])
        df = df.sort_values("Confiança (%)", ascending=False)

        st.subheader("📊 Probabilidades por classe")
        st.bar_chart(df)
        st.table(df.style.format("{:.2f}%"))

        # Log da predição
        logging.info(f"Predição: {CLASS_LABELS[idx]} ({conf:.2f}%), vet={probs}")
