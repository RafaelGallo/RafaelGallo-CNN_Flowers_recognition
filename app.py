import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Caminho relativo dentro do repositório
MODEL_PATH = os.path.join("model", "flower_classifier.h5")

@st.cache_resource
def load_flower_model():
    return load_model(MODEL_PATH)

model = load_flower_model()

st.title("🌸 Classificador de Flores")

uploaded_file = st.file_uploader("Envie uma foto de flor", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem enviada", use_column_width=True)

    # Pré-processamento
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Previsão
    preds = model.predict(x)[0]
    labels = ["Não é flor", "É flor"]
    idx = np.argmax(preds)
    confidence = preds[idx] * 100

    st.write(f"**Resultado:** {labels[idx]}  \n**Confiança:** {confidence:.2f}%")
