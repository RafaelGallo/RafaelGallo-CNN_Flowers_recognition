import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

MODEL_URL = "https://huggingface.co/Gallorafael2222/Cnn/resolve/main/flower_classifier.h5"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "flower_classifier.h5")

@st.cache_resource
def load_flower_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        local_fp = get_file(
            fname="flower_classifier.h5",
            origin=MODEL_URL,
            cache_subdir=MODEL_DIR,
            extract=False
        )
        if local_fp != MODEL_PATH:
            os.replace(local_fp, MODEL_PATH)
    return load_model(MODEL_PATH)

model = load_flower_model()

st.title("🌸 Classificador de Flores")
uploaded_file = st.file_uploader("Envie uma foto de flor", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_column_width=True)
    img = img.resize((224,224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    labels = ["Não é flor","É flor"]
    idx = np.argmax(preds)
    conf = preds[idx] * 100
    st.write(f"**Resultado:** {labels[idx]}  \n**Confiança:** {conf:.2f}%")
