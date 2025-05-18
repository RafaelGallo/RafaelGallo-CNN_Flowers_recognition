import streamlit as st
from src.utils import load_flower_model, predict_flower
from PIL import Image

# Labels do seu modelo
CLASS_LABELS = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Carrega o modelo
model = load_flower_model()
st.title("🌸 Classificador de Flores")
st.write("⚙️ Modelo espera input com shape:", model.input_shape)

uploaded_file = st.file_uploader("Envie uma foto de flor", type=["jpg","jpeg","png"])
if uploaded_file:
    st.image(Image.open(uploaded_file), use_container_width=True)
    idx, conf, preds = predict_flower(uploaded_file)
    if len(CLASS_LABELS) != len(preds):
        st.error(f"Erro: modelo retornou {len(preds)} classes, mas você forneceu {len(CLASS_LABELS)} rótulos.")
    else:
        st.write(f"**Resultado:** {CLASS_LABELS[idx].capitalize()}")
        st.write(f"**Confiança:** {conf:.2f}%")
