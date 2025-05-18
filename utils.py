# utils.py

import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Carrega variáveis de .env automaticamente
load_dotenv()

# Leitura de configurações do ambiente
HF_REPO_ID  = os.getenv("HF_REPO_ID", "Gallorafael2222/Cnn")
HF_FILENAME = os.getenv("HF_FILENAME", "flower_classifier.h5")

_model = None  # variável interna para cache

def load_flower_model():
    """
    Baixa (ou usa cache) o arquivo H5 do HF Hub e carrega um único modelo em memória.
    Retorna a instância do modelo Keras.
    """
    global _model
    if _model is None:
        # baixa e retorna o path no cache do hf_hub_download
        model_fp = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            library_name="streamlit_app"
        )
        _model = load_model(model_fp)
    return _model

def preprocess_image(uploaded_file, target_shape):
    """
    Recebe um arquivo PIL.Image ou bytes, converte para RGB, redimensiona para target_shape (h, w, c),
    normaliza em [0,1] e retorna um numpy array com shape (1, h, w, c).
    """
    # Abre a imagem e garante RGB
    img = Image.open(uploaded_file).convert("RGB")
    h, w, c = target_shape
    img = img.resize((w, h))
    if c == 1:
        img = img.convert("L")
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_flower(uploaded_file):
    """
    Carrega o modelo, pré-processa a imagem e retorna (label_idx, confidence, preds_array).
    Usa a variável de ambiente CLASS_LABELS, que deve ser definida em app.py.
    """
    model = load_flower_model()
    # model.input_shape = (None, h, w, c)
    _, h, w, c = model.input_shape
    x = preprocess_image(uploaded_file, (h, w, c))
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx] * 100)
    return idx, conf, preds
