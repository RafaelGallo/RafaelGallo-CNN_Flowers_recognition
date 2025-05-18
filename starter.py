# starter.py
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from config import MODEL_PATH, CLASS_LABELS

def predict(img_path):
    # Carrega modelo
    model = load_model(MODEL_PATH)
    # Preprocessa
    img = Image.open(img_path).convert("RGB")
    _, h, w, c = model.input_shape
    img = img.resize((w, h))
    arr = image.img_to_array(img) / 255.0
    x = np.expand_dims(arr, axis=0)
    # Prediz
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    conf = preds[idx] * 100
    return CLASS_LABELS[idx], conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classifica uma imagem de flor")
    parser.add_argument("image", help="Caminho para o arquivo de imagem")
    args = parser.parse_args()
    label, confidence = predict(args.image)
    print(f"Resultado: {label} (confiança: {confidence:.2f}%)")
