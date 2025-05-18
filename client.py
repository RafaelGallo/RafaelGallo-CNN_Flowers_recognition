# client.py
import os
import requests

BASE_URL = os.getenv("APP_URL", "http://localhost:8501")

def send_image(img_path):
    files = {"file": open(img_path, "rb")}
    r = requests.post(f"{BASE_URL}/predict", files=files)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Caminho para a imagem a enviar")
    args = parser.parse_args()
    result = send_image(args.image)
    print("Resposta da API:", result)
