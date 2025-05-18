# test_app.py
import os
import pytest
import requests

# URL base da sua app em teste (ajuste para dev/local ou CI)
BASE_URL = os.getenv("APP_URL", "http://localhost:8501")

@pytest.fixture(scope="module")
def sample_image(tmp_path):
    # baixa ou copia uma imagem de teste para enviar
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Flower_poster_2.jpg/800px-Flower_poster_2.jpg"
    img_path = tmp_path / "flower.jpg"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(img_path, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    return img_path

def test_healthcheck():
    """Verifica se a página inicial responde 200."""
    r = requests.get(BASE_URL)
    assert r.status_code == 200

def test_prediction_endpoint(sample_image):
    """Envia imagem e espera JSON com resultado e confiança."""
    files = {"file": open(sample_image, "rb")}
    r = requests.post(f"{BASE_URL}/predict", files=files)
    assert r.status_code == 200
    data = r.json()
    assert "result" in data and "confidence" in data
    assert isinstance(data["result"], str)
    assert isinstance(data["confidence"], float)
