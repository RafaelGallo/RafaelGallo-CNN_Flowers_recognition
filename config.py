# config.py
import os
from dotenv import load_dotenv

# Carrega variáveis de .env (no root do projeto)
load_dotenv()

# Configurações gerais
STREAMLIT_PORT   = int(os.getenv("STREAMLIT_SERVER_PORT", 8501))
MODEL_REPO_ID    = os.getenv("HF_REPO_ID", "Gallorafael2222/Cnn")
MODEL_FILENAME   = os.getenv("HF_FILENAME", "flower_classifier.h5")
CLASS_LABELS     = os.getenv("CLASS_LABELS", "daisy,dandelion,rose,sunflower,tulip").split(",")

# Paths locais
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
