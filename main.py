from dotenv import load_dotenv
import os

# carrega as variáveis definidas em .env para o os.environ
load_dotenv()

# agora você pode usar
HF_REPO_ID  = os.environ["HF_REPO_ID"]
HF_FILENAME = os.environ["HF_FILENAME"]

# e o token será automaticamente usado pela huggingface_hub