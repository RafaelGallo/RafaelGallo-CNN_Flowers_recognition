# Use uma imagem oficial Python slim
FROM python:3.9-slim

# Evita buffering de saída (para logs em tempo real)
ENV PYTHONUNBUFFERED=1

# Define diretório de trabalho
WORKDIR /app

# Copia apenas arquivos de dependências primeiro (para cache de build)
COPY requirements.txt ./ 

# Instala dependências do sistema necessárias para TensorFlow e h5py
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libhdf5-serial-dev \
      libatlas-base-dev \
      && rm -rf /var/lib/apt/lists/*

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação
COPY . .

# Exponha a porta padrão do Streamlit
EXPOSE 8501

# Variáveis de ambiente para Streamlit
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Comando de inicialização
CMD ["streamlit", "run", "app.py"]

docker build -t flower-classifier-app .
docker run --env-file .env -p 8501:8501 flower-classifier-app
