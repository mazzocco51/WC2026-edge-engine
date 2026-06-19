# ---------------------------------------------------------------------------
# Dockerfile UNICO condiviso da API e dashboard.
# Stessa immagine, comando diverso (vedi docker-compose.yml): meno duplicazione,
# build piu' veloce grazie alla cache dei layer.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Best practice Python in container:
# - non scrivere file .pyc
# - output non bufferizzato (i log compaiono subito)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Installiamo prima SOLO le dipendenze: se il codice cambia ma i requirements
#    no, Docker riusa questo layer dalla cache (build molto piu' rapide).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Poi copiamo il codice applicativo.
COPY app ./app
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit

# Documenta le porte usate (API 8000, Streamlit 8501).
EXPOSE 8000 8501

# Comando di default: il backend API. La dashboard sovrascrive il command
# nel docker-compose.yml.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
