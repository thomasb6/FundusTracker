FROM python:3.9.13-slim

# Installer les dépendances système nécessaires (compilateur, librairies Cairo)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Copier le code dans l’image
COPY . ./

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "main.py"]
