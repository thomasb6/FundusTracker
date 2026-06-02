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

# --- ÉTAPE OPTIMISATION CACHE ---
# 1. On copie UNIQUEMENT le fichier des dépendances Python
COPY requirements.txt ./

# 2. On installe les dépendances. Cette couche sera mise en cache par Docker.
# Elle ne sera réexécutée QUE si tu modifies ton fichier requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt
# --------------------------------

# 3. On copie le reste de ton code (qui change souvent sur tes petites MAJ)
COPY . ./

# Toutes les données persistantes (base SQLite des comptes, dossiers/patients,
# clé secrète des sessions) vivent ici. Monter un volume hôte sur ce chemin :
#   docker run -v /opt/fundustracker_data:/app/userdata ...
VOLUME ["/app/userdata"]

EXPOSE 8080

CMD ["python", "main.py"]
