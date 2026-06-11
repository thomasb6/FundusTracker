FROM python:3.11-slim

# Installer les dépendances système nécessaires (compilateur, librairies Cairo)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libcairo2-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ca-certificates \
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

# --- BACKEND DE FONDATION (onglet admin expérimental) ---
# torch CPU (index dédié pour éviter les wheels CUDA, inutiles ici) + DINOv2.
# Couche séparée, mise en cache : reconstruite seulement si ce fichier change.
# Mettre INSTALL_FOUNDATION=0 au build pour une image cœur sans torch.
ARG INSTALL_FOUNDATION=1
COPY requirements-foundation.txt ./
RUN if [ "$INSTALL_FOUNDATION" = "1" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
            torch==2.6.0 torchvision==0.21.0 ; \
    fi
# --------------------------------

# 3. On copie le reste de ton code (qui change souvent sur tes petites MAJ)
COPY . ./

# Toutes les données persistantes (base SQLite des comptes, dossiers/patients,
# clé secrète des sessions) vivent ici. Monter un volume hôte sur ce chemin :
#   docker run -v /opt/fundustracker_data:/app/userdata ...
VOLUME ["/app/userdata"]

EXPOSE 8080

# Serveur WSGI de production (le serveur de dev Flask n'est pas fait pour ça).
# 1 worker : l'app charge exemples et état en mémoire à l'import ; 8 threads
# couvrent les requêtes concurrentes. Timeout long pour SIFT/exports PDF.
CMD ["gunicorn", "-b", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "120", "--access-logfile", "-", "main:server"]
