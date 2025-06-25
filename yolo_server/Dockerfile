# yolo_server/Dockerfile - VERSION AVEC YOLO RÉEL

FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système nécessaires pour OpenCV et YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    libgtk-3-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Créer le dossier pour les modèles
RUN mkdir -p /app/models

# Copier le code de l'application
COPY main.py .

# Télécharger YOLOv8n au build (optionnel, peut aussi se faire au runtime)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLO download skipped, will download at runtime"

# Exposer le port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]