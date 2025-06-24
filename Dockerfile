# Dockerfile (à la racine du projet IA)
FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier TOUT le code de l'application
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p /app/models /app/logs

# Exposer le port
EXPOSE 8001

# Commande de démarrage avec le bon chemin
CMD ["uvicorn", "api_ia_fastapi.app.main:app", "--host", "0.0.0.0", "--port", "8001"]