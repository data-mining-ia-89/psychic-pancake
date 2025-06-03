#!/bin/bash

HDFS_CMD="hdfs dfs"

# Attendre que HDFS soit prêt (namenode)
until $HDFS_CMD -ls /; do
  echo "⏳ En attente du démarrage de HDFS..."
  sleep 5
done

# Répertoires locaux pour stocker les datasets
DATA_DIR="datasets"
TEXT_DIR="$DATA_DIR/text"
IMAGE_DIR="$DATA_DIR/images"

# Répertoires HDFS
HDFS_TEXT_DIR="/data/text"
HDFS_IMAGE_DIR="/data/images"

# Vérifier si kaggle CLI est installé
if ! command -v kaggle &> /dev/null
then
    echo "❌ Kaggle CLI non installé ! Installez-le avec : pip install kaggle"
    exit 1
fi

# Créer les dossiers locaux
mkdir -p "$TEXT_DIR" "$IMAGE_DIR"

# Télécharger les datasets
echo "📥 Téléchargement des bases de données..."
kaggle datasets download -d snap/amazon-fine-food-reviews -p "$TEXT_DIR" --force
kaggle datasets download -d jessicali9530/celeba-dataset -p "$IMAGE_DIR" --force

# Extraire les fichiers
echo "📦 Extraction des fichiers..."
unzip -o "$TEXT_DIR/amazon-fine-food-reviews.zip" -d "$TEXT_DIR"
unzip -o "$IMAGE_DIR/celeba-dataset.zip" -d "$IMAGE_DIR"

# Vérifier si Hadoop est installé et démarré
if ! hdfs dfs -ls / &> /dev/null
then
    echo "❌ Hadoop HDFS n'est pas accessible ! Vérifiez que le cluster est bien démarré."
    exit 1
fi

# Créer les répertoires HDFS
echo "📂 Création des répertoires HDFS..."
hdfs dfs -mkdir -p "$HDFS_TEXT_DIR"
hdfs dfs -mkdir -p "$HDFS_IMAGE_DIR"

# Copier les fichiers vers HDFS
echo "🚀 Envoi des données vers HDFS..."
hdfs dfs -put -f "$TEXT_DIR/Reviews.csv" "$HDFS_TEXT_DIR/"
hdfs dfs -put -f "$IMAGE_DIR/img_align_celeba.zip" "$HDFS_IMAGE_DIR/"

echo "✅ Données téléchargées et stockées dans HDFS avec succès !"
