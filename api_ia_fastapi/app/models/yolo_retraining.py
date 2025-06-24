# app/models/yolo_retraining.py

import os
import requests
import zipfile
from ultralytics import YOLO
import yaml
from pathlib import Path
import cv2
import numpy as np
from hdfs import InsecureClient
import shutil

class YOLORetrainer:
    def __init__(self, hdfs_url="http://namenode:9870", hdfs_user="hadoop"):
        self.hdfs_client = InsecureClient(hdfs_url, user=hdfs_user)
        self.data_dir = Path("./yolo_training_data")
        self.model = None
        
    def download_images_from_hdfs(self, hdfs_image_path="/data/images"):
        """Télécharger les images depuis HDFS pour le re-entraînement"""
        
        print(f"📥 Téléchargement des images depuis HDFS: {hdfs_image_path}")
        
        # Créer le répertoire local
        local_images_dir = self.data_dir / "raw_images"
        local_images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Lister les fichiers dans HDFS
            files = self.hdfs_client.list(hdfs_image_path)
            
            for file_name in files:
                if file_name.endswith(('.jpg', '.jpeg', '.png', '.zip')):
                    hdfs_file_path = f"{hdfs_image_path}/{file_name}"
                    local_file_path = local_images_dir / file_name
                    
                    print(f"📥 Téléchargement: {file_name}")
                    
                    # Télécharger le fichier
                    with self.hdfs_client.read(hdfs_file_path) as reader:
                        with open(local_file_path, 'wb') as writer:
                            shutil.copyfileobj(reader, writer)
                    
                    # Si c'est un zip, l'extraire
                    if file_name.endswith('.zip'):
                        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                            zip_ref.extractall(local_images_dir)
                        os.remove(local_file_path)  # Supprimer le zip après extraction
            
            print(f"✅ Images téléchargées dans: {local_images_dir}")
            return local_images_dir
            
        except Exception as e:
            print(f"❌ Erreur téléchargement HDFS: {e}")
            return None
    
    def preprocess_images(self, images_dir, target_size=(640, 640)):
        """Prétraitement des images pour YOLO"""
        
        processed_dir = self.data_dir / "processed_images"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        processed_count = 0
        
        print(f"🔧 Prétraitement des images...")
        
        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    source_path = os.path.join(root, file)
                    
                    try:
                        # Lire l'image
                        img = cv2.imread(source_path)
                        if img is None:
                            continue
                        
                        # Redimensionner
                        img_resized = cv2.resize(img, target_size)
                        
                        # Amélioration de la qualité
                        img_enhanced = cv2.bilateralFilter(img_resized, 9, 75, 75)
                        
                        # Sauvegarder
                        processed_path = processed_dir / f"img_{processed_count:06d}.jpg"
                        cv2.imwrite(str(processed_path), img_enhanced)
                        
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            print(f"📷 Traité {processed_count} images...")
                            
                    except Exception as e:
                        print(f"⚠️ Erreur traitement {file}: {e}")
        
        print(f"✅ {processed_count} images prétraitées")
        return processed_dir, processed_count
    
    def generate_annotations(self, images_dir, method="auto"):
        """Génération d'annotations pour les images"""
        
        annotations_dir = self.data_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        if method == "auto":
            return self._auto_annotate(images_dir, annotations_dir)
        else:
            return self._manual_annotate_template(images_dir, annotations_dir)
    
    def _auto_annotate(self, images_dir, annotations_dir):
        """Annotation automatique avec un modèle YOLO pré-entraîné"""
        
        print("🤖 Annotation automatique avec YOLOv8...")
        
        # Charger modèle pré-entraîné pour l'annotation
        pretrained_model = YOLO('yolov8n.pt')
        
        annotated_count = 0
        
        for img_file in images_dir.glob("*.jpg"):
            try:
                # Prédiction avec le modèle pré-entraîné
                results = pretrained_model(str(img_file))
                
                # Créer fichier d'annotation YOLO format
                annotation_file = annotations_dir / f"{img_file.stem}.txt"
                
                with open(annotation_file, 'w') as f:
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                # Format YOLO: class_id x_center y_center width height (normalisé)
                                cls_id = int(box.cls[0])
                                x_center = float(box.xywhn[0][0])
                                y_center = float(box.xywhn[0][1])
                                width = float(box.xywhn[0][2])
                                height = float(box.xywhn[0][3])
                                
                                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
                
                annotated_count += 1
                
                if annotated_count % 50 == 0:
                    print(f"📝 Annoté {annotated_count} images...")
                    
            except Exception as e:
                print(f"⚠️ Erreur annotation {img_file}: {e}")
        
        print(f"✅ {annotated_count} images annotées automatiquement")
        return annotated_count
    
    def _manual_annotate_template(self, images_dir, annotations_dir):
        """Créer des templates pour annotation manuelle"""
        
        print("📝 Création de templates pour annotation manuelle...")
        
        # Créer un fichier README pour l'annotation manuelle
        readme_content = """
# Instructions d'annotation manuelle

Pour chaque image dans le dossier images/, créez un fichier .txt correspondant dans annotations/

Format YOLO:
class_id x_center y_center width height

Où toutes les valeurs sont normalisées entre 0 et 1.

Exemple pour image001.jpg -> annotations/image001.txt:
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2

Classes suggérées:
0: person
1: vehicle  
2: animal
3: object
"""
        
        with open(annotations_dir / "README_ANNOTATION.md", 'w') as f:
            f.write(readme_content)
        
        # Créer des fichiers d'annotation vides
        empty_count = 0
        for img_file in images_dir.glob("*.jpg"):
            annotation_file = annotations_dir / f"{img_file.stem}.txt"
            if not annotation_file.exists():
                annotation_file.touch()
                empty_count += 1
        
        print(f"📝 {empty_count} fichiers d'annotation créés pour annotation manuelle")
        return empty_count
    
    def create_dataset_config(self, dataset_dir, class_names=None):
        """Créer le fichier de configuration YOLO"""
        
        if class_names is None:
            class_names = ['person', 'vehicle', 'animal', 'object']  # Classes par défaut
        
        config = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        config_file = dataset_dir / 'dataset.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        print(f"✅ Configuration créée: {config_file}")
        return config_file
    
    def train_yolo_model(self, dataset_config, epochs=100, imgsz=640, batch_size=16):
        """Lancer l'entraînement YOLO"""
        
        print(f"🚀 Début de l'entraînement YOLO...")
        print(f"📊 Paramètres: epochs={epochs}, imgsz={imgsz}, batch={batch_size}")
        
        # Charger le modèle de base
        self.model = YOLO('yolov8n.pt')  # Partir du modèle pré-entraîné
        
        # Lancer l'entraînement
        results = self.model.train(
            data=str(dataset_config),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name='custom_yolo_retrained',
            save=True,
            cache=True,
            device='auto',  # Utilise GPU si disponible
            workers=4,
            optimizer='SGD',
            lr0=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            patience=50,  # Early stopping
            verbose=True
        )
        
        print(f"✅ Entraînement terminé!")
        return results
    
    def evaluate_model(self, dataset_config):
        """Évaluer les performances du modèle re-entraîné"""
        
        if self.model is None:
            # Charger le meilleur modèle sauvegardé
            model_path = "./runs/detect/custom_yolo_retrained/weights/best.pt"
            self.model = YOLO(model_path)
        
        print("📊 Évaluation du modèle...")
        
        # Validation sur le dataset
        results = self.model.val(data=str(dataset_config))
        
        # Afficher les métriques
        print(f"📈 mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"📈 mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        return results
    
    def save_model_for_production(self, output_path="./models/yolo_custom.pt"):
        """Sauvegarder le modèle pour la production"""
        
        if self.model is None:
            model_path = "./runs/detect/custom_yolo_retrained/weights/best.pt"
            self.model = YOLO(model_path)
        
        # Copier le modèle vers le répertoire de production
        best_model_path = "./runs/detect/custom_yolo_retrained/weights/best.pt"
        shutil.copy2(best_model_path, output_path)
        
        print(f"✅ Modèle sauvegardé pour production: {output_path}")
        return output_path
    
    def test_custom_model(self, test_image_path):
        """Tester le modèle custom sur une image"""
        
        if self.model is None:
            model_path = "./models/yolo_custom.pt"
            self.model = YOLO(model_path)
        
        print(f"🧪 Test du modèle custom sur: {test_image_path}")
        
        results = self.model(test_image_path)
        
        # Afficher les résultats
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    print(f"Détecté: classe {cls_id}, confiance: {confidence:.3f}")
            else:
                print("Aucun objet détecté")
        
        return results

# Pipeline complet de re-entraînement
def full_yolo_retraining_pipeline():
    """Pipeline complet de re-entraînement YOLO avec images HDFS"""
    
    print("🚀 === PIPELINE COMPLET RE-ENTRAÎNEMENT YOLO ===")
    
    # Initialiser le retrainer
    retrainer = YOLORetrainer()
    
    try:
        # 1. Télécharger images depuis HDFS
        print("\n📥 ÉTAPE 1: Téléchargement depuis HDFS")
        raw_images_dir = retrainer.download_images_from_hdfs()
        
        if raw_images_dir is None:
            print("❌ Échec téléchargement HDFS")
            return False
        
        # 2. Prétraitement des images
        print("\n🔧 ÉTAPE 2: Prétraitement des images")
        processed_images_dir, img_count = retrainer.preprocess_images(raw_images_dir)
        
        if img_count < 10:
            print(f"⚠️ Trop peu d'images ({img_count}), minimum 10 requis")
            return False
        
        # 3. Génération d'annotations
        print("\n📝 ÉTAPE 3: Génération d'annotations")
        annotation_count = retrainer.generate_annotations(processed_images_dir, method="auto")
        
        # 4. Création du dataset YOLO
        print("\n📂 ÉTAPE 4: Création dataset YOLO")
        dataset_dir = retrainer.create_yolo_dataset(processed_images_dir, retrainer.data_dir / "annotations")
        
        # 5. Configuration du dataset
        print("\n⚙️ ÉTAPE 5: Configuration dataset")
        dataset_config = retrainer.create_dataset_config(dataset_dir)
        
        # 6. Entraînement (adapté selon le nombre d'images)
        print("\n🚀 ÉTAPE 6: Entraînement YOLO")
        epochs = min(50, max(10, img_count // 5))  # Adapter epochs selon données
        batch_size = min(16, max(1, img_count // 10))  # Adapter batch size
        
        training_results = retrainer.train_yolo_model(
            dataset_config, 
            epochs=epochs, 
            batch_size=batch_size
        )
        
        # 7. Évaluation
        print("\n📊 ÉTAPE 7: Évaluation du modèle")
        eval_results = retrainer.evaluate_model(dataset_config)
        
        # 8. Sauvegarde pour production
        print("\n💾 ÉTAPE 8: Sauvegarde pour production")
        production_model_path = retrainer.save_model_for_production()
        
        print("\n🎉 === RE-ENTRAÎNEMENT YOLO TERMINÉ AVEC SUCCÈS ===")
        print(f"✅ Images traitées: {img_count}")
        print(f"✅ Annotations générées: {annotation_count}")
        print(f"✅ Epochs d'entraînement: {epochs}")
        print(f"✅ Modèle sauvegardé: {production_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans le pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

# Script pour intégration avec l'API principale
def retrain_yolo_from_hadoop():
    """Fonction d'entrée pour re-entraînement depuis l'API principale"""
    
    success = full_yolo_retraining_pipeline()
    
    if success:
        # Redémarrer le service YOLO avec le nouveau modèle
        print("🔄 Redémarrage du service YOLO avec nouveau modèle...")
        # Ici vous pourriez ajouter le code pour redémarrer le container YOLO
        return {"status": "success", "message": "YOLO model retrained successfully"}
    else:
        return {"status": "error", "message": "YOLO retraining failed"}

if __name__ == "__main__":
    full_yolo_retraining_pipeline()

# Ajouter la méthode manquante à la classe YOLORetrainer
def create_yolo_dataset(self, images_dir, annotations_dir, train_split=0.8):
    """Créer la structure de dataset YOLO"""
    
    dataset_dir = self.data_dir / "yolo_dataset"
    
    # Créer structure YOLO
    for split in ['train', 'val']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Lister toutes les images avec annotations
    image_files = []
    for img_file in images_dir.glob("*.jpg"):
        annotation_file = annotations_dir / f"{img_file.stem}.txt"
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            image_files.append(img_file)
    
    # Split train/val
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * train_split)
    
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Copier les fichiers
    for split, files in [('train', train_files), ('val', val_files)]:
        for img_file in files:
            # Copier image
            shutil.copy2(img_file, dataset_dir / split / 'images' / img_file.name)
            
            # Copier annotation
            annotation_file = annotations_dir / f"{img_file.stem}.txt"
            shutil.copy2(annotation_file, dataset_dir / split / 'labels' / f"{img_file.stem}.txt")
    
    print(f"✅ Dataset créé: {len(train_files)} train, {len(val_files)} val")
    return dataset_dir

# Ajouter la méthode à la classe YOLORetrainer
YOLORetrainer.create_yolo_dataset = create_yolo_dataset