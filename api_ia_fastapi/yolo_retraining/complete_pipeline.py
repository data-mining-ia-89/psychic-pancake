# yolo_retraining/complete_pipeline.py
"""
PIPELINE COMPLET DE RÉ-ENTRAÎNEMENT YOLO
Conforme au cahier des charges du projet Hadoop et IA
"""

import os
import requests
import zipfile
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import subprocess

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLORetrainingPipeline:
    """
    Pipeline complet de ré-entraînement YOLO avec images HDFS
    Étapes: HDFS → Prétraitement → Annotation → Entraînement → Validation
    """
    
    def __init__(self, 
                 namenode_url: str = "http://namenode:9870",
                 hdfs_images_path: str = "/data/images/scraped",
                 output_dir: str = "./yolo_retraining"):
        
        self.namenode_url = namenode_url
        self.hdfs_images_path = hdfs_images_path
        self.output_dir = Path(output_dir)
        
        # Créer la structure de dossiers
        self.setup_directories()
        
        # Configuration par défaut
        self.config = {
            "target_image_size": (640, 640),
            "min_images_required": 50,
            "train_val_split": 0.8,
            "annotation_confidence_threshold": 0.25,
            "training_epochs": 50,
            "batch_size": 16,
            "learning_rate": 0.01
        }
        
        logger.info(f"🚀 YOLO Retraining Pipeline initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🌐 HDFS source: {hdfs_images_path}")
    
    def setup_directories(self):
        """Créer la structure de dossiers nécessaire"""
        subdirs = [
            "raw_images", "processed_images", "annotations", 
            "dataset/train/images", "dataset/train/labels",
            "dataset/val/images", "dataset/val/labels",
            "models", "results", "logs"
        ]
        
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Directory structure created")
    
    def step_1_download_from_hdfs(self) -> bool:
        """
        ÉTAPE 1: Télécharger les images depuis HDFS
        """
        logger.info("📥 ÉTAPE 1: Téléchargement des images depuis HDFS...")
        
        try:
            # Utiliser l'API WebHDFS pour lister les fichiers
            list_url = f"{self.namenode_url}/webhdfs/v1{self.hdfs_images_path}?op=LISTSTATUS"
            response = requests.get(list_url, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"❌ Erreur accès HDFS: {response.status_code}")
                return False
            
            files_info = response.json()["FileStatuses"]["FileStatus"]
            image_files = [f for f in files_info if f["pathSuffix"].lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            logger.info(f"📊 Trouvé {len(image_files)} images dans HDFS")
            
            if len(image_files) < self.config["min_images_required"]:
                logger.warning(f"⚠️ Pas assez d'images ({len(image_files)} < {self.config['min_images_required']})")
                return False
            
            # Télécharger chaque image
            downloaded_count = 0
            for i, file_info in enumerate(image_files[:200]):  # Limiter à 200 images
                try:
                    filename = file_info["pathSuffix"]
                    download_url = f"{self.namenode_url}/webhdfs/v1{self.hdfs_images_path}/{filename}?op=OPEN"
                    
                    img_response = requests.get(download_url, timeout=60, allow_redirects=True)
                    if img_response.status_code == 200:
                        local_path = self.output_dir / "raw_images" / filename
                        with open(local_path, 'wb') as f:
                            f.write(img_response.content)
                        downloaded_count += 1
                        
                        if downloaded_count % 20 == 0:
                            logger.info(f"📥 Téléchargé {downloaded_count}/{len(image_files)} images...")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erreur téléchargement {filename}: {e}")
                    continue
            
            logger.info(f"✅ Téléchargement terminé: {downloaded_count} images")
            return downloaded_count >= self.config["min_images_required"]
            
        except Exception as e:
            logger.error(f"❌ Erreur HDFS: {e}")
            return False
    
    def step_2_preprocess_images(self) -> int:
        """
        ÉTAPE 2: Prétraitement des images
        """
        logger.info("🔧 ÉTAPE 2: Prétraitement des images...")
        
        raw_images_dir = self.output_dir / "raw_images"
        processed_images_dir = self.output_dir / "processed_images"
        
        image_files = list(raw_images_dir.glob("*"))
        processed_count = 0
        
        for img_path in image_files:
            try:
                # Lire l'image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Redimensionner à la taille cible
                img_resized = cv2.resize(img, self.config["target_image_size"])
                
                # Amélioration qualité
                img_enhanced = cv2.bilateralFilter(img_resized, 9, 75, 75)
                
                # Normalisation de luminosité
                lab = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                img_final = cv2.merge([l, a, b])
                img_final = cv2.cvtColor(img_final, cv2.COLOR_LAB2BGR)
                
                # Sauvegarder avec nom standardisé
                output_path = processed_images_dir / f"img_{processed_count:06d}.jpg"
                cv2.imwrite(str(output_path), img_final, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                processed_count += 1
                
                if processed_count % 50 == 0:
                    logger.info(f"🔧 Traité {processed_count} images...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Erreur traitement {img_path}: {e}")
                continue
        
        logger.info(f"✅ Prétraitement terminé: {processed_count} images")
        return processed_count
    
    def step_3_auto_annotate(self, processed_count: int) -> int:
        """
        ÉTAPE 3: Annotation automatique avec modèle pré-entraîné
        """
        logger.info("🤖 ÉTAPE 3: Annotation automatique...")
        
        # Charger modèle YOLO pré-entraîné pour annotation
        pretrained_model = YOLO('yolov8n.pt')
        
        processed_images_dir = self.output_dir / "processed_images"
        annotations_dir = self.output_dir / "annotations"
        
        annotated_count = 0
        
        for img_file in processed_images_dir.glob("*.jpg"):
            try:
                # Prédiction avec modèle pré-entraîné
                results = pretrained_model(str(img_file), conf=self.config["annotation_confidence_threshold"])
                
                # Créer fichier annotation YOLO
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
                
                # Ne garder que les images avec au moins une annotation
                if annotation_file.stat().st_size > 0:
                    annotated_count += 1
                else:
                    annotation_file.unlink()  # Supprimer fichier vide
                
                if annotated_count % 25 == 0:
                    logger.info(f"📝 Annoté {annotated_count} images...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Erreur annotation {img_file}: {e}")
                continue
        
        logger.info(f"✅ Annotation terminée: {annotated_count} images annotées")
        return annotated_count
    
    def step_4_create_dataset(self, annotated_count: int) -> str:
        """
        ÉTAPE 4: Créer le dataset YOLO (train/val split)
        """
        logger.info("📂 ÉTAPE 4: Création du dataset YOLO...")
        
        processed_images_dir = self.output_dir / "processed_images"
        annotations_dir = self.output_dir / "annotations"
        
        # Lister toutes les images avec annotations
        annotated_images = []
        for img_file in processed_images_dir.glob("*.jpg"):
            annotation_file = annotations_dir / f"{img_file.stem}.txt"
            if annotation_file.exists() and annotation_file.stat().st_size > 0:
                annotated_images.append(img_file)
        
        logger.info(f"📊 Images avec annotations: {len(annotated_images)}")
        
        # Mélanger et séparer train/val
        np.random.shuffle(annotated_images)
        split_idx = int(len(annotated_images) * self.config["train_val_split"])
        
        train_images = annotated_images[:split_idx]
        val_images = annotated_images[split_idx:]
        
        logger.info(f"📊 Train: {len(train_images)}, Validation: {len(val_images)}")
        
        # Copier les fichiers dans la structure YOLO
        for split, images in [('train', train_images), ('val', val_images)]:
            for img_file in images:
                # Copier image
                dest_img = self.output_dir / "dataset" / split / "images" / img_file.name
                shutil.copy2(img_file, dest_img)
                
                # Copier annotation
                annotation_file = annotations_dir / f"{img_file.stem}.txt"
                dest_label = self.output_dir / "dataset" / split / "labels" / f"{img_file.stem}.txt"
                shutil.copy2(annotation_file, dest_label)
        
        # Créer fichier de configuration YOLO
        dataset_config = {
            'path': str((self.output_dir / "dataset").absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': 80,  # Nombre de classes COCO
            'names': self._get_coco_class_names()
        }
        
        config_file = self.output_dir / "dataset.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(dataset_config, f)
        
        logger.info(f"✅ Dataset créé: {config_file}")
        return str(config_file)
    
    def step_5_train_model(self, dataset_config: str) -> str:
        """
        ÉTAPE 5: Entraînement du modèle YOLO
        """
        logger.info("🚀 ÉTAPE 5: Entraînement YOLO...")
        
        # Charger modèle de base
        model = YOLO('yolov8n.pt')
        
        # Configuration d'entraînement
        training_config = {
            'data': dataset_config,
            'epochs': self.config["training_epochs"],
            'imgsz': 640,
            'batch': self.config["batch_size"],
            'lr0': self.config["learning_rate"],
            'name': 'yolo_retrained_hadoop',
            'project': str(self.output_dir / "results"),
            'save': True,
            'cache': False,  # Pour éviter les problèmes de mémoire
            'device': 'cpu',  # Utiliser CPU par défaut (changez en 'auto' si GPU)
            'workers': 2,
            'patience': 20,  # Early stopping
            'verbose': True
        }
        
        logger.info(f"📊 Configuration: epochs={training_config['epochs']}, batch={training_config['batch']}")
        
        # Lancer l'entraînement
        try:
            results = model.train(**training_config)
            
            # Chemin du meilleur modèle
            best_model_path = self.output_dir / "results" / "yolo_retrained_hadoop" / "weights" / "best.pt"
            
            if best_model_path.exists():
                # Copier vers dossier models
                final_model_path = self.output_dir / "models" / "yolo_custom_retrained.pt"
                shutil.copy2(best_model_path, final_model_path)
                
                logger.info(f"✅ Entraînement terminé: {final_model_path}")
                return str(final_model_path)
            else:
                raise Exception("Best model not found after training")
                
        except Exception as e:
            logger.error(f"❌ Erreur entraînement: {e}")
            raise
    
    def step_6_evaluate_model(self, model_path: str) -> Dict[str, Any]:
        """
        ÉTAPE 6: Évaluation du modèle ré-entraîné
        """
        logger.info("📊 ÉTAPE 6: Évaluation du modèle...")
        
        try:
            # Charger le modèle ré-entraîné
            custom_model = YOLO(model_path)
            
            # Validation sur le dataset
            dataset_config = str(self.output_dir / "dataset.yaml")
            val_results = custom_model.val(data=dataset_config)
            
            # Comparaison avec modèle original
            original_model = YOLO('yolov8n.pt')
            original_results = original_model.val(data=dataset_config)
            
            # Métriques d'évaluation
            evaluation = {
                "custom_model": {
                    "mAP50": float(val_results.results_dict.get('metrics/mAP50(B)', 0)),
                    "mAP50-95": float(val_results.results_dict.get('metrics/mAP50-95(B)', 0)),
                    "precision": float(val_results.results_dict.get('metrics/precision(B)', 0)),
                    "recall": float(val_results.results_dict.get('metrics/recall(B)', 0))
                },
                "original_model": {
                    "mAP50": float(original_results.results_dict.get('metrics/mAP50(B)', 0)),
                    "mAP50-95": float(original_results.results_dict.get('metrics/mAP50-95(B)', 0)),
                    "precision": float(original_results.results_dict.get('metrics/precision(B)', 0)),
                    "recall": float(original_results.results_dict.get('metrics/recall(B)', 0))
                }
            }
            
            # Calcul amélioration
            improvement = {}
            for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
                custom_val = evaluation['custom_model'][metric]
                original_val = evaluation['original_model'][metric]
                if original_val > 0:
                    improvement[metric] = ((custom_val - original_val) / original_val) * 100
                else:
                    improvement[metric] = 0
            
            evaluation["improvement_percentage"] = improvement
            
            # Sauvegarder l'évaluation
            eval_file = self.output_dir / "results" / "evaluation_report.json"
            with open(eval_file, 'w') as f:
                json.dump(evaluation, f, indent=2)
            
            logger.info(f"📈 mAP50 custom: {evaluation['custom_model']['mAP50']:.3f}")
            logger.info(f"📈 mAP50 original: {evaluation['original_model']['mAP50']:.3f}")
            logger.info(f"📈 Amélioration mAP50: {improvement['mAP50']:.1f}%")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"❌ Erreur évaluation: {e}")
            return {}
    
    def step_7_deploy_model(self, model_path: str) -> bool:
        """
        ÉTAPE 7: Déploiement du modèle ré-entraîné
        """
        logger.info("🚀 ÉTAPE 7: Déploiement du modèle...")
        
        try:
            # Copier le modèle vers le dossier de production
            production_models_dir = Path("./models")
            production_models_dir.mkdir(exist_ok=True)
            
            production_model_path = production_models_dir / "yolo_custom_production.pt"
            shutil.copy2(model_path, production_model_path)
            
            # Créer fichier de métadonnées
            metadata = {
                "model_type": "YOLOv8_custom_retrained",
                "training_date": datetime.now().isoformat(),
                "source_images": self.hdfs_images_path,
                "training_epochs": self.config["training_epochs"],
                "model_path": str(production_model_path),
                "status": "ready_for_production"
            }
            
            metadata_file = production_models_dir / "yolo_custom_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Modèle déployé: {production_model_path}")
            
            # Optionnel: Redémarrer le service YOLO
            try:
                subprocess.run(["docker", "restart", "yolo-api-server"], check=True)
                logger.info("🔄 Service YOLO redémarré")
            except subprocess.CalledProcessError:
                logger.warning("⚠️ Impossible de redémarrer le service YOLO automatiquement")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur déploiement: {e}")
            return False
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Exécuter le pipeline complet de ré-entraînement
        """
        logger.info("🚀 === DÉBUT PIPELINE COMPLET RÉ-ENTRAÎNEMENT YOLO ===")
        
        pipeline_results = {
            "start_time": datetime.now().isoformat(),
            "steps_completed": [],
            "success": False,
            "final_model_path": None,
            "evaluation": {},
            "errors": []
        }
        
        try:
            # ÉTAPE 1: Téléchargement HDFS
            if not self.step_1_download_from_hdfs():
                raise Exception("Échec téléchargement HDFS")
            pipeline_results["steps_completed"].append("hdfs_download")
            
            # ÉTAPE 2: Prétraitement
            processed_count = self.step_2_preprocess_images()
            if processed_count == 0:
                raise Exception("Aucune image traitée")
            pipeline_results["steps_completed"].append("preprocessing")
            
            # ÉTAPE 3: Annotation
            annotated_count = self.step_3_auto_annotate(processed_count)
            if annotated_count < 10:
                raise Exception(f"Trop peu d'images annotées: {annotated_count}")
            pipeline_results["steps_completed"].append("annotation")
            
            # ÉTAPE 4: Création dataset
            dataset_config = self.step_4_create_dataset(annotated_count)
            pipeline_results["steps_completed"].append("dataset_creation")
            
            # ÉTAPE 5: Entraînement
            model_path = self.step_5_train_model(dataset_config)
            pipeline_results["steps_completed"].append("training")
            pipeline_results["final_model_path"] = model_path
            
            # ÉTAPE 6: Évaluation
            evaluation = self.step_6_evaluate_model(model_path)
            pipeline_results["steps_completed"].append("evaluation")
            pipeline_results["evaluation"] = evaluation
            
            # ÉTAPE 7: Déploiement
            if self.step_7_deploy_model(model_path):
                pipeline_results["steps_completed"].append("deployment")
            
            pipeline_results["success"] = True
            pipeline_results["end_time"] = datetime.now().isoformat()
            
            logger.info("🎉 === PIPELINE RÉ-ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS ===")
            
        except Exception as e:
            logger.error(f"❌ Erreur pipeline: {e}")
            pipeline_results["errors"].append(str(e))
            pipeline_results["success"] = False
            pipeline_results["end_time"] = datetime.now().isoformat()
        
        # Sauvegarder rapport final
        report_file = self.output_dir / "pipeline_report.json"
        with open(report_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        return pipeline_results
    
    def _get_coco_class_names(self) -> Dict[int, str]:
        """Noms des classes COCO pour YOLOv8"""
        return {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
            67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }


# SCRIPT D'INTÉGRATION POUR L'API PRINCIPALE
def integrate_yolo_retraining_with_api():
    """
    Fonction d'intégration pour l'API IA principale
    """
    
    def retrain_yolo_from_hadoop() -> Dict[str, Any]:
        """Endpoint pour déclencher le ré-entraînement depuis l'API"""
        
        logger.info("🚀 Démarrage ré-entraînement YOLO depuis API...")
        
        try:
            # Initialiser le pipeline
            pipeline = YOLORetrainingPipeline()
            
            # Lancer le pipeline complet
            results = pipeline.run_complete_pipeline()
            
            if results["success"]:
                return {
                    "status": "success",
                    "message": "YOLO model retrained successfully",
                    "model_path": results["final_model_path"],
                    "evaluation": results["evaluation"],
                    "steps_completed": results["steps_completed"]
                }
            else:
                return {
                    "status": "error", 
                    "message": "YOLO retraining failed",
                    "errors": results["errors"],
                    "steps_completed": results["steps_completed"]
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur intégration API: {e}")
            return {
                "status": "error",
                "message": f"Retraining pipeline failed: {str(e)}"
            }
    
    return retrain_yolo_from_hadoop


# FONCTION PRINCIPALE POUR TESTS
def main():
    """Test complet du pipeline"""
    
    print("🚀 === TEST PIPELINE RÉ-ENTRAÎNEMENT YOLO ===")
    print("Récupération images HDFS → Prétraitement → Annotation → Entraînement")
    
    # Initialiser pipeline
    pipeline = YOLORetrainingPipeline()
    
    # Lancer pipeline complet
    results = pipeline.run_complete_pipeline()
    
    if results["success"]:
        print("\n🎉 === RÉ-ENTRAÎNEMENT RÉUSSI ===")
        print(f"✅ Modèle final: {results['final_model_path']}")
        print(f"✅ Étapes complétées: {', '.join(results['steps_completed'])}")
        
        if results["evaluation"]:
            eval_data = results["evaluation"]
            print(f"📈 mAP50 custom: {eval_data['custom_model']['mAP50']:.3f}")
            print(f"📈 mAP50 original: {eval_data['original_model']['mAP50']:.3f}")
            
            improvement = eval_data["improvement_percentage"]["mAP50"]
            print(f"📈 Amélioration: {improvement:.1f}%")
        
        print("\n🚀 PRÊT POUR LA SOUTENANCE!")
        print("✅ Ré-entraînement YOLO avec images HDFS")
        print("✅ Pipeline automatisé complet")
        print("✅ Évaluation et comparaison modèles")
        print("✅ Déploiement automatique")
        
    else:
        print("\n❌ === RÉ-ENTRAÎNEMENT ÉCHOUÉ ===")
        print(f"Erreurs: {results['errors']}")
        print(f"Étapes complétées: {results['steps_completed']}")


if __name__ == "__main__":
    main()