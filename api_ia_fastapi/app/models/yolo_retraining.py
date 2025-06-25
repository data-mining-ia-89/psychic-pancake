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
        
        # Create the local directory
        local_images_dir = self.data_dir / "raw_images"
        local_images_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # List files in HDFS
            files = self.hdfs_client.list(hdfs_image_path)
            
            for file_name in files:
                if file_name.endswith(('.jpg', '.jpeg', '.png', '.zip')):
                    hdfs_file_path = f"{hdfs_image_path}/{file_name}"
                    local_file_path = local_images_dir / file_name

                    print(f"📥 Downloading: {file_name}")

                    # Download the file
                    with self.hdfs_client.read(hdfs_file_path) as reader:
                        with open(local_file_path, 'wb') as writer:
                            shutil.copyfileobj(reader, writer)

                    # If it's a zip file, extract it
                    if file_name.endswith('.zip'):
                        with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                            zip_ref.extractall(local_images_dir)
                        os.remove(local_file_path)  # Remove the zip after extraction

            print(f"✅ Images downloaded to: {local_images_dir}")
            return local_images_dir
            
        except Exception as e:
            print(f"❌ HDFS download error: {e}")
            return None
    
    def preprocess_images(self, images_dir, target_size=(640, 640)):
        """Preprocess images for YOLO"""

        processed_dir = self.data_dir / "processed_images"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        processed_count = 0

        print(f"🔧 Preprocessing images...")

        for root, dirs, files in os.walk(images_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    source_path = os.path.join(root, file)
                    
                    try:
                        # Read the image
                        img = cv2.imread(source_path)
                        if img is None:
                            continue

                        # Resize
                        img_resized = cv2.resize(img, target_size)

                        # Enhance quality
                        img_enhanced = cv2.bilateralFilter(img_resized, 9, 75, 75)

                        # Save
                        processed_path = processed_dir / f"img_{processed_count:06d}.jpg"
                        cv2.imwrite(str(processed_path), img_enhanced)
                        
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            print(f"📷 Processed {processed_count} images...")

                    except Exception as e:
                        print(f"⚠️ Processing error {file}: {e}")

        print(f"✅ {processed_count} images processed")
        return processed_dir, processed_count
    
    def generate_annotations(self, images_dir, method="auto"):
        """Generate annotations for the images"""

        annotations_dir = self.data_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        if method == "auto":
            return self._auto_annotate(images_dir, annotations_dir)
        else:
            return self._manual_annotate_template(images_dir, annotations_dir)
    
    def _auto_annotate(self, images_dir, annotations_dir):
        """Automatic annotation with a pre-trained YOLO model"""

        print("🤖 Automatic annotation with YOLOv8...")

        # Load pre-trained model for annotation
        pretrained_model = YOLO('yolov8n.pt')
        
        annotated_count = 0
        
        for img_file in images_dir.glob("*.jpg"):
            try:
                # Prediction with the pre-trained model
                results = pretrained_model(str(img_file))
                
                # Create YOLO format annotation file
                annotation_file = annotations_dir / f"{img_file.stem}.txt"
                
                with open(annotation_file, 'w') as f:
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                # Format YOLO: class_id x_center y_center width height (normalized)
                                cls_id = int(box.cls[0])
                                x_center = float(box.xywhn[0][0])
                                y_center = float(box.xywhn[0][1])
                                width = float(box.xywhn[0][2])
                                height = float(box.xywhn[0][3])
                                
                                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
                
                annotated_count += 1
                
                if annotated_count % 50 == 0:
                    print(f"📝 Annotated {annotated_count} images...")

            except Exception as e:
                print(f"⚠️ Annotation error {img_file}: {e}")

        print(f"✅ {annotated_count} images annotated automatically")
        return annotated_count
    
    def _manual_annotate_template(self, images_dir, annotations_dir):
        """Create templates for manual annotation"""

        print("📝 Creating templates for manual annotation...")

        # Create a README file for manual annotation
        readme_content = """
# Instructions for Manual Annotation

For each image in the images/ folder, create a corresponding .txt file in annotations/

YOLO Format:
class_id x_center y_center width height

Where all values are normalized between 0 and 1.

Example for image001.jpg -> annotations/image001.txt:
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2

Suggested classes:
0: person
1: vehicle  
2: animal
3: object
"""
        
        with open(annotations_dir / "README_ANNOTATION.md", 'w') as f:
            f.write(readme_content)

        # Create empty annotation files
        empty_count = 0
        for img_file in images_dir.glob("*.jpg"):
            annotation_file = annotations_dir / f"{img_file.stem}.txt"
            if not annotation_file.exists():
                annotation_file.touch()
                empty_count += 1

        print(f"📝 {empty_count} empty annotation files created for manual annotation")
        return empty_count
    
    def create_dataset_config(self, dataset_dir, class_names=None):
        """Create YOLO configuration file"""

        if class_names is None:
            class_names = ['person', 'vehicle', 'animal', 'object']  # Default classes

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
        """Start YOLO training"""

        print(f"🚀 Starting YOLO training...")
        print(f"📊 Parameters: epochs={epochs}, imgsz={imgsz}, batch={batch_size}")

        # Load base model
        self.model = YOLO('yolov8n.pt')  # Start from pre-trained model

        # Start training
        results = self.model.train(
            data=str(dataset_config),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            name='custom_yolo_retrained',
            save=True,
            cache=True,
            device='auto',  # Use GPU if available
            workers=4,
            optimizer='SGD',
            lr0=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            patience=50,  # Early stopping
            verbose=True
        )

        print(f"✅ Training completed!")
        return results
    
    def evaluate_model(self, dataset_config):
        """Evaluate the performance of the fine-tuned model"""

        if self.model is None:
            # Load the best saved model
            model_path = "./runs/detect/custom_yolo_retrained/weights/best.pt"
            self.model = YOLO(model_path)

        print("📊 Evaluating the model...")

        # Validation on the dataset
        results = self.model.val(data=str(dataset_config))

        # Display metrics
        print(f"📈 mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"📈 mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        
        return results
    
    def save_model_for_production(self, output_path="./models/yolo_custom.pt"):
        """Save the model for production"""

        if self.model is None:
            model_path = "./runs/detect/custom_yolo_retrained/weights/best.pt"
            self.model = YOLO(model_path)

        # Copy the model to the production directory
        best_model_path = "./runs/detect/custom_yolo_retrained/weights/best.pt"
        shutil.copy2(best_model_path, output_path)

        print(f"✅ Model saved for production: {output_path}")
        return output_path
    
    def test_custom_model(self, test_image_path):
        """Test the custom model on an image"""

        if self.model is None:
            model_path = "./models/yolo_custom.pt"
            self.model = YOLO(model_path)

        print(f"🧪 Testing the custom model on: {test_image_path}")

        results = self.model(test_image_path)

        # Display results
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    print(f"DDetected: class {cls_id}, confidence: {confidence:.3f}")
            else:
                print("No objects detected")

        return results

# Complete YOLO retraining pipeline
def full_yolo_retraining_pipeline():
    """Complete YOLO retraining pipeline with HDFS images"""

    print("🚀 === COMPLETE YOLO RETRAINING PIPELINE ===")

    # Initialize retrainer
    retrainer = YOLORetrainer()
    
    try:
        # 1. Download images from HDFS
        print("\n📥 STEP 1: Downloading from HDFS")
        raw_images_dir = retrainer.download_images_from_hdfs()
        
        if raw_images_dir is None:
            print("❌ HDFS download failed")
            return False

        # 2. Preprocess images
        print("\n🔧 STEP 2: Preprocessing images")
        processed_images_dir, img_count = retrainer.preprocess_images(raw_images_dir)
        
        if img_count < 10:
            print(f"⚠️ Not enough images ({img_count}), minimum 10 required for training")
            return False

        # 3. Generate annotations
        print("\n📝 STEP 3: Generating annotations")
        annotation_count = retrainer.generate_annotations(processed_images_dir, method="auto")

        # 4. Create YOLO dataset
        print("\n📂 STEP 4: Creating YOLO dataset")
        dataset_dir = retrainer.create_yolo_dataset(processed_images_dir, retrainer.data_dir / "annotations")

        # 5. Configure dataset
        print("\n⚙️ STEP 5: Configuring dataset")
        dataset_config = retrainer.create_dataset_config(dataset_dir)

        # 6. Training (adapted based on the number of images)
        print("\n🚀 STEP 6: Training YOLO")
        epochs = min(50, max(10, img_count // 5))  # Adapt epochs based on data
        batch_size = min(16, max(1, img_count // 10))  # Adapt batch size

        training_results = retrainer.train_yolo_model(
            dataset_config, 
            epochs=epochs, 
            batch_size=batch_size
        )

        # 7. Evaluation
        print("\n📊 STEP 7: Evaluating the model")
        eval_results = retrainer.evaluate_model(dataset_config)

        # 8. Save for production
        print("\n💾 STEP 8: Saving for production")
        production_model_path = retrainer.save_model_for_production()

        print("\n🎉 === YOLO RETRAINING COMPLETED SUCCESSFULLY ===")
        print(f"✅ Processed images: {img_count}")
        print(f"✅ Generated annotations: {annotation_count}")
        print(f"✅ Training epochs: {epochs}")
        print(f"✅ Model saved: {production_model_path}")

        return True
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

# Script for integration with the main API
def retrain_yolo_from_hadoop():
    """Entry point for retraining from the main API"""

    success = full_yolo_retraining_pipeline()
    
    if success:
        # Restart YOLO service with new model
        print("🔄 Restarting YOLO service with new model...")
        # Here you could add code to restart the YOLO container
        return {"status": "success", "message": "YOLO model retrained successfully"}
    else:
        return {"status": "error", "message": "YOLO retraining failed"}

if __name__ == "__main__":
    full_yolo_retraining_pipeline()

# Add missing method to YOLORetrainer class
def create_yolo_dataset(self, images_dir, annotations_dir, train_split=0.8):
    """Create YOLO dataset structure"""

    dataset_dir = self.data_dir / "yolo_dataset"

    # Create YOLO structure
    for split in ['train', 'val']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # List all images with annotations
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

    # Copy files
    for split, files in [('train', train_files), ('val', val_files)]:
        for img_file in files:
            # Copy image
            shutil.copy2(img_file, dataset_dir / split / 'images' / img_file.name)

            # Copy annotation
            annotation_file = annotations_dir / f"{img_file.stem}.txt"
            shutil.copy2(annotation_file, dataset_dir / split / 'labels' / f"{img_file.stem}.txt")

    print(f"✅ Dataset created: {len(train_files)} train, {len(val_files)} val")
    return dataset_dir

# Add method to YOLORetrainer class
YOLORetrainer.create_yolo_dataset = create_yolo_dataset