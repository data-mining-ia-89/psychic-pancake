# yolo_server/main.py - WORKING VERSION WITH REAL YOLO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import io
from PIL import Image
import logging
import time
import os

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YOLO Detection API",
    description="Real YOLO object detection API",
    version="1.0.0"
)

# Global variables for the model
model = None
model_loaded = False

def load_yolo_model():
    """Load the YOLO model at startup"""
    global model, model_loaded
    
    try:
        logger.info("🔄 Loading YOLO model...")

        # Try to load a custom model first, otherwise YOLOv8n by default
        model_paths = [
            "./models/yolo_custom.pt",
            "./yolov8n.pt",
            "yolov8n.pt"  # Auto-download if missing
        ]
        
        for model_path in model_paths:
            try:
                if os.path.exists(model_path) or model_path == "yolov8n.pt":
                    model = YOLO(model_path)
                    logger.info(f"✅ YOLO model loaded: {model_path}")
                    model_loaded = True
                    return True
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {model_path}: {e}")
                continue
        
        raise Exception("No YOLO model could be loaded")
        
    except Exception as e:
        logger.error(f"❌ Error loading YOLO model: {e}")
        model_loaded = False
        return False

# Load the model at startup
load_yolo_model()

@app.get("/")
def read_root():
    return {
        "message": "YOLO Detection API", 
        "status": "healthy",
        "model_loaded": model_loaded
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "service": "yolo-api",
        "model_status": model_loaded
    }

@app.get("/model/info")
def model_info():
    """Informations sur le modèle YOLO chargé"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")
    
    try:
        return {
            "model_type": "YOLOv8",
            "classes_count": len(model.names),
            "class_names": model.names,
            "input_size": "640x640",
            "status": "ready"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

def preprocess_image(image_bytes: bytes, target_size=(640, 640)):
    """Preprocess the image for YOLO"""
    try:
        # Convert bytes to PIL image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array for OpenCV
        img_array = np.array(image)

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr
        
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

def format_yolo_results(results, confidence_threshold=0.25):
    """Format YOLO results for the API"""
    detections = []
    
    try:
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Extract data from each detection
                    confidence = float(boxes.conf[i])

                    # Filter by confidence threshold    
                    if confidence < confidence_threshold:
                        continue
                    
                    class_id = int(boxes.cls[i])
                    class_name = model.names[class_id] if class_id < len(model.names) else "unknown"

                    # Bounding box coordinates
                    bbox = boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
                    
                    detection = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": {
                            "x1": round(bbox[0], 2),
                            "y1": round(bbox[1], 2), 
                            "x2": round(bbox[2], 2),
                            "y2": round(bbox[3], 2),
                            "width": round(bbox[2] - bbox[0], 2),
                            "height": round(bbox[3] - bbox[1], 2)
                        }
                    }
                    
                    detections.append(detection)
        
        return detections
        
    except Exception as e:
        logger.error(f"Error formatting YOLO results: {e}")
        return []

@app.post("/predict")
async def predict_objects(
    image: UploadFile = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45
):
    """
    Object detection with YOLO - REAL VERSION
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="YOLO model not loaded. Check server logs."
        )
    
    # Check file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    try:
        start_time = time.time()

        # Read the image
        image_bytes = await image.read()
        logger.info(f"📥 Processing image: {image.filename} ({len(image_bytes)} bytes)")

        # Preprocess the image
        processed_image = preprocess_image(image_bytes)

        # YOLO Configuration
        model.conf = confidence  # Confidence threshold
        model.iou = iou_threshold  # IoU threshold for NMS

        # Run YOLO detection
        logger.info("🔍 Running YOLO detection...")
        results = model(processed_image, verbose=False)

        # Format results
        detections = format_yolo_results(results, confidence)

        processing_time = round((time.time() - start_time) * 1000, 2)  # in ms

        # Detection statistics
        detection_stats = {
            "total_detections": len(detections),
            "high_confidence_count": sum(1 for d in detections if d["confidence"] > 0.7),
            "classes_detected": list(set(d["class_name"] for d in detections)),
            "processing_time_ms": processing_time
        }
        
        logger.info(f"✅ Detection completed: {len(detections)} objects found in {processing_time}ms")
        
        return JSONResponse(content={
            "status": "success",
            "filename": image.filename,
            "image_size_bytes": len(image_bytes),
            "detections": detections,
            "statistics": detection_stats,
            "model_info": {
                "model_type": "YOLOv8",
                "confidence_threshold": confidence,
                "iou_threshold": iou_threshold
            }
        })
        
    except ValueError as ve:
        logger.error(f"❌ Image processing error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
        
    except Exception as e:
        logger.error(f"❌ Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(images: list[UploadFile] = File(...)):
    """Traitement par batch de plusieurs images"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")
    
    if len(images) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    start_time = time.time()
    
    for i, image in enumerate(images):
        try:
            logger.info(f"📥 Processing batch image {i+1}/{len(images)}: {image.filename}")

            # Read and process the image
            image_bytes = await image.read()
            processed_image = preprocess_image(image_bytes)

            # YOLO Configuration
            model.conf = 0.25
            model.iou = 0.45

            # Run YOLO detection
            yolo_results = model(processed_image, verbose=False)
            detections = format_yolo_results(yolo_results, 0.25)
            
            results.append({
                "image_index": i,
                "filename": image.filename,
                "status": "success",
                "detections": detections,
                "objects_count": len(detections)
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing {image.filename}: {e}")
            results.append({
                "image_index": i,
                "filename": image.filename,
                "status": "error",
                "error": str(e),
                "detections": [],
                "objects_count": 0
            })
    
    total_time = round((time.time() - start_time) * 1000, 2)
    
    return JSONResponse(content={
        "batch_status": "completed",
        "total_images": len(images),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "total_processing_time_ms": total_time,
        "results": results
    })

@app.post("/reload_model")
async def reload_model():
    """Recharger le modèle YOLO (utile après fine-tuning)"""
    global model, model_loaded
    
    try:
        logger.info("🔄 Reloading YOLO model...")
        success = load_yolo_model()
        
        if success:
            return {"status": "success", "message": "Model reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
            
    except Exception as e:
        logger.error(f"❌ Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for the unified AI API
@app.post("/analyze/image")
async def analyze_image_unified(
    image: UploadFile = File(...),
    task: str = "detection"
):
    """
    Unified endpoint for the main AI API
    Compatible with the format expected by the unified API
    """
    if task not in ["detection", "classification"]:
        raise HTTPException(status_code=400, detail="Task must be 'detection' or 'classification'")
    
    try:
        # Read and process the image directly
        image_bytes = await image.read()
        processed_image = preprocess_image(image_bytes)

        # YOLO Configuration
        model.conf = 0.25
        model.iou = 0.45

        # Run YOLO detection
        results = model(processed_image, verbose=False)
        detections = format_yolo_results(results, 0.25)
        
        if task == "detection":
            return {
                "task": "detection",
                "result": {
                    "object_detection": {
                        "objects_count": len(detections),
                        "objects_detected": detections
                    }
                }
            }
        
        elif task == "classification":
            # For classification, return the object with the highest confidence
            if detections:
                best_detection = max(detections, key=lambda x: x["confidence"])
                return {
                    "task": "classification", 
                    "result": {
                        "image_classification": {
                            "main_class": best_detection["class_name"],
                            "confidence": best_detection["confidence"],
                            "all_detections": detections
                        }
                    }
                }
            else:
                return {
                    "task": "classification",
                    "result": {
                        "image_classification": {
                            "main_class": "no_object_detected",
                            "confidence": 0.0,
                            "all_detections": []
                        }
                    }
                }
        
    except Exception as e:
        logger.error(f"Error in unified endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Check if the model is loaded
    if not model_loaded:
        logger.error("❌ Cannot start server: YOLO model not loaded")
        exit(1)
    
    logger.info("🚀 Starting YOLO API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)