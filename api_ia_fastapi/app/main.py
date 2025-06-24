# api_ia_fastapi/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Union, Dict, Any
import requests
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified AI API - Hadoop Project",
    description="Single entry point for text and image analysis",
    version="1.0.0"
)

# Modèles de requête
class UnifiedRequest(BaseModel):
    data_type: str  # "text" ou "image"
    content: Union[str, dict]  # texte ou image en base64
    task: str
    metadata: Optional[dict] = None

# ============ ENDPOINT UNIFIÉ PRINCIPAL ============
@app.post("/analyze")
async def unified_analyze(request: UnifiedRequest):
    """
    Point d'entrée unique pour analyse texte et image
    Compatible avec les données Hadoop prétraitées
    """
    try:
        if request.data_type == "text":
            result = await process_text_analysis_simple(
                text=request.content,
                task=request.task,
                metadata=request.metadata
            )
        elif request.data_type == "image":
            result = await process_image_analysis_simple(
                image_data=request.content,
                task=request.task,
                metadata=request.metadata
            )
        else:
            raise HTTPException(status_code=400, detail="data_type must be 'text' or 'image'")
        
        return {
            "status": "success",
            "data_type": request.data_type,
            "task": request.task,
            "result": result,
            "metadata": request.metadata
        }
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ FONCTIONS DE TRAITEMENT SIMPLIFIÉES ============
async def process_text_analysis_simple(text: str, task: str, metadata: dict = None):
    """Traitement simplifié du texte pour test"""
    
    if not text or len(text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text too short for analysis")
    
    if task == "sentiment":
        # Analyse simple basée sur mots-clés
        positive_words = ["good", "great", "amazing", "excellent", "love", "awesome", "fantastic"]
        negative_words = ["bad", "terrible", "awful", "hate", "horrible", "worst"]
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        if positive_score > negative_score:
            sentiment = "POSITIVE"
            confidence = min(0.9, 0.6 + (positive_score * 0.1))
        elif negative_score > positive_score:
            sentiment = "NEGATIVE" 
            confidence = min(0.9, 0.6 + (negative_score * 0.1))
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
        
        return {
            "sentiment": {
                "label": sentiment,
                "confidence": confidence,
                "text_length": len(text),
                "positive_indicators": positive_score,
                "negative_indicators": negative_score
            }
        }
    
    elif task == "classification":
        # Classification simple par mots-clés
        tech_words = ["python", "ai", "machine learning", "technology", "software", "programming"]
        business_words = ["market", "sales", "revenue", "business", "profit", "company"]
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in tech_words):
            return {
                "classification": {
                    "category": "technology",
                    "confidence": 0.8
                }
            }
        elif any(word in text_lower for word in business_words):
            return {
                "classification": {
                    "category": "business", 
                    "confidence": 0.7
                }
            }
        else:
            return {
                "classification": {
                    "category": "general",
                    "confidence": 0.5
                }
            }
    
    elif task == "summarization":
        # Résumé simple - prendre les premières phrases
        sentences = text.split('. ')
        if len(sentences) <= 2:
            summary = text
        else:
            summary = '. '.join(sentences[:2]) + '.'
        
        return {
            "summarization": {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text)
            }
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported text task: {task}")

async def process_image_analysis_simple(image_data: Union[str, dict], task: str, metadata: dict = None):
    """Traitement d'image via l'API YOLO"""
    
    try:
        # Appeler l'API YOLO (qui retourne déjà des résultats mockés)
        yolo_url = "http://yolo-api:8000/predict"
        
        # Pour ce test, on simule juste un appel
        # En réalité il faudrait convertir le base64 en fichier
        mock_yolo_response = {
            "detections": [
                {
                    "class": 0,
                    "label": "person",
                    "confidence": 0.85,
                    "bbox": [100, 150, 200, 300]
                }
            ],
            "status": "success"
        }
        
        if task == "detection":
            return {
                "object_detection": {
                    "objects_count": len(mock_yolo_response["detections"]),
                    "objects_detected": mock_yolo_response["detections"],
                    "model": "yolov8n_mock"
                }
            }
        
        elif task == "classification":
            # Prendre l'objet avec la plus haute confiance
            if mock_yolo_response["detections"]:
                best_detection = max(mock_yolo_response["detections"], 
                                   key=lambda x: x.get("confidence", 0))
                return {
                    "image_classification": {
                        "main_class": best_detection.get("label", "unknown"),
                        "confidence": best_detection.get("confidence", 0)
                    }
                }
            else:
                return {
                    "image_classification": {
                        "main_class": "unknown",
                        "confidence": 0
                    }
                }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported image task: {task}")
    
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")

# ============ ENDPOINTS DE SANTÉ ============
@app.get("/")
def root():
    return {"message": "AI API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "yolo": "available"
        }
    }

@app.get("/models/status")
async def models_status():
    """État des modèles IA"""
    return {
        "llm": {
            "status": "available",
            "tasks": ["sentiment", "classification", "summarization"],
            "type": "simple_rules"
        },
        "yolo": {
            "status": "available", 
            "tasks": ["detection", "classification"],
            "type": "mock_api"
        }
    }

# ============ ENDPOINT BATCH POUR HADOOP ============
@app.post("/analyze/batch")
async def batch_analyze(requests: list[UnifiedRequest]):
    """
    Traitement par batch pour les données Hadoop
    """
    results = []
    
    for req in requests:
        try:
            if req.data_type == "text":
                result = await process_text_analysis_simple(req.content, req.task, req.metadata)
            else:
                result = await process_image_analysis_simple(req.content, req.task, req.metadata)
            
            results.append({
                "id": req.metadata.get("id") if req.metadata else None,
                "status": "success",
                "result": result
            })
        except Exception as e:
            results.append({
                "id": req.metadata.get("id") if req.metadata else None,
                "status": "error",
                "error": str(e)
            })
    
    return {"batch_results": results, "total_processed": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)