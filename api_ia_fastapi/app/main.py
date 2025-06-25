# api_ia_fastapi/app/main.py - SIMPLIFIED ALL-IN-ONE VERSION

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Union, Dict, Any, List
import requests
import json
import logging
import time

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified AI API - Hadoop Project with LM Studio",
    description="Single entry point for text (LM Studio) and image (YOLO) analysis",
    version="2.0.0"
)

namenode_url = "http://namenode:9870"  # Name of the Hadoop container


# ============ LM STUDIO INTEGRATION CLASS ============
class LMStudioService:
    """Service to integrate LM Studio - Simplified Version"""

    def __init__(self, base_url: str = "http://host.docker.internal:1234"):
        self.base_url = base_url
        self.api_url = f"{base_url}/v1/chat/completions"
        self.models_url = f"{base_url}/v1/models"
        self.available = False
        self.current_model = None

        # Check if LM Studio is available
        self._check_availability()
    
    def _check_availability(self):
        """Check if LM Studio is accessible"""
        try:
            response = requests.get(self.models_url, timeout=5)
            if response.status_code == 200:
                models = response.json()
                if models.get("data"):
                    self.available = True
                    self.current_model = models["data"][0]["id"]
                    logger.info(f"✅ LM Studio available with model: {self.current_model}")
                else:
                    logger.warning("⚠️ LM Studio running but no model loaded")
            else:
                logger.warning(f"⚠️ LM Studio responded with status: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ LM Studio not available: {e}")
            self.available = False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment via LM Studio"""
        if not self.available:
            raise Exception("LM Studio service not available")
        
        start_time = time.time()
        
        system_prompt = """You are a sentiment analysis expert. Analyze the sentiment and respond with ONLY a JSON object:
{"sentiment": "POSITIVE/NEGATIVE/NEUTRAL", "confidence": 0.85, "reasoning": "brief explanation"}"""
        
        payload = {
            "model": self.current_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze sentiment: {text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 200,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            processing_time = round((time.time() - start_time) * 1000, 2)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                try:
                    parsed = json.loads(content.strip())
                    return {
                        "sentiment": {
                            "label": parsed.get("sentiment", "NEUTRAL"),
                            "confidence": float(parsed.get("confidence", 0.5)),
                            "reasoning": parsed.get("reasoning", ""),
                            "processing_time_ms": processing_time,
                            "model_used": self.current_model
                        }
                    }
                except json.JSONDecodeError:
                    # Fallback simple
                    content_lower = content.lower()
                    if "positive" in content_lower:
                        sentiment = "POSITIVE"
                        confidence = 0.7
                    elif "negative" in content_lower:
                        sentiment = "NEGATIVE"
                        confidence = 0.7
                    else:
                        sentiment = "NEUTRAL"
                        confidence = 0.5
                    
                    return {
                        "sentiment": {
                            "label": sentiment,
                            "confidence": confidence,
                            "reasoning": "Fallback analysis",
                            "processing_time_ms": processing_time,
                            "model_used": self.current_model
                        }
                    }
            else:
                raise Exception(f"LM Studio API error: {response.status_code}")
        except Exception as e:
            raise Exception(f"Sentiment analysis failed: {str(e)}")

# Global instance of LMStudioService
lm_studio_service = LMStudioService()

# ============ Pydantic MODELS ============
class UnifiedRequest(BaseModel):
    data_type: str  # "text" or "image"
    content: Union[str, dict]  # text or image in base64
    task: str
    metadata: Optional[dict] = None

# ============ MAIN UNIFIED ENDPOINT ============
@app.post("/analyze")
async def unified_analyze(request: UnifiedRequest):
    """Unified endpoint for text (LM Studio) and image (YOLO) analysis"""
    try:
        if request.data_type == "text":
            result = await process_text_analysis(
                text=request.content,
                task=request.task,
                metadata=request.metadata
            )
        elif request.data_type == "image":
            result = await process_image_analysis(
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
            "metadata": request.metadata,
            "api_version": "2.0_simplified"
        }
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ TEXT PROCESSING ============
async def process_text_analysis(text: str, task: str, metadata: dict = None):
    """Process text with LM Studio or fallback"""

    if not text or len(text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text too short for analysis")

    # Try LM Studio first
    if lm_studio_service.available and task == "sentiment":
        try:
            return lm_studio_service.analyze_sentiment(text)
        except Exception as e:
            logger.warning(f"LM Studio failed, using fallback: {e}")

    # Fallback analysis
    return await process_text_fallback(text, task, metadata)

async def process_text_fallback(text: str, task: str, metadata: dict = None):
    """Fallback analysis if LM Studio is unavailable"""
    """Simple keyword-based analysis for sentiment, classification, or summarization"""
    if task == "sentiment":
        positive_words = ["good", "great", "amazing", "excellent", "love", "awesome", "fantastic", "perfect", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "hate", "horrible", "worst", "disappointing", "poor", "disgusting"]
        
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
                "reasoning": "Keyword-based fallback analysis",
                "text_length": len(text),
                "positive_indicators": positive_score,
                "negative_indicators": negative_score,
                "model_used": "fallback_keywords"
            }
        }
    
    elif task == "classification":
        tech_words = ["python", "ai", "machine learning", "technology", "software", "programming", "computer", "algorithm"]
        business_words = ["market", "sales", "revenue", "business", "profit", "company", "finance", "investment"]
        entertainment_words = ["movie", "music", "game", "entertainment", "show", "film", "art", "sport"]
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in tech_words):
            category = "technology"
            confidence = 0.8
        elif any(word in text_lower for word in business_words):
            category = "business"
            confidence = 0.7
        elif any(word in text_lower for word in entertainment_words):
            category = "entertainment"
            confidence = 0.7
        else:
            category = "general"
            confidence = 0.5
        
        return {
            "classification": {
                "category": category,
                "confidence": confidence,
                "reasoning": "Keyword-based classification",
                "model_used": "fallback_keywords"
            }
        }
    
    elif task == "summarization":
        sentences = text.split('. ')
        max_sentences = metadata.get("max_sentences", 3) if metadata else 3
        
        if len(sentences) <= max_sentences:
            summary = text
        else:
            summary = '. '.join(sentences[:max_sentences]) + '.'
        
        return {
            "summarization": {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text),
                "model_used": "fallback_simple"
            }
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported text task: {task}")

# ============ IMAGE PROCESSING ============
async def process_image_analysis(image_data: Union[str, dict], task: str, metadata: dict = None):
    """Image processing - integration with YOLO API"""

    try:
        # Call YOLO API (on internal port 8000)
        yolo_url = "http://yolo-api:8000/analyze/image"

        # For now, simulation - will be replaced by actual YOLO call
        mock_yolo_response = {
            "detections": [
                {
                    "class_name": "person",
                    "confidence": 0.85,
                    "bbox": {"x1": 100, "y1": 150, "x2": 200, "y2": 300}
                }
            ]
        }
        
        if task == "detection":
            return {
                "object_detection": {
                    "objects_count": len(mock_yolo_response["detections"]),
                    "objects_detected": mock_yolo_response["detections"],
                    "model": "yolov8n_via_api"
                }
            }
        
        elif task == "classification":
            if mock_yolo_response["detections"]:
                best_detection = max(mock_yolo_response["detections"], 
                                   key=lambda x: x.get("confidence", 0))
                return {
                    "image_classification": {
                        "main_class": best_detection.get("class_name", "unknown"),
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

# ============ HEALTH CHECK ENDPOINTS ============
@app.get("/")
def root():
    return {
        "message": "AI API with LM Studio integration (simplified)", 
        "status": "healthy",
        "version": "2.0.0-simplified"
    }

@app.get("/health")
async def health_check():
    """Check API and service status"""

    # Re-check LM Studio status
    lm_studio_service._check_availability()
    lm_studio_status = "available" if lm_studio_service.available else "unavailable"
    
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "lm_studio": lm_studio_status,
            "yolo": "available"
        },
        "lm_studio_model": lm_studio_service.current_model if lm_studio_service.available else None
    }

@app.get("/models/status")
async def models_status():
    """État des modèles IA"""
    
    lm_studio_info = {}
    if lm_studio_service.available:
        lm_studio_info = {
            "status": "available",
            "tasks": ["sentiment", "classification", "summarization"],
            "type": "lm_studio_api",
            "current_model": lm_studio_service.current_model
        }
    else:
        lm_studio_info = {
            "status": "unavailable",
            "fallback": "keyword_based_analysis"
        }
    
    return {
        "llm": lm_studio_info,
        "yolo": {
            "status": "available", 
            "tasks": ["detection", "classification"],
            "type": "yolo_api"
        }
    }

# ============ BATCH ENDPOINT FOR HADOOP ============
@app.post("/analyze/batch")
async def batch_analyze(requests: List[UnifiedRequest]):
    """Batch processing for Hadoop data"""
    results = []
    
    for req in requests:
        try:
            if req.data_type == "text":
                result = await process_text_analysis(req.content, req.task, req.metadata)
            else:
                result = await process_image_analysis(req.content, req.task, req.metadata)
            
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
    
    return {
        "batch_results": results, 
        "total_processed": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error")
    }

# ============ SPECIALIZED ENDPOINTS ============
@app.post("/analyze/sentiment")
async def analyze_sentiment_direct(text: str):
    """Direct endpoint for sentiment analysis"""
    try:
        result = await process_text_analysis(text, "sentiment")
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/classify")
async def classify_text_direct(text: str):
    """Direct endpoint for classification"""
    try:
        result = await process_text_analysis(text, "classification")
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)