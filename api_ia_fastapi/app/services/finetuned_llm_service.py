# api_ia_fastapi/app/services/finetuned_llm_service.py

import torch
from transformers import pipeline
import requests
import json
import logging
import time
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class FineTunedLLMService:
    """
    Service for using the fine-tuned LLM model
    Compatible with your existing LM Studio
    """
    
    def __init__(self, finetuned_model_path="./models/finetuned_sentiment_model"):
        self.finetuned_model_path = finetuned_model_path
        self.finetuned_classifier = None
        self.model_loaded = False
        
        # LM Studio Configuration (your existing setup)
        self.lm_studio_url = "http://host.docker.internal:1234/v1/chat/completions"
        self.lm_studio_model = "mistralai/mathstral-7b-v0.1"

        # Load fine-tuned model
        self._load_finetuned_model()
    
    def _load_finetuned_model(self):
        """Load fine-tuned model"""
        try:
            if os.path.exists(self.finetuned_model_path):
                self.finetuned_classifier = pipeline(
                    "text-classification",
                    model=self.finetuned_model_path,
                    tokenizer=self.finetuned_model_path,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.model_loaded = True
                logger.info(f"✅ Fine-tuned model loaded from {self.finetuned_model_path}")
            else:
                logger.warning(f"⚠️ Fine-tuned model not found: {self.finetuned_model_path}")
                self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Error loading fine-tuned model: {e}")
            self.model_loaded = False
    
    def analyze_sentiment_finetuned(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze sentiment with the fine-tuned model
        """
        if not self.model_loaded:
            raise Exception("Fine-tuned model not available")
        
        start_time = time.time()
        
        try:
            # Use the fine-tuned model
            result = self.finetuned_classifier(text)
            processing_time = round((time.time() - start_time) * 1000, 2)

            # Map numeric labels to sentiments
            label_map = {
                "LABEL_0": "NEGATIVE",
                "LABEL_1": "NEUTRAL", 
                "LABEL_2": "POSITIVE"
            }
            
            predicted_label = result[0]['label']
            sentiment = label_map.get(predicted_label, "UNKNOWN")
            confidence = result[0]['score']
            
            return {
                "sentiment": {
                    "label": sentiment,
                    "confidence": float(confidence),
                    "reasoning": f"Fine-tuned model prediction with {confidence:.3f} confidence",
                    "text_length": len(text),
                    "processing_time_ms": processing_time,
                    "model_used": "fine_tuned_distilbert",
                    "model_type": "fine_tuned"
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing fine-tuned: {e}")
            raise Exception(f"Fine-tuned sentiment analysis failed: {str(e)}")
    
    def analyze_sentiment_lm_studio(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze sentiment with LM Studio (your existing setup)
        """
        start_time = time.time()
        
        system_prompt = "You are a sentiment analysis expert. Analyze the sentiment and respond with ONLY a JSON object: {\"sentiment\": \"POSITIVE/NEGATIVE/NEUTRAL\", \"confidence\": 0.85, \"reasoning\": \"brief explanation\"}"
        
        payload = {
            "model": self.lm_studio_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze sentiment: {text}"}
            ],
            "temperature": 0.1,
            "max_tokens": 200,
            "stream": False
        }
        
        try:
            response = requests.post(self.lm_studio_url, json=payload, timeout=30)
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
                            "text_length": len(text),
                            "processing_time_ms": processing_time,
                            "model_used": self.lm_studio_model,
                            "model_type": "lm_studio"
                        }
                    }
                except json.JSONDecodeError:
                    # Fallback si pas de JSON valide
                    return self._fallback_lm_studio_analysis(content, processing_time)
            else:
                raise Exception(f"LM Studio API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Erreur LM Studio: {e}")
            raise Exception(f"LM Studio sentiment analysis failed: {str(e)}")
    
    def analyze_sentiment_comparative(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze sentiment with BOTH models and compare
        """
        start_time = time.time()
        
        results = {
            "text": text,
            "comparative_analysis": True,
            "models_compared": 2
        }

        # Analyze with fine-tuned model
        try:
            if self.model_loaded:
                finetuned_result = self.analyze_sentiment_finetuned(text, metadata)
                results["finetuned_model"] = finetuned_result["sentiment"]
            else:
                results["finetuned_model"] = {"error": "Model not loaded"}
        except Exception as e:
            results["finetuned_model"] = {"error": str(e)}

        # Analyze with LM Studio
        try:
            lm_studio_result = self.analyze_sentiment_lm_studio(text, metadata)
            results["lm_studio_model"] = lm_studio_result["sentiment"]
        except Exception as e:
            results["lm_studio_model"] = {"error": str(e)}

        # Compare results
        if ("error" not in results.get("finetuned_model", {}) and
            "error" not in results.get("lm_studio_model", {})):
            
            ft_sentiment = results["finetuned_model"]["label"]
            lm_sentiment = results["lm_studio_model"]["label"]
            
            agreement = ft_sentiment == lm_sentiment

            # Choose final result (priority to fine-tuned)
            if agreement:
                final_result = results["finetuned_model"]
                final_result["consensus"] = True
            else:
                # Take fine-tuned by default, but report disagreement
                final_result = results["finetuned_model"]
                final_result["consensus"] = False
                final_result["disagreement_details"] = {
                    "finetuned_says": ft_sentiment,
                    "lm_studio_says": lm_sentiment
                }
            
            results["final_sentiment"] = final_result
            results["agreement"] = agreement
        else:
            # Fallback if one of the analyses failed
            if "error" not in results.get("finetuned_model", {}):
                results["final_sentiment"] = results["finetuned_model"]
            elif "error" not in results.get("lm_studio_model", {}):
                results["final_sentiment"] = results["lm_studio_model"]
            else:
                results["final_sentiment"] = {"error": "Both models failed"}
        
        total_time = round((time.time() - start_time) * 1000, 2)
        results["total_processing_time_ms"] = total_time
        
        return results
    
    def _fallback_lm_studio_analysis(self, content: str, processing_time: float) -> Dict[str, Any]:
        """Fallback analysis for LM Studio"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["positive", "good", "great"]):
            sentiment = "POSITIVE"
            confidence = 0.7
        elif any(word in content_lower for word in ["negative", "bad", "terrible"]):
            sentiment = "NEGATIVE"
            confidence = 0.7
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
        
        return {
            "sentiment": {
                "label": sentiment,
                "confidence": confidence,
                "reasoning": "Fallback keyword analysis",
                "processing_time_ms": processing_time,
                "model_used": self.lm_studio_model,
                "model_type": "lm_studio_fallback"
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Information about available models"""
        return {
            "finetuned_model": {
                "available": self.model_loaded,
                "path": self.finetuned_model_path,
                "type": "fine_tuned_distilbert",
                "tasks": ["sentiment_analysis"]
            },
            "lm_studio_model": {
                "available": True,
                "model": self.lm_studio_model,
                "type": "lm_studio_api",
                "tasks": ["sentiment_analysis", "classification", "summarization"]
            },
            "comparative_mode": {
                "available": self.model_loaded,
                "description": "Compares both models and provides consensus"
            }
        }

# Global instance of the service
finetuned_llm_service = FineTunedLLMService()


# Update your main.py to integrate fine-tuning
# api_ia_fastapi/app/main_with_finetuning.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Union, Dict, Any, List
import logging
import time

# Import du nouveau service fine-tuné
from .services.finetuned_llm_service import finetuned_llm_service

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified AI API - Hadoop Project with Fine-tuned LLM",
    description="Single entry point for text (Fine-tuned + LM Studio) and image (YOLO) analysis",
    version="3.0.0"
)

class UnifiedRequest(BaseModel):
    data_type: str  # "text" or "image"
    content: Union[str, dict]
    task: str
    model_preference: Optional[str] = "finetuned"  # "finetuned", "lm_studio", "comparative"
    metadata: Optional[dict] = None

@app.post("/analyze")
async def unified_analyze(request: UnifiedRequest):
    """Single entry point with fine-tuned model support"""
    try:
        if request.data_type == "text":
            result = await process_text_analysis_enhanced(
                text=request.content,
                task=request.task,
                model_preference=request.model_preference,
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
            "model_preference": request.model_preference,
            "result": result,
            "metadata": request.metadata,
            "api_version": "3.0_with_finetuning"
        }
    
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_text_analysis_enhanced(
    text: str, 
    task: str, 
    model_preference: str = "finetuned",
    metadata: dict = None
):
    """Text processing with fine-tuned model"""

    if not text or len(text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text too short for analysis")
    
    if task == "sentiment":
        # Use model according to preference
        if model_preference == "finetuned" and finetuned_llm_service.model_loaded:
            return finetuned_llm_service.analyze_sentiment_finetuned(text, metadata)
        
        elif model_preference == "lm_studio":
            return finetuned_llm_service.analyze_sentiment_lm_studio(text, metadata)
        
        elif model_preference == "comparative" and finetuned_llm_service.model_loaded:
            return finetuned_llm_service.analyze_sentiment_comparative(text, metadata)
        
        else:
            # Fallback to LM Studio if fine-tuned not available
            try:
                return finetuned_llm_service.analyze_sentiment_lm_studio(text, metadata)
            except Exception:
                return await process_text_fallback(text, task, metadata)
    
    elif task == "classification":
        # For classification, use LM Studio (or add fine-tuning classification)
        try:
            return finetuned_llm_service.analyze_classification_lm_studio(text, metadata)
        except Exception:
            return await process_text_fallback(text, task, metadata)
    
    else:
        return await process_text_fallback(text, task, metadata)

async def process_text_fallback(text: str, task: str, metadata: dict = None):
    """Basic fallback analysis"""
    if task == "sentiment":
        positive_words = ["good", "great", "amazing", "excellent", "love", "awesome"]
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
                "reasoning": "Keyword-based fallback analysis",
                "model_used": "fallback_keywords",
                "model_type": "fallback"
            }
        }
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")

async def process_image_analysis(image_data: Union[str, dict], task: str, metadata: dict = None):
    """Image processing - integration with YOLO API (unchanged)"""
    try:
        # YOLO Simulation (to be replaced with real integration)
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

@app.get("/")
def root():
    return {
        "message": "AI API with Fine-tuned LLM + LM Studio", 
        "status": "healthy",
        "version": "3.0.0-finetuned"
    }

@app.get("/health")
async def health_check():
    """Check the health of the API and models"""

    model_info = finetuned_llm_service.get_model_info()
    
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "finetuned_model": "available" if model_info["finetuned_model"]["available"] else "unavailable",
            "lm_studio": "available",
            "yolo": "available"
        },
        "models": model_info
    }

@app.get("/models/comparison")
async def models_comparison():
    """Detailed comparison of models"""

    test_texts = [
        "This product is absolutely amazing! I love it!",
        "Terrible quality, completely disappointed with this purchase.",
        "Average product, does what it should but nothing special."
    ]
    
    comparison_results = []
    
    for text in test_texts:
        try:
            result = finetuned_llm_service.analyze_sentiment_comparative(text)
            comparison_results.append({
                "text": text,
                "finetuned_sentiment": result.get("finetuned_model", {}).get("label"),
                "lm_studio_sentiment": result.get("lm_studio_model", {}).get("label"),
                "agreement": result.get("agreement", False),
                "final_decision": result.get("final_sentiment", {}).get("label")
            })
        except Exception as e:
            comparison_results.append({
                "text": text,
                "error": str(e)
            })
    
    return {
        "comparison_results": comparison_results,
        "model_info": finetuned_llm_service.get_model_info()
    }

@app.post("/analyze/sentiment/finetuned")
async def analyze_sentiment_finetuned_direct(text: str):
    """Direct endpoint for the fine-tuned model"""
    try:
        result = finetuned_llm_service.analyze_sentiment_finetuned(text)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/sentiment/comparative")
async def analyze_sentiment_comparative_direct(text: str):
    """Direct endpoint for model comparison"""
    try:
        result = finetuned_llm_service.analyze_sentiment_comparative(text)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)