# api_ia_fastapi/app/services/llm_service.py
# INTÉGRATION LM STUDIO POUR LE PROJET

import requests
import json
import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)

class LMStudioService:
    """Service pour intégrer LM Studio via son API REST"""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self.api_url = f"{base_url}/v1/chat/completions"
        self.models_url = f"{base_url}/v1/models"
        self.available = False
        self.current_model = None
        
        # Vérifier si LM Studio est disponible
        self._check_availability()
    
    def _check_availability(self):
        """Vérifier si LM Studio est accessible"""
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
    
    def is_available(self) -> bool:
        """Vérifier si le service est disponible"""
        return self.available
    
    def get_models(self) -> Dict[str, Any]:
        """Récupérer la liste des modèles disponibles"""
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_sentiment(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyser le sentiment d'un texte via LM Studio
        """
        if not self.available:
            raise Exception("LM Studio service not available")
        
        start_time = time.time()
        
        # Prompt optimisé pour l'analyse de sentiment
        system_prompt = """You are a sentiment analysis expert. Analyze the sentiment of the given text and respond with ONLY a JSON object containing:
- "sentiment": one of "POSITIVE", "NEGATIVE", or "NEUTRAL"
- "confidence": a number between 0 and 1
- "reasoning": a brief explanation

Example response:
{"sentiment": "POSITIVE", "confidence": 0.85, "reasoning": "Text expresses satisfaction and positive emotions"}"""
        
        user_prompt = f"Analyze the sentiment of this text: {text}"
        
        payload = {
            "model": self.current_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # Faible pour plus de consistance
            "max_tokens": 200,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            processing_time = round((time.time() - start_time) * 1000, 2)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Parser la réponse JSON du LLM
                try:
                    parsed_result = json.loads(content.strip())
                    
                    return {
                        "sentiment": {
                            "label": parsed_result.get("sentiment", "NEUTRAL"),
                            "confidence": float(parsed_result.get("confidence", 0.5)),
                            "reasoning": parsed_result.get("reasoning", ""),
                            "text_length": len(text),
                            "processing_time_ms": processing_time,
                            "model_used": self.current_model
                        }
                    }
                except json.JSONDecodeError:
                    # Fallback si le LLM ne retourne pas du JSON valide
                    return self._fallback_sentiment_analysis(content, processing_time)
            
            else:
                raise Exception(f"LM Studio API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise Exception(f"Sentiment analysis failed: {str(e)}")
    
    def classify_topic(self, text: str, categories: list = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Classifier le sujet d'un texte via LM Studio
        """
        if not self.available:
            raise Exception("LM Studio service not available")
        
        if categories is None:
            categories = ["technology", "business", "entertainment", "sports", "politics", "health", "science", "general"]
        
        start_time = time.time()
        
        system_prompt = f"""You are a text classification expert. Classify the given text into one of these categories: {', '.join(categories)}.
Respond with ONLY a JSON object containing:
- "category": the most appropriate category from the list
- "confidence": a number between 0 and 1
- "reasoning": a brief explanation

Example response:
{{"category": "technology", "confidence": 0.90, "reasoning": "Text discusses software and programming concepts"}}"""
        
        user_prompt = f"Classify this text: {text}"
        
        payload = {
            "model": self.current_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 150,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            processing_time = round((time.time() - start_time) * 1000, 2)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                try:
                    parsed_result = json.loads(content.strip())
                    
                    return {
                        "classification": {
                            "category": parsed_result.get("category", "general"),
                            "confidence": float(parsed_result.get("confidence", 0.5)),
                            "reasoning": parsed_result.get("reasoning", ""),
                            "available_categories": categories,
                            "processing_time_ms": processing_time,
                            "model_used": self.current_model
                        }
                    }
                except json.JSONDecodeError:
                    return self._fallback_classification(content, categories, processing_time)
            
            else:
                raise Exception(f"LM Studio API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error in topic classification: {e}")
            raise Exception(f"Topic classification failed: {str(e)}")
    
    def summarize_text(self, text: str, max_sentences: int = 3, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Résumer un texte via LM Studio
        """
        if not self.available:
            raise Exception("LM Studio service not available")
        
        start_time = time.time()
        
        system_prompt = f"""You are a text summarization expert. Create a concise summary of the given text in maximum {max_sentences} sentences.
Respond with ONLY a JSON object containing:
- "summary": the summarized text
- "compression_ratio": ratio of summary length to original length
- "key_points": array of key points (max 5)

Example response:
{{"summary": "Brief summary text here.", "compression_ratio": 0.25, "key_points": ["point1", "point2"]}}"""
        
        user_prompt = f"Summarize this text: {text}"
        
        payload = {
            "model": self.current_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 300,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            processing_time = round((time.time() - start_time) * 1000, 2)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                try:
                    parsed_result = json.loads(content.strip())
                    summary = parsed_result.get("summary", "")
                    
                    return {
                        "summarization": {
                            "summary": summary,
                            "original_length": len(text),
                            "summary_length": len(summary),
                            "compression_ratio": len(summary) / len(text) if text else 0,
                            "key_points": parsed_result.get("key_points", []),
                            "processing_time_ms": processing_time,
                            "model_used": self.current_model
                        }
                    }
                except json.JSONDecodeError:
                    return self._fallback_summarization(content, text, processing_time)
            
            else:
                raise Exception(f"LM Studio API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            raise Exception(f"Summarization failed: {str(e)}")
    
    def _fallback_sentiment_analysis(self, content: str, processing_time: float) -> Dict[str, Any]:
        """Analyse de sentiment de fallback si JSON parsing échoue"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["positive", "good", "great", "excellent"]):
            sentiment = "POSITIVE"
            confidence = 0.7
        elif any(word in content_lower for word in ["negative", "bad", "terrible", "awful"]):
            sentiment = "NEGATIVE" 
            confidence = 0.7
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
        
        return {
            "sentiment": {
                "label": sentiment,
                "confidence": confidence,
                "reasoning": "Fallback analysis based on keyword detection",
                "processing_time_ms": processing_time,
                "model_used": self.current_model
            }
        }
    
    def _fallback_classification(self, content: str, categories: list, processing_time: float) -> Dict[str, Any]:
        """Classification de fallback"""
        # Chercher la première catégorie mentionnée dans la réponse
        content_lower = content.lower()
        detected_category = "general"
        
        for category in categories:
            if category.lower() in content_lower:
                detected_category = category
                break
        
        return {
            "classification": {
                "category": detected_category,
                "confidence": 0.6,
                "reasoning": "Fallback classification based on keyword detection",
                "processing_time_ms": processing_time,
                "model_used": self.current_model
            }
        }
    
    def _fallback_summarization(self, content: str, original_text: str, processing_time: float) -> Dict[str, Any]:
        """Résumé de fallback"""
        # Prendre les premières phrases comme résumé
        sentences = original_text.split('. ')
        summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else original_text
        
        return {
            "summarization": {
                "summary": summary,
                "original_length": len(original_text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(original_text) if original_text else 0,
                "processing_time_ms": processing_time,
                "model_used": self.current_model
            }
        }

# Instance globale du service
lm_studio_service = LMStudioService()