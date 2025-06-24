# app/utils/hadoop_formatter.py

from datetime import datetime, timezone
from typing import Dict, Any, List
import json
import uuid

class HadoopFormatter:
    """Formateur pour optimiser les réponses API pour stockage Hadoop"""
    
    @staticmethod
    def format_text_analysis_result(
        original_text: str,
        analysis_result: Dict[str, Any],
        task: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Formater les résultats d'analyse de texte pour Hadoop"""
        
        base_result = {
            # Identifiants et métadonnées
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_type": "text",
            "task": task,
            
            # Données source
            "source": {
                "original_text": original_text,
                "text_length": len(original_text),
                "word_count": len(original_text.split()) if original_text else 0,
                "metadata": metadata or {}
            },
            
            # Résultats selon la tâche
            "analysis": HadoopFormatter._format_text_analysis_by_task(analysis_result, task),
            
            # Métadonnées techniques
            "processing": {
                "model_type": "llm",
                "processing_time_ms": metadata.get("processing_time_ms") if metadata else None,
                "api_version": "1.0",
                "status": "success"
            }
        }
        
        return base_result
    
    @staticmethod
    def _format_text_analysis_by_task(result: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Formater selon le type de tâche d'analyse de texte"""
        
        if task == "sentiment":
            return {
                "sentiment": {
                    "label": result.get("sentiment", "UNKNOWN"),
                    "confidence": float(result.get("confidence", 0.0)),
                    "positive_score": result.get("positive_score", 0.0),
                    "negative_score": result.get("negative_score", 0.0),
                    "neutral_score": result.get("neutral_score", 0.0)
                }
            }
        
        elif task == "classification":
            return {
                "classification": {
                    "category": result.get("category", "unknown"),
                    "confidence": float(result.get("confidence", 0.0)),
                    "all_categories": result.get("all_categories", []),
                    "top_categories": result.get("top_categories", [])[:5]  # Top 5 max
                }
            }
        
        elif task == "summarization":
            return {
                "summarization": {
                    "summary": result.get("summary", ""),
                    "summary_length": len(result.get("summary", "")),
                    "compression_ratio": float(result.get("compression_ratio", 0.0)),
                    "key_points": result.get("key_points", []),
                    "summary_quality_score": result.get("quality_score", 0.0)
                }
            }
        
        else:
            return {"raw_result": result}
    
    @staticmethod
    def format_image_analysis_result(
        image_metadata: Dict[str, Any],
        analysis_result: Dict[str, Any],
        task: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Formater les résultats d'analyse d'image pour Hadoop"""
        
        base_result = {
            # Identifiants et métadonnées
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_type": "image",
            "task": task,
            
            # Données source
            "source": {
                "image_info": {
                    "filename": image_metadata.get("filename", "unknown"),
                    "size_bytes": image_metadata.get("size_bytes", 0),
                    "dimensions": image_metadata.get("dimensions", {}),
                    "format": image_metadata.get("format", "unknown")
                },
                "metadata": metadata or {}
            },
            
            # Résultats selon la tâche
            "analysis": HadoopFormatter._format_image_analysis_by_task(analysis_result, task),
            
            # Métadonnées techniques
            "processing": {
                "model_type": "yolo",
                "model_version": "yolov8n",
                "processing_time_ms": metadata.get("processing_time_ms") if metadata else None,
                "api_version": "1.0",
                "status": "success"
            }
        }
        
        return base_result
    
    @staticmethod
    def _format_image_analysis_by_task(result: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Formater selon le type de tâche d'analyse d'image"""
        
        if task == "detection":
            detections = result.get("detections", [])
            return {
                "object_detection": {
                    "objects_count": len(detections),
                    "objects_detected": [
                        {
                            "class_id": det.get("class", -1),
                            "class_name": det.get("label", "unknown"),
                            "confidence": float(det.get("confidence", 0.0)),
                            "bbox": {
                                "x1": float(det.get("bbox", [0,0,0,0])[0]),
                                "y1": float(det.get("bbox", [0,0,0,0])[1]),
                                "x2": float(det.get("bbox", [0,0,0,0])[2]),
                                "y2": float(det.get("bbox", [0,0,0,0])[3]),
                                "width": float(det.get("bbox", [0,0,0,0])[2] - det.get("bbox", [0,0,0,0])[0]),
                                "height": float(det.get("bbox", [0,0,0,0])[3] - det.get("bbox", [0,0,0,0])[1])
                            }
                        }
                        for det in detections
                    ],
                    "detection_summary": HadoopFormatter._create_detection_summary(detections)
                }
            }
        
        elif task == "classification":
            return {
                "image_classification": {
                    "main_class": result.get("classification", "unknown"),
                    "confidence": float(result.get("confidence", 0.0)),
                    "all_classes": result.get("all_classes", [])
                }
            }
        
        else:
            return {"raw_result": result}
    
    @staticmethod
    def _create_detection_summary(detections: List[Dict]) -> Dict[str, Any]:
        """Créer un résumé des détections pour analyse rapide"""
        
        if not detections:
            return {"total": 0, "classes": {}, "high_confidence_count": 0}
        
        # Compter par classe
        class_counts = {}
        high_confidence_count = 0
        total_confidence = 0
        
        for det in detections:
            class_name = det.get("label", "unknown")
            confidence = det.get("confidence", 0.0)
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += confidence
            
            if confidence > 0.7:
                high_confidence_count += 1
        
        return {
            "total": len(detections),
            "classes": class_counts,
            "high_confidence_count": high_confidence_count,
            "average_confidence": total_confidence / len(detections) if detections else 0.0,
            "most_detected_class": max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else None
        }
    
    @staticmethod
    def format_batch_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Formater les résultats de traitement par batch"""
        
        batch_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Statistiques du batch
        total_count = len(results)
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = total_count - success_count
        
        # Répartition par type et tâche
        type_stats = {}
        task_stats = {}
        
        for result in results:
            if result.get("status") == "success":
                data_type = result.get("data_type", "unknown")
                task = result.get("task", "unknown")
                
                type_stats[data_type] = type_stats.get(data_type, 0) + 1
                task_stats[task] = task_stats.get(task, 0) + 1
        
        return {
            "batch_info": {
                "batch_id": batch_id,
                "timestamp": timestamp,
                "total_items": total_count,
                "successful_items": success_count,
                "failed_items": error_count,
                "success_rate": success_count / total_count if total_count > 0 else 0.0
            },
            "statistics": {
                "by_data_type": type_stats,
                "by_task": task_stats
            },
            "results": results,
            "hadoop_metadata": {
                "partition_key": timestamp.split('T')[0],  # Date pour partitionnement
                "processing_node": "ai-api-node",
                "batch_size": total_count
            }
        }
    
    @staticmethod
    def format_error_result(
        error_message: str,
        data_type: str = "unknown",
        task: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Formater les résultats d'erreur pour Hadoop"""
        
        return {
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_type": data_type,
            "task": task,
            "status": "error",
            "error": {
                "message": error_message,
                "type": "processing_error",
                "metadata": metadata or {}
            },
            "processing": {
                "api_version": "1.0",
                "status": "failed"
            }
        }
    
    @staticmethod
    def create_hive_schema() -> str:
        """Créer le schéma Hive pour stocker les résultats d'analyse"""
        
        schema = """
        CREATE TABLE IF NOT EXISTS ia_analysis_results (
            analysis_id STRING,
            timestamp STRING,
            data_type STRING,
            task STRING,
            status STRING,
            
            -- Source data
            source_text STRING,
            source_metadata MAP<STRING, STRING>,
            
            -- Text analysis results
            sentiment_label STRING,
            sentiment_confidence DOUBLE,
            classification_category STRING,
            classification_confidence DOUBLE,
            summary_text STRING,
            compression_ratio DOUBLE,
            
            -- Image analysis results
            objects_count INT,
            detections ARRAY<STRUCT<
                class_name: STRING,
                confidence: DOUBLE,
                bbox_x1: DOUBLE,
                bbox_y1: DOUBLE,
                bbox_x2: DOUBLE,
                bbox_y2: DOUBLE
            >>,
            main_image_class STRING,
            image_class_confidence DOUBLE,
            
            -- Technical metadata
            model_type STRING,
            processing_time_ms BIGINT,
            api_version STRING,
            
            -- Error information
            error_message STRING
        )
        PARTITIONED BY (
            processing_date STRING,
            data_type_partition STRING
        )
        STORED AS PARQUET
        LOCATION '/data/ia_results'
        """
        
        return schema
    
    @staticmethod
    def prepare_for_hive_insert(formatted_result: Dict[str, Any]) -> Dict[str, Any]:
        """Préparer les données formatées pour insertion Hive"""
        
        # Extraction des champs principaux
        base_fields = {
            "analysis_id": formatted_result.get("analysis_id"),
            "timestamp": formatted_result.get("timestamp"),
            "data_type": formatted_result.get("data_type"),
            "task": formatted_result.get("task"),
            "status": formatted_result.get("processing", {}).get("status", "unknown")
        }
        
        # Champs source
        source = formatted_result.get("source", {})
        base_fields.update({
            "source_text": source.get("original_text"),
            "source_metadata": source.get("metadata", {})
        })
        
        # Champs spécifiques selon le type d'analyse
        analysis = formatted_result.get("analysis", {})
        
        if "sentiment" in analysis:
            sentiment = analysis["sentiment"]
            base_fields.update({
                "sentiment_label": sentiment.get("label"),
                "sentiment_confidence": sentiment.get("confidence")
            })
        
        elif "classification" in analysis:
            classification = analysis["classification"]
            base_fields.update({
                "classification_category": classification.get("category"),
                "classification_confidence": classification.get("confidence")
            })
        
        elif "summarization" in analysis:
            summarization = analysis["summarization"]
            base_fields.update({
                "summary_text": summarization.get("summary"),
                "compression_ratio": summarization.get("compression_ratio")
            })
        
        elif "object_detection" in analysis:
            detection = analysis["object_detection"]
            base_fields.update({
                "objects_count": detection.get("objects_count"),
                "detections": detection.get("objects_detected", [])
            })
        
        elif "image_classification" in analysis:
            img_class = analysis["image_classification"]
            base_fields.update({
                "main_image_class": img_class.get("main_class"),
                "image_class_confidence": img_class.get("confidence")
            })
        
        # Métadonnées techniques
        processing = formatted_result.get("processing", {})
        base_fields.update({
            "model_type": processing.get("model_type"),
            "processing_time_ms": processing.get("processing_time_ms"),
            "api_version": processing.get("api_version")
        })
        
        # Gestion des erreurs
        error = formatted_result.get("error", {})
        if error:
            base_fields["error_message"] = error.get("message")
        
        # Champs de partitionnement
        timestamp = formatted_result.get("timestamp", "")
        base_fields.update({
            "processing_date": timestamp.split('T')[0] if timestamp else None,
            "data_type_partition": formatted_result.get("data_type")
        })
        
        return base_fields

# Exemple d'utilisation
def example_usage():
    """Exemple d'utilisation du formateur Hadoop"""
    
    formatter = HadoopFormatter()
    
    # Exemple résultat sentiment
    text_result = {
        "sentiment": "POSITIVE",
        "confidence": 0.89,
        "positive_score": 0.89,
        "negative_score": 0.05,
        "neutral_score": 0.06
    }
    
    formatted = formatter.format_text_analysis_result(
        original_text="This product is amazing!",
        analysis_result=text_result,
        task="sentiment",
        metadata={"source": "review", "processing_time_ms": 150}
    )
    
    print("Formatage sentiment:")
    print(json.dumps(formatted, indent=2))
    
    # Préparer pour Hive
    hive_ready = formatter.prepare_for_hive_insert(formatted)
    print("\nPrêt pour Hive:")
    print(json.dumps(hive_ready, indent=2))