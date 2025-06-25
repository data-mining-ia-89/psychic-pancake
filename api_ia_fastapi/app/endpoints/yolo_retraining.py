# api_ia_fastapi/app/endpoints/yolo_retraining.py
"""
INTÉGRATION RÉ-ENTRAÎNEMENT YOLO DANS L'API PRINCIPALE
Endpoints pour déclencher et monitorer le ré-entraînement
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
import json
import os
from pathlib import Path
import requests

# Import du pipeline de ré-entraînement
from .yolo_retraining_pipeline import YOLORetrainingPipeline

logger = logging.getLogger(__name__)

# Router pour les endpoints YOLO
router = APIRouter(prefix="/yolo", tags=["yolo-retraining"])

# Variables globales pour le suivi des tâches
retraining_status = {
    "is_running": False,
    "current_task": None,
    "progress": 0,
    "start_time": None,
    "last_update": None,
    "results": {}
}

class RetrainingRequest(BaseModel):
    hdfs_images_path: Optional[str] = "/data/images/scraped"
    epochs: Optional[int] = 50
    batch_size: Optional[int] = 16
    learning_rate: Optional[float] = 0.01
    force_restart: Optional[bool] = False

class RetrainingStatus(BaseModel):
    is_running: bool
    progress: int
    current_step: Optional[str]
    estimated_time_remaining: Optional[str]
    results: Optional[Dict]

@router.post("/retrain/start")
async def start_retraining(
    request: RetrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Démarrer le ré-entraînement YOLO avec les images HDFS
    """
    global retraining_status
    
    # Vérifier si un ré-entraînement est en cours
    if retraining_status["is_running"] and not request.force_restart:
        raise HTTPException(
            status_code=409,
            detail="Retraining already in progress. Use force_restart=true to stop current task."
        )
    
    try:
        # Valider l'accès HDFS
        namenode_url = "http://namenode:9870"
        list_url = f"{namenode_url}/webhdfs/v1{request.hdfs_images_path}?op=LISTSTATUS"
        
        response = requests.get(list_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot access HDFS path: {request.hdfs_images_path}"
            )
        
        files_info = response.json()["FileStatuses"]["FileStatus"]
        image_files = [f for f in files_info if f["pathSuffix"].lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(image_files) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough images in HDFS path: {len(image_files)} found, minimum 10 required"
            )
        
        # Initialiser le statut
        retraining_status.update({
            "is_running": True,
            "current_task": "initialization",
            "progress": 0,
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "results": {},
            "config": {
                "hdfs_path": request.hdfs_images_path,
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "total_images": len(image_files)
            }
        })
        
        # Lancer la tâche en arrière-plan
        background_tasks.add_task(
            run_retraining_background,
            request
        )
        
        return {
            "status": "started",
            "message": "YOLO retraining started successfully",
            "task_id": retraining_status["start_time"],
            "estimated_duration": "15-30 minutes",
            "images_found": len(image_files),
            "config": retraining_status["config"]
        }
        
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Hadoop cluster: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error starting retraining: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start retraining: {str(e)}"
        )

@router.get("/retrain/status")
async def get_retraining_status() -> RetrainingStatus:
    """
    Obtenir le statut du ré-entraînement en cours
    """
    global retraining_status
    
    # Estimer le temps restant
    estimated_time = None
    if retraining_status["is_running"] and retraining_status["start_time"]:
        start_time = datetime.fromisoformat(retraining_status["start_time"])
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if retraining_status["progress"] > 10:
            total_estimated = (elapsed / retraining_status["progress"]) * 100
            remaining = total_estimated - elapsed
            estimated_time = f"{int(remaining // 60)}m {int(remaining % 60)}s"
    
    return RetrainingStatus(
        is_running=retraining_status["is_running"],
        progress=retraining_status["progress"],
        current_step=retraining_status["current_task"],
        estimated_time_remaining=estimated_time,
        results=retraining_status["results"] if retraining_status["results"] else None
    )

@router.post("/retrain/stop")
async def stop_retraining():
    """
    Arrêter le ré-entraînement en cours
    """
    global retraining_status
    
    if not retraining_status["is_running"]:
        raise HTTPException(
            status_code=404,
            detail="No retraining task is currently running"
        )
    
    retraining_status.update({
        "is_running": False,
        "current_task": "stopped",
        "progress": 0,
        "last_update": datetime.now().isoformat()
    })
    
    return {
        "status": "stopped",
        "message": "Retraining task stopped successfully"
    }

@router.get("/retrain/history")
async def get_retraining_history():
    """
    Historique des ré-entraînements précédents
    """
    try:
        history_dir = Path("./yolo_retraining")
        history_files = list(history_dir.glob("**/pipeline_report.json"))
        
        history = []
        for file_path in history_files[-10:]:  # 10 derniers
            try:
                with open(file_path, 'r') as f:
                    report = json.load(f)
                history.append({
                    "date": report.get("start_time"),
                    "success": report.get("success"),
                    "steps_completed": report.get("steps_completed", []),
                    "model_path": report.get("final_model_path"),
                    "evaluation": report.get("evaluation", {})
                })
            except Exception as e:
                logger.warning(f"Error reading history file {file_path}: {e}")
                continue
        
        return {
            "history": sorted(history, key=lambda x: x["date"], reverse=True),
            "total_runs": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return {"history": [], "total_runs": 0}

@router.get("/model/current")
async def get_current_model_info():
    """
    Informations sur le modèle YOLO actuellement déployé
    """
    try:
        # Vérifier les modèles disponibles
        models_dir = Path("./models")
        models = {}
        
        for model_file in models_dir.glob("*.pt"):
            model_info = {
                "path": str(model_file),
                "size_mb": round(model_file.stat().st_size / (1024*1024), 2),
                "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            }
            
            # Lire métadonnées si disponibles
            metadata_file = models_dir / f"{model_file.stem}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                model_info.update(metadata)
            
            models[model_file.name] = model_info
        
        # Déterminer le modèle actuel
        current_model = None
        if "yolo_custom_production.pt" in models:
            current_model = "yolo_custom_production.pt"
        elif "yolov8n.pt" in models:
            current_model = "yolov8n.pt"
        
        return {
            "current_model": current_model,
            "available_models": models,
            "yolo_service_status": await check_yolo_service_status()
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/switch")
async def switch_model(model_name: str):
    """
    Changer le modèle YOLO utilisé par le service
    """
    try:
        models_dir = Path("./models")
        model_path = models_dir / model_name
        
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found"
            )
        
        # Copier le modèle vers le nom de production
        production_path = models_dir / "yolo_current.pt"
        import shutil
        shutil.copy2(model_path, production_path)
        
        # Redémarrer le service YOLO
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "restart", "yolo-api-server"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                restart_status = "success"
            else:
                restart_status = f"failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            restart_status = "timeout"
        except Exception as e:
            restart_status = f"error: {str(e)}"
        
        return {
            "status": "success",
            "message": f"Switched to model {model_name}",
            "model_path": str(production_path),
            "service_restart": restart_status
        }
        
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def check_yolo_service_status():
    """Vérifier le statut du service YOLO"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://yolo-api:8000/health", timeout=5.0)
            if response.status_code == 200:
                return {"status": "healthy", "details": response.json()}
            else:
                return {"status": "unhealthy", "code": response.status_code}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}

async def run_retraining_background(request: RetrainingRequest):
    """
    Fonction de ré-entraînement exécutée en arrière-plan
    """
    global retraining_status
    
    try:
        logger.info("🚀 Starting background YOLO retraining...")
        
        # Initialiser le pipeline avec configuration personnalisée
        pipeline = YOLORetrainingPipeline(
            hdfs_images_path=request.hdfs_images_path
        )
        
        # Mettre à jour la configuration
        pipeline.config.update({
            "training_epochs": request.epochs,
            "batch_size": request.batch_size,
            "learning_rate": request.learning_rate
        })
        
        # Hook pour mettre à jour le statut pendant le pipeline
        def update_progress(step: str, progress: int):
            retraining_status.update({
                "current_task": step,
                "progress": progress,
                "last_update": datetime.now().isoformat()
            })
            logger.info(f"📊 Retraining progress: {step} ({progress}%)")
        
        # Simuler les étapes avec mise à jour du statut
        update_progress("downloading_from_hdfs", 10)
        await asyncio.sleep(1)  # Permettre à FastAPI de traiter d'autres requêtes
        
        if not pipeline.step_1_download_from_hdfs():
            raise Exception("HDFS download failed")
        
        update_progress("preprocessing_images", 25)
        await asyncio.sleep(1)
        
        processed_count = pipeline.step_2_preprocess_images()
        if processed_count == 0:
            raise Exception("No images processed")
        
        update_progress("auto_annotation", 40)
        await asyncio.sleep(1)
        
        annotated_count = pipeline.step_3_auto_annotate(processed_count)
        if annotated_count < 10:
            raise Exception(f"Too few annotated images: {annotated_count}")
        
        update_progress("creating_dataset", 55)
        await asyncio.sleep(1)
        
        dataset_config = pipeline.step_4_create_dataset(annotated_count)
        
        update_progress("training_model", 70)
        await asyncio.sleep(1)
        
        model_path = pipeline.step_5_train_model(dataset_config)
        
        update_progress("evaluating_model", 85)
        await asyncio.sleep(1)
        
        evaluation = pipeline.step_6_evaluate_model(model_path)
        
        update_progress("deploying_model", 95)
        await asyncio.sleep(1)
        
        deployment_success = pipeline.step_7_deploy_model(model_path)
        
        # Finaliser
        final_results = {
            "success": True,
            "model_path": model_path,
            "processed_images": processed_count,
            "annotated_images": annotated_count,
            "evaluation": evaluation,
            "deployment_success": deployment_success,
            "completion_time": datetime.now().isoformat()
        }
        
        retraining_status.update({
            "is_running": False,
            "current_task": "completed",
            "progress": 100,
            "last_update": datetime.now().isoformat(),
            "results": final_results
        })
        
        logger.info("✅ Background YOLO retraining completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Background retraining failed: {e}")
        
        retraining_status.update({
            "is_running": False,
            "current_task": "failed",
            "progress": 0,
            "last_update": datetime.now().isoformat(),
            "results": {
                "success": False,
                "error": str(e),
                "completion_time": datetime.now().isoformat()
            }
        })


# INTÉGRATION DANS LE MAIN.PY PRINCIPAL
"""
Pour intégrer dans votre api_ia_fastapi/app/main.py, ajoutez:

from .endpoints.yolo_retraining import router as yolo_router
app.include_router(yolo_router)

Et créez aussi ce endpoint simple dans main.py:
"""

@app.post("/retrain/yolo")
async def retrain_yolo_simple():
    """Endpoint simple pour déclencher le ré-entraînement depuis Hadoop"""
    try:
        # Import du pipeline
        from .yolo_retraining_pipeline import YOLORetrainingPipeline
        
        pipeline = YOLORetrainingPipeline()
        results = pipeline.run_complete_pipeline()
        
        if results["success"]:
            return {
                "status": "success",
                "message": "YOLO model retrained successfully with Hadoop images",
                "model_path": results["final_model_path"],
                "steps_completed": results["steps_completed"],
                "evaluation": results.get("evaluation", {})
            }
        else:
            return {
                "status": "error",
                "message": "YOLO retraining failed",
                "errors": results["errors"]
            }
            
    except Exception as e:
        logger.error(f"Simple retraining error: {e}")
        raise HTTPException(status_code=500, detail=str(e))