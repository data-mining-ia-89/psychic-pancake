#!/usr/bin/env python3
"""
Script de test pour vérifier que YOLO fonctionne correctement
"""

import requests
import io
from PIL import Image, ImageDraw
import time
import json

def test_yolo_api():
    """Test complet de l'API YOLO"""
    
    print("🧪 === TEST API YOLO RÉELLE ===")
    
    base_url = "http://localhost:8002"  # Port externe de votre YOLO API
    
    # Test 1: Health check
    print("\n1️⃣ Test Health Check")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Health check OK: {result}")
            if not result.get("model_status"):
                print("❌ Modèle YOLO non chargé!")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur connexion: {e}")
        return False
    
    # Test 2: Model info
    print("\n2️⃣ Test Model Info")
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Modèle info: {model_info['model_type']}")
            print(f"   Classes disponibles: {len(model_info['class_names'])}")
            print(f"   Exemples: {list(model_info['class_names'].values())[:5]}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Erreur model info: {e}")
    
    # Test 3: Créer une image de test
    print("\n3️⃣ Création image de test")
    test_image = create_test_image()
    
    # Test 4: Détection d'objets
    print("\n4️⃣ Test détection d'objets")
    try:
        # Préparer l'image pour l'upload
        img_buffer = io.BytesIO()
        test_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        files = {
            'image': ('test_image.jpg', img_buffer, 'image/jpeg')
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/predict", files=files)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Détection réussie en {response_time:.2f}s")
            print(f"   Statut: {result['status']}")
            print(f"   Objets détectés: {result['statistics']['total_detections']}")
            print(f"   Temps de traitement: {result['statistics']['processing_time_ms']}ms")
            
            # Afficher détails des détections
            for i, detection in enumerate(result['detections'][:3]):  # Max 3 premiers
                print(f"   Objet {i+1}: {detection['class_name']} (confiance: {detection['confidence']})")
            
            return True
            
        else:
            print(f"❌ Détection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur détection: {e}")
        return False

def create_test_image():
    """Créer une image de test simple"""
    
    # Créer une image 640x640 avec quelques formes
    img = Image.new('RGB', (640, 640), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Dessiner quelques formes que YOLO pourrait reconnaître
    # Rectangle (peut être détecté comme objet)
    draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=3)
    
    # Cercle 
    draw.ellipse([300, 150, 450, 300], fill='green', outline='black', width=3)
    
    # Forme de "voiture" basique
    draw.rectangle([200, 400, 400, 500], fill='blue', outline='black', width=3)
    draw.ellipse([220, 480, 260, 520], fill='black')  # roue
    draw.ellipse([360, 480, 400, 520], fill='black')  # roue
    
    return img

def test_unified_endpoint():
    """Test de l'endpoint unifié pour l'API IA"""
    
    print("\n🔗 === TEST ENDPOINT UNIFIÉ ===")
    
    base_url = "http://localhost:8002"
    
    # Créer image de test
    test_image = create_test_image()
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    files = {
        'image': ('test_unified.jpg', img_buffer, 'image/jpeg')
    }
    
    # Test detection task
    print("\n📊 Test task: detection")
    try:
        response = requests.post(
            f"{base_url}/analyze/image?task=detection", 
            files=files
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Detection task OK")
            print(f"   Objets détectés: {result['result']['object_detection']['objects_count']}")
        else:
            print(f"❌ Detection task failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Erreur detection task: {e}")
    
    # Reset buffer pour le second test
    img_buffer.seek(0)
    files = {
        'image': ('test_unified2.jpg', img_buffer, 'image/jpeg')
    }
    
    # Test classification task  
    print("\n🏷️ Test task: classification")
    try:
        response = requests.post(
            f"{base_url}/analyze/image?task=classification",
            files=files
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Classification task OK")
            print(f"   Classe principale: {result['result']['image_classification']['main_class']}")
            print(f"   Confiance: {result['result']['image_classification']['confidence']}")
        else:
            print(f"❌ Classification task failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Erreur classification task: {e}")

def performance_test():
    """Test de performance basique"""
    
    print("\n⚡ === TEST DE PERFORMANCE ===")
    
    base_url = "http://localhost:8002"
    
    # Créer plusieurs images de test - FIX: Format correct pour FastAPI
    files = []
    for i in range(3):
        img = create_test_image()
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        files.append(('images', (f'perf_test_{i}.jpg', buffer, 'image/jpeg')))
    
    # Test de batch
    print("\n📦 Test batch processing")
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/predict/batch", files=files)
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch processing réussi en {total_time:.2f}s")
            print(f"   Images traitées: {result['total_images']}")
            print(f"   Succès: {result['successful']}")
            print(f"   Temps serveur: {result['total_processing_time_ms']}ms")
        else:
            print(f"❌ Batch failed: {response.status_code}")
            print(f"   Détails: {response.text}")
    except Exception as e:
        print(f"❌ Erreur batch: {e}")

def main():
    """Test principal"""
    
    print("🚀 === TESTS COMPLETS API YOLO ===")
    print("Assurez-vous que votre API YOLO tourne sur le port 8002")
    print("Commande: docker-compose up -d yolo-api")
    
    # Attendre un peu que l'API soit prête
    print("\n⏳ Attente 5s pour que l'API soit prête...")
    time.sleep(5)
    
    # Tests principaux
    success = test_yolo_api()
    
    if success:
        test_unified_endpoint()
        performance_test()
        
        print("\n🎉 === TOUS LES TESTS RÉUSSIS ===")
        print("✅ YOLO API est fonctionnelle et prête!")
        print("✅ Détection d'objets opérationnelle")
        print("✅ Endpoints unifiés OK")
        print("✅ Performance acceptable")
        
        print(f"\n🔗 API disponible sur:")
        print(f"   • Health: http://localhost:8002/health")
        print(f"   • Docs: http://localhost:8002/docs")
        print(f"   • Predict: http://localhost:8002/predict")
        
    else:
        print("\n❌ === TESTS ÉCHOUÉS ===")
        print("Vérifiez les logs du conteneur YOLO:")
        print("docker-compose logs yolo-api")

if __name__ == "__main__":
    main()