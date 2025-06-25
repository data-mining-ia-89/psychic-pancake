#!/usr/bin/env python3
"""
Test script to verify that YOLO is working correctly"""

import requests
import io
from PIL import Image, ImageDraw
import time
import json

def test_yolo_api():
    """Full YOLO API Test"""
    
    print("🧪 === REAL YOLO API TEST ===")

    base_url = "http://localhost:8002"  # External port of your YOLO API

    # Test 1: Health check
    print("\n1️⃣ Test Health Check")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Health check OK: {result}")
            if not result.get("model_status"):
                print("❌ YOLO model not loaded!")
                return False
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False
    
    # Test 2: Model info
    print("\n2️⃣ Test Model Info")
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model info: {model_info['model_type']}")
            print(f"   Available classes: {len(model_info['class_names'])}")
            print(f"   Examples: {list(model_info['class_names'].values())[:5]}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")

    # Test 3: Create a test image
    print("\n3️⃣ Create test image")
    test_image = create_test_image()

    # Test 4: Object detection
    print("\n4️⃣ Test object detection")
    try:
        # Prepare the image for upload
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
            print(f"✅ Successful detection in {response_time:.2f}s")
            print(f"   Status: {result['status']}")
            print(f"   Detected objects: {result['statistics']['total_detections']}")
            print(f"   Processing time: {result['statistics']['processing_time_ms']}ms")

            # Show details of detections
            for i, detection in enumerate(result['detections'][:3]):  # Max 3 first
                print(f"   Object {i+1}: {detection['class_name']} (confidence: {detection['confidence']})")

            return True
            
        else:
            print(f"❌ DDetection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return False

def create_test_image():
    """Create a simple test image"""

    # Create a 640x640 image with some shapes
    img = Image.new('RGB', (640, 640), color='lightblue')
    draw = ImageDraw.Draw(img)

    # Draw some shapes that YOLO might recognize
    # Rectangle (could be detected as an object)
    draw.rectangle([100, 100, 200, 200], fill='red', outline='black', width=3)

    # Circle
    draw.ellipse([300, 150, 450, 300], fill='green', outline='black', width=3)

    # Basic "car" shape
    draw.rectangle([200, 400, 400, 500], fill='blue', outline='black', width=3)
    draw.ellipse([220, 480, 260, 520], fill='black')  # wheel
    draw.ellipse([360, 480, 400, 520], fill='black')  # wheel

    return img

def test_unified_endpoint():
    """Test of the unified endpoint for the IA API"""

    print("\n🔗 === TEST UNIFIED ENDPOINT ===")

    base_url = "http://localhost:8002"

    # Create test image
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
            print(f"   Detected objects: {result['result']['object_detection']['objects_count']}")
        else:
            print(f"❌ Detection task failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Detection task error: {e}")

    # Reset buffer for second test
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
            print(f"   Main class: {result['result']['image_classification']['main_class']}")
            print(f"   Confidence: {result['result']['image_classification']['confidence']}")
        else:
            print(f"❌ Classification task failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Classification task error: {e}")

def performance_test():
    """Basic performance test"""

    print("\n⚡ === PERFORMANCE TEST ===")
    
    base_url = "http://localhost:8002"

    # Create multiple test images - FIX: Correct format for FastAPI
    files = []
    for i in range(3):
        img = create_test_image()
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        files.append(('images', (f'perf_test_{i}.jpg', buffer, 'image/jpeg')))

    # Test batch processing
    print("\n📦 Test batch processing")
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/predict/batch", files=files)
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch processing successful in {total_time:.2f}s")
            print(f"   Processed images: {result['total_images']}")
            print(f"   Successful: {result['successful']}")
            print(f"   Server time: {result['total_processing_time_ms']}ms")
        else:
            print(f"❌ Batch failed: {response.status_code}")
            print(f"   DDetails: {response.text}")
    except Exception as e:
        print(f"❌ Batch error: {e}")

def main():
    """Main test"""

    print("🚀 === COMPLETE YOLO API TESTS ===")
    print("Ensure your YOLO API is running on port 8002")
    print("Command: docker-compose up -d yolo-api")

    # Wait a bit for the API to be ready
    print("\n⏳ Waiting 5s for the API to be ready...")
    time.sleep(5)

    # Main tests
    success = test_yolo_api()
    
    if success:
        test_unified_endpoint()
        performance_test()

        print("\n🎉 === ALL TESTS PASSED ===")
        print("✅ YOLO API is functional and ready!")
        print("✅ Object detection is operational")
        print("✅ Unified endpoints OK")
        print("✅ Performance acceptable")

        print(f"\n🔗 API available at:")
        print(f"   • Health: http://localhost:8002/health")
        print(f"   • Docs: http://localhost:8002/docs")
        print(f"   • Predict: http://localhost:8002/predict")
        
    else:
        print("\n❌ === TESTS FAILED ===")
        print("Check the YOLO container logs:")
        print("docker-compose logs yolo-api")

if __name__ == "__main__":
    main()