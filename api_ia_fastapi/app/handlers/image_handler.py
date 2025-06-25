# app/handlers/image_handler.py

import requests

def process_image(filename: str, image_bytes: bytes) -> dict:
    """
    Sends the image to the local YOLO API for object detection.
    """
    try:
        files = {"image": (filename, image_bytes, "image/jpeg")}
        response = requests.post("http://localhost:8000/predict", files=files)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
