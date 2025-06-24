from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="YOLO API Simple")

@app.get("/")
def read_root():
    return {"message": "YOLO API is running", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "yolo-api"}

@app.post("/predict")
async def predict_simple(image: UploadFile = File(...)):
    """
    Endpoint simplifié pour test - remplace YOLO par une réponse mockée
    """
    try:
        # Lire l'image
        content = await image.read()
        
        # Mock response pour test (remplacera YOLO plus tard)
        mock_detections = [
            {
                "class": 0,
                "label": "person", 
                "confidence": 0.85,
                "bbox": [100, 150, 200, 300]
            },
            {
                "class": 2,
                "label": "car",
                "confidence": 0.72,
                "bbox": [300, 100, 500, 250]
            }
        ]
        
        return JSONResponse(content={
            "detections": mock_detections,
            "image_size": len(content),
            "filename": image.filename,
            "status": "success",
            "note": "Mock response - YOLO model will be integrated later"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)