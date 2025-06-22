from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from yolo_utils import detect_objects

app = FastAPI(title="YOLOv8 API")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    content = await image.read()
    try:
        result = detect_objects(content)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
