from fastapi import FastAPI, UploadFile, File
from typing import Optional
from fastapi.responses import JSONResponse
from app.models.request_models import TextRequest
from app.handlers.text_handler import process_text
from app.handlers.image_handler import process_image

app = FastAPI(title="AI Project - Text and Image Analysis")

# ----------- Unified endpoint -----------
@app.post("/analyze")
async def analyze(text: Optional[str] = None, image: Optional[UploadFile] = File(None)):
    if text:
        result = process_text(text)
        return JSONResponse(content={"type": "text", "result": result})

    elif image:
        content = await image.read()
        result = process_image(image.filename, content)
        return JSONResponse(content={"type": "image", "result": result})

    return JSONResponse(status_code=400, content={"error": "No text or image provided"})

# ----------- Run locally -----------
# uvicorn app.main:app --reload
