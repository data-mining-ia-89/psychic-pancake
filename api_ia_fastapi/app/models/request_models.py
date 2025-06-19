# app/models/request_models.py

from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str
