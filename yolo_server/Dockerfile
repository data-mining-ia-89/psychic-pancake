FROM python:3.10-slim

WORKDIR /app

# Installer seulement FastAPI et uvicorn
RUN pip install fastapi uvicorn python-multipart

# Copier seulement main.py
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]