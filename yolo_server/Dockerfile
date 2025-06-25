# yolo_server/Dockerfile - VERSION WITH REAL YOLO

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    libgtk-3-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for models
RUN mkdir -p /app/models

# Copy application code
COPY main.py .

# Download YOLOv8n at build time (optional, can also be done at runtime)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || echo "YOLO download skipped, will download at runtime"

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Startup command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]