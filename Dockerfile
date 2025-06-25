# Dockerfile (at the root of the IA project)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/logs

# Expose the port
EXPOSE 8001

# Start command with the correct path
CMD ["uvicorn", "api_ia_fastapi.app.main:app", "--host", "0.0.0.0", "--port", "8001"]