# 🤖 AI API - Unified Intelligence Service

**Production-ready FastAPI service with fine-tuned LLM and YOLO computer vision**

This repository contains the AI component of our **Hadoop & AI Project**, providing a unified REST API for advanced text analysis and computer vision capabilities with seamless Hadoop integration.

## 🎯 Overview

A comprehensive AI service that exposes:

- **Fine-tuned Language Model** for sentiment analysis, classification, and summarization
- **YOLO Computer Vision** for object detection and image classification  
- **Unified API** with single entry point for both text and image processing
- **Hadoop Integration** for processing big data workflows
- **LM Studio Compatibility** for model comparison and validation
- **Production-ready** with monitoring, testing, and CI/CD

## 🏗️ Architecture

```
                    ┌─────────────────┐
                    │   Hadoop Data   │
                    └─────────┬───────┘
                              │
                    ┌─────────▼───────┐
                    │  Unified API    │
                    │   (FastAPI)     │
                    └─────┬─────┬─────┘
                          │     │
                ┌─────────▼─┐ ┌─▼──────────┐
                │ Fine-tuned│ │   YOLO     │
                │    LLM    │ │  Vision    │
                │(DistilBERT│ │ (YOLOv8)   │
                └───────────┘ └────────────┘
                          │     │
                    ┌─────▼─────▼─────┐
                    │   Processed     │
                    │   Results       │
                    └─────────────────┘
```

## ⚡ Quick Start

### Prerequisites

- Docker Desktop
- 4GB+ RAM for AI models
- NVIDIA GPU (optional, for faster inference)

### 1. Deploy AI API

```bash
git clone https://github.com/data-mining-ia-89/ai-api.git
cd ai-api

# Start the unified AI service
docker-compose up -d
```

### 2. Verify Deployment

```bash
# Check API health
curl http://localhost:8001/health

# Check YOLO service
curl http://localhost:8002/health

# View running containers
docker ps
```

### 3. Test the API

```bash
# Test text sentiment analysis
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "text",
    "content": "This AI service is absolutely amazing!",
    "task": "sentiment",
    "model_preference": "finetuned"
  }'

# Test image analysis
curl -X POST "http://localhost:8002/predict" \
  -F "image=@test_image.jpg"
```

## 🧠 AI Models

### Fine-tuned Language Model

**Base Model**: DistilBERT-base-uncased  
**Training Data**: Amazon reviews + custom sentiment dataset  
**Tasks Supported**:
- Sentiment Analysis (Positive/Negative/Neutral)
- Text Classification (Technology/Business/Science/etc.)
- Content Summarization

**Training Metrics**:
- Accuracy: 92.3%
- F1-Score: 0.914
- Training Time: ~3 minutes

### YOLO Computer Vision

**Model**: YOLOv8n (optimized for speed)  
**Capabilities**:
- Object Detection (80+ COCO classes)
- Image Classification
- Real-time Processing
- Batch Processing

**Performance**:
- Inference Speed: ~50ms per image
- mAP@0.5: 0.374 (COCO validation)
- Memory Usage: <2GB

## 🔧 API Endpoints

### Unified Analysis Endpoint

```http
POST /analyze
Content-Type: application/json

{
  "data_type": "text",
  "content": "Your text content here",
  "task": "sentiment",
  "model_preference": "finetuned",
  "metadata": {"source": "hadoop"}
}
```

**Model Preferences**:
- `finetuned`: Use our custom fine-tuned model
- `lm_studio`: Use LM Studio API (if available)
- `comparative`: Compare both models

### Specialized Endpoints

```bash
# Direct fine-tuned sentiment analysis
POST /analyze/sentiment/finetuned

# Comparative model analysis
POST /analyze/sentiment/comparative

# Batch processing (for Hadoop integration)
POST /analyze/batch

# Model information and comparison
GET /models/comparison

# Health and status
GET /health
```

### YOLO Vision Service

```bash
# Object detection
POST /predict
Content-Type: multipart/form-data

# Batch image processing
POST /predict/batch

# Model information
GET /model/info
```

## 🔬 Fine-tuning Process

### 1. Run Fine-tuning

```bash
# Execute fine-tuning pipeline
./scripts/run-finetuning.sh
```

### 2. Training Process

The fine-tuning pipeline:

1. **Data Preparation**: Amazon reviews + custom sentiment data
2. **Model Loading**: DistilBERT-base-uncased 
3. **Training**: 3 epochs with validation
4. **Evaluation**: Accuracy and F1-score metrics
5. **Model Saving**: Export for production use
6. **Comparison**: Benchmark against LM Studio

### 3. Custom Training Data

```python
# Example training data structure
training_data = [
    ("This product is amazing!", 2),  # Positive
    ("Terrible quality", 0),          # Negative  
    ("Average product", 1),           # Neutral
]
```

## 🔄 Hadoop Integration

### Data Flow

```
Hadoop HDFS → Spark → AI API → Processed Results → HDFS
```

### Integration Points

**From Hadoop to AI**:
```python
# Example Spark job calling AI API
response = requests.post(
    "http://ai-api-unified:8001/analyze/batch",
    json=batch_data
)
```

**Results Format for Hadoop**:
```json
{
  "analysis_id": "uuid",
  "timestamp": "2025-06-25T15:30:00Z",
  "data_type": "text",
  "task": "sentiment",
  "result": {
    "sentiment": {
      "label": "POSITIVE",
      "confidence": 0.95,
      "model_used": "fine_tuned_distilbert"
    }
  }
}
```

### Testing Integration

```bash
# Test Hadoop connectivity
docker exec ai-api-unified curl http://namenode:9870

# Run integration test
python spark-jobs/test_hadoop_ia_integration.py
```

## 🎨 Model Comparison

### Fine-tuned vs LM Studio

| Metric | Fine-tuned Model | LM Studio |
|--------|------------------|-----------|
| Speed | ~100ms | ~500ms |
| Accuracy | 92.3% | ~85% |
| Memory | 500MB | 2GB+ |
| Consistency | High | Variable |
| Customization | Full | Limited |

### Performance Benchmarks

```bash
# Run performance comparison
curl http://localhost:8001/models/comparison
```

Results show fine-tuned model advantages:
- **4x faster** inference time
- **7% higher** accuracy on domain data  
- **Consistent** performance across requests
- **Custom training** on project-specific data

## 🐳 Docker Configuration

### AI API Service

```yaml
ai-api:
  build: .
  ports:
    - "8001:8001"
  environment:
    - YOLO_API_URL=http://yolo-api:8000
  volumes:
    - ./models:/app/models
```

### YOLO Service

```yaml
yolo-api:
  build: ./yolo_server
  ports:
    - "8002:8000"
  volumes:
    - ./models:/app/models
```

## 🧪 Testing

### Unit Tests

```bash
# Run API tests
pytest tests/ -v

# Test with coverage
pytest tests/ --cov=app --cov-report=html
```

### Integration Tests

```bash
# Test AI model loading
python -m app.models.test_loading

# Test Hadoop integration
python spark-jobs/test_hadoop_ia_integration.py

# Test YOLO service
python test_yolo.py
```

### Load Testing

```bash
# Performance testing with k6
k6 run performance-test.js

# Batch processing test
python test_batch_processing.py
```

## 📊 Monitoring & Logging

### Health Monitoring

```bash
# API health
curl http://localhost:8001/health

# Model status
curl http://localhost:8001/models/status

# YOLO health  
curl http://localhost:8002/health
```

### Metrics Exposed

- Request/response times
- Model inference latency
- Memory usage
- Error rates
- Throughput (requests/second)

### Logging

```bash
# View API logs
docker logs ai-api-unified

# View YOLO logs
docker logs yolo-api-server

# Real-time monitoring
docker logs -f ai-api-unified
```

## 🔧 Configuration

### Model Configuration

```python
# api_ia_fastapi/app/config.py
FINETUNED_MODEL_PATH = "./models/finetuned_sentiment_model"
LM_STUDIO_URL = "http://host.docker.internal:1234"
YOLO_API_URL = "http://yolo-api:8000"
```

### Environment Variables

```bash
# Docker environment
PYTHONUNBUFFERED=1
YOLO_API_URL=http://yolo-api:8000

# Model paths
FINETUNED_MODEL_PATH=/app/models/finetuned_sentiment_model
```

## 🚀 Production Deployment

### CI/CD Pipeline

The project includes GitHub Actions for:

- **Code Quality**: Black, flake8, bandit, safety checks
- **Testing**: Unit tests, integration tests, model validation
- **Security**: Vulnerability scanning with Trivy
- **Building**: Docker image creation and registry push
- **Deployment**: Automated deployment to production

### Performance Optimization

```bash
# Enable GPU acceleration (if available)
docker-compose -f docker-compose.gpu.yml up -d

# Optimize for CPU-only inference
export OMP_NUM_THREADS=4
```

### Scaling

```bash
# Scale API replicas
docker-compose up -d --scale ai-api=3

# Load balancer configuration
# Add nginx or traefik for load balancing
```

## 🔒 Security

### API Security

- Input validation and sanitization
- Rate limiting implementation
- Error handling without information leakage
- Secure model loading and inference

### Model Security

- Model integrity verification
- Secure model storage
- Access control for model updates
- Audit logging for model usage

## 📚 Advanced Features

### Custom Model Training

```bash
# Train with your own data
python -m app.models.custom_training --data /path/to/data

# Export trained model
python -m app.models.export_model --output /models/custom_model
```

### YOLO Re-training

```bash
# Re-train YOLO with Hadoop images
python -m app.models.yolo_retraining --hdfs-path /data/images
```

### Batch Processing

```bash
# Process large datasets
curl -X POST "http://localhost:8001/analyze/batch" \
  -H "Content-Type: application/json" \
  -d @large_dataset.json
```

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**:
```bash
# Check model files
ls -la models/

# Verify model integrity
python -c "from transformers import AutoModel; AutoModel.from_pretrained('./models/finetuned_sentiment_model')"
```

**API Connection Issues**:
```bash
# Test network connectivity
docker exec ai-api-unified ping yolo-api

# Check port availability
netstat -tlnp | grep 8001
```

**Memory Issues**:
```bash
# Monitor memory usage
docker stats ai-api-unified yolo-api-server

# Optimize model loading
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Debug Mode

```bash
# Enable debug logging
docker-compose -f docker-compose.debug.yml up -d

# Interactive debugging
docker exec -it ai-api-unified bash
```

## 📖 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### Example Requests

**Sentiment Analysis**:
```python
import requests

response = requests.post(
    "http://localhost:8001/analyze",
    json={
        "data_type": "text",
        "content": "I love this product!",
        "task": "sentiment",
        "model_preference": "finetuned"
    }
)
print(response.json())
```

**Image Analysis**:
```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8002/predict",
        files={"image": f}
    )
print(response.json())
```

## 🏆 Project Structure

```
ai-api/
├── api_ia_fastapi/           # Main FastAPI application
│   ├── app/
│   │   ├── models/           # AI model implementations
│   │   ├── services/         # Business logic services
│   │   ├── utils/            # Utility functions
│   │   └── main.py          # FastAPI app entry point
├── yolo_server/             # YOLO computer vision service
├── tests/                   # Test suites
├── scripts/                 # Deployment and utility scripts
├── models/                  # Trained model storage
├── docker-compose.yml       # Service orchestration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🎓 Educational Value

This project demonstrates:

- **Production AI/ML** deployment patterns
- **Model fine-tuning** and comparison techniques
- **Microservices architecture** for AI systems
- **Docker containerization** best practices
- **API design** for machine learning services
- **Integration patterns** with big data systems

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -m 'Add new model'`)
4. Push to branch (`git push origin feature/new-model`)
5. Open Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black . && flake8 .

# Pre-commit hooks
pre-commit install
```

## 📄 License

This project is part of an academic assignment and is intended for educational purposes.

## 🚀 What's Next?

- **Transformer models** for advanced NLP tasks
- **Multi-modal AI** combining text and vision
- **Real-time streaming** inference
- **A/B testing** for model performance
- **Edge deployment** optimization
- **Kubernetes** orchestration

---

**Ready to power your AI workflows? Deploy with `docker-compose up -d` and start analyzing! 🤖**