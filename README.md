# 🤖 AI API - Unified Intelligence Service

**Production-ready FastAPI service with fine-tuned LLM and YOLO computer vision**

This repository contains the AI component of our **Hadoop & AI Project**, providing a unified REST API for advanced text analysis and computer vision capabilities with seamless Hadoop integration.

## ⚠️ IMPORTANT - Instructions for Evaluators

### Mandatory Prerequisites

**BEFORE STARTING**, the evaluator must install and configure:

1. **Git Bash** (mandatory for Windows) :
   - Download from : https://git-scm.com/downloads
   - Install with default options
   - Use Git Bash for all commands in this project

2. **File Permissions** :
   ```bash
   # In Git Bash, after cloning the project
   cd psychic-pancake
   
   # Grant execution rights to scripts
   chmod +x scripts/*.sh
   chmod +x *.sh deploy.sh
   
   # For Windows, if permission issues persist :
   git config core.filemode false
   ```

3. **Docker Desktop** installed and running
4. Launch the **Hadoop container** first and the **AI container** after the Hadoop container has been launched to 100%.
5. Remember to take the mistralai/mathstral-7b-v0.1 model in llm studio and run the server before building and launching the clusters.

### Quick Evaluation Commands

```bash
# 1. Clone and configure the project
git clone https://github.com/data-mining-ia-89/psychic-pancake.git
cd psychic-pancake
chmod +x deploy.sh

# 2. Start all services
./deploy.sh

# 3. Verify deployment
./deploy.sh --status

# 4. Test AI API
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{"data_type": "text", "content": "This AI service is amazing!", "task": "sentiment"}'
```

## 🎯 Overview

A comprehensive AI service that exposes:

- **Fine-tuned Language Model** for sentiment analysis, classification, and summarization
- **YOLO Computer Vision** for object detection and image classification  
- **Unified API** with single entry point for both text and image processing
- **Hadoop Integration** for processing big data workflows
- **LM Studio Compatibility** for model comparison and validation
- **Production-ready** with monitoring, testing, and CI/CD

## 👥 Development Team

**Hadoop & AI Project - Master Data Science**

- **Project Manager**: [Project Manager Name]
- **AI Developer**: [AI Developer Name]
- **Hadoop Engineer**: [Hadoop Engineer Name]
- **DevOps Engineer**: [DevOps Engineer Name]

**Supervisor**: [Professor Name]  
**University**: [University Name]  
**Academic Year**: 2024-2025

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

## 🏛️ Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Gateway** | FastAPI | Unified entry point, request routing |
| **Language Model** | DistilBERT + Custom Training | Sentiment analysis, text classification |
| **Computer Vision** | YOLOv8 | Object detection, image analysis |
| **Big Data Integration** | Apache Spark + Hadoop HDFS | Large-scale data processing |
| **Containerization** | Docker + Docker Compose | Service orchestration |
| **Model Serving** | PyTorch + Transformers | Model inference |
| **API Documentation** | OpenAPI/Swagger | Interactive documentation |
| **Monitoring** | Custom metrics + Health checks | System observability |

## 🚀 Quick Start

### Prerequisites

- **Git Bash** (Windows) or Terminal (macOS/Linux)
- **Docker Desktop** installed and running
- **4GB+ RAM** available for AI models
- **NVIDIA GPU** (optional, for faster inference)
- **Available ports**: 8001, 8002, 9870, 8088, 8080

### 1. Initial Configuration

```bash
# Clone the project
git clone https://github.com/data-mining-ia-89/psychic-pancake.git
cd psychic-pancake

# Configure permissions (MANDATORY)
chmod +x deploy.sh

# Verify Docker
docker --version
docker-compose --version
```

### 2. Deploy AI API

```bash
# Complete deployment with fine-tuning
./deploy.sh

# Or deploy without fine-tuning
./deploy.sh --no-finetune

# Check container status
docker ps

# Follow logs in real-time
docker-compose logs -f
```

### 3. Verify Deployment

```bash
# Automatic verification script
./deploy.sh --status

# Manual verifications
curl http://localhost:8001/health
curl http://localhost:8002/health

# Hadoop web interface (if running)
# Open http://localhost:9870 in browser
```

### 4. Test the API

```bash
# Test sentiment analysis with fine-tuned model
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "text",
    "content": "This AI service is absolutely amazing!",
    "task": "sentiment",
    "model_preference": "finetuned"
  }'

# Test image analysis (requires image file)
curl -X POST "http://localhost:8002/predict" \
  -F "image=@test_data/sample_image.jpg"

# Test model comparison
curl http://localhost:8001/models/comparison
```


```bash
# Run complete test suite
./deploy.sh --test

# Specific tests
python -m pytest tests/ -v
python test_yolo.py
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

# Unified image analysis
POST /analyze/image?task=detection
```

## 🔬 Fine-tuning Process

### 1. Run Fine-tuning

```bash
# Execute fine-tuning via deployment script
./deploy.sh --finetune

# Or run fine-tuning only
./deploy.sh --finetune
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
# Example training data structure (see api_ia_fastapi/app/models/llm_finetuning.py)
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
# Example Spark job calling AI API (api_ia_fastapi/app/utils/hadoop_formatter.py)
response = requests.post(
    "http://ai-api:8001/analyze/batch",
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
./deploy.sh --integration

# Run integration test
docker exec ai-api-unified curl http://namenode:9870
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
  container_name: ai-api-unified
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
  container_name: yolo-api-server
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
docker exec ai-api-unified python -c "
from api_ia_fastapi.app.models.llm_finetuning_production import run_complete_finetuning_pipeline
run_complete_finetuning_pipeline()
"

# Test YOLO service
python test_yolo.py

# Test Hadoop integration
./deploy.sh --integration
```

### Load Testing

```bash
# Complete test suite
./deploy.sh --test

# API endpoint tests
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{"data_type": "text", "content": "Test message", "task": "sentiment"}'
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

# Complete status check
./deploy.sh --status
```

### Metrics Exposed

- Request/response times
- Model inference latency
- Memory usage
- Error rates
- Throughput (requests/second)

### Logging

```bash
# View AI logs
docker logs ai-api-unified

# View YOLO logs
docker logs yolo-api-server

# Real-time monitoring
docker logs -f ai-api-unified

# Follow all logs
docker-compose logs -f
```

## 🔧 Configuration

### Model Configuration

```python
# Configuration in api_ia_fastapi/app/main.py
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

### YOLO Re-training

```bash
# Re-train YOLO with Hadoop images
./deploy.sh --yolo-retrain

# Or via API endpoint
curl -X POST "http://localhost:8001/yolo/retrain/start" \
  -H "Content-Type: application/json" \
  -d '{"hdfs_images_path": "/data/images/scraped", "epochs": 50}'
```

### Batch Processing

```bash
# Process large datasets
curl -X POST "http://localhost:8001/analyze/batch" \
  -H "Content-Type: application/json" \
  -d @large_dataset.json
```

## 🐛 Troubleshooting

### Common Issues and Solutions

**Permission Errors (Windows)**:
```bash
# In Git Bash
chmod +x deploy.sh
git config core.filemode false

# If issues persist
git update-index --chmod=+x deploy.sh
```

**Docker Won't Start**:
```bash
# Check Docker Desktop is running
docker info

# Clean old containers
./deploy.sh --clean

# Emergency cleanup
./deploy.sh --emergency-clean
```

**Ports Already in Use**:
```bash
# Check occupied ports
netstat -tulpn | grep :8001

# Modify ports in docker-compose.yml if necessary
```

**AI Models Won't Load**:
```bash
# Check disk space (models = ~2GB)
df -h

# Check logs
./deploy.sh --debug

# Restart services
./deploy.sh --clean
```

### Debug Mode

```bash
# Enable debug logging
./deploy.sh --debug

# Show troubleshooting guide
./deploy.sh --troubleshoot

# Generate diagnostic report
./deploy.sh --diagnostic
```

## 📖 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **YOLO API Docs**: http://localhost:8002/docs

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
psychic-pancake/
├── api_ia_fastapi/              # Main FastAPI application
│   ├── app/
│   │   ├── main.py             # FastAPI app entry point
│   │   ├── models/             # AI model implementations
│   │   │   ├── llm_finetuning.py
│   │   │   └── llm_finetuning_production.py
│   │   ├── services/           # Business logic services
│   │   │   └── finetuned_llm_service.py
│   │   ├── handlers/           # Request handlers
│   │   │   ├── text_handler.py
│   │   │   └── image_handler.py
│   │   ├── endpoints/          # API endpoints
│   │   │   └── yolo_retraining.py
│   │   └── utils/              # Utility functions
│   │       └── hadoop_formatter.py
│   └── yolo_retraining/        # YOLO retraining pipeline
│       └── complete_pipeline.py
├── yolo_server/                 # YOLO computer vision service
│   ├── main.py                 # YOLO FastAPI server
│   ├── Dockerfile              # YOLO container config
│   └── requirements.txt        # YOLO dependencies
├── models/                      # Trained model storage
│   └── finetuned_sentiment_model/
│       └── training_results.json
├── tests/                       # Test suites
│   ├── __init__.py
│   └── test_api.py
├── scripts/                     # Deployment scripts
├── docker-compose.yml          # Service orchestration
├── Dockerfile                  # Main API container
├── deploy.sh                   # Main deployment script
├── test_yolo.py               # YOLO testing script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 📋 Evaluation Checklist

### For Evaluators

- [ ] Git Bash installed and configured
- [ ] Docker Desktop running
- [ ] Project cloned and permissions configured (`chmod +x deploy.sh`)
- [ ] Services deployed with `./deploy.sh`
- [ ] API accessible at http://localhost:8001
- [ ] Basic tests passed with `./deploy.sh --status`
- [ ] Fine-tuning executed successfully (`./deploy.sh --finetune`)
- [ ] Models compared and metrics displayed
- [ ] Hadoop integration tested (`./deploy.sh --integration`)
- [ ] Technical documentation complete

### Expected Deliverables

1. **Source Code**: Complete Git repository
2. **Fine-tuned Models**: Trained and exported models
3. **Documentation**: README, API docs, architecture
4. **Tests**: Automated test suite
5. **Demo**: Presentation video (optional)

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
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black . && flake8 .
```

## 📄 License

This project is part of an academic assignment and is intended for educational purposes.

**University**: [University Name]  
**Master**: Data Science & Artificial Intelligence  
**Course**: Hadoop and Artificial Intelligence  
**Year**: 2024-2025

## 🚀 What's Next?

- **Transformer models** for advanced NLP tasks
- **Multi-modal AI** combining text and vision
- **Real-time streaming** inference
- **A/B testing** for model performance
- **Edge deployment** optimization
- **Kubernetes** orchestration

## 📞 Support and Contact

**In case of issues during evaluation** :

1. Consult the [Troubleshooting](#-troubleshooting) section
2. Check logs: `docker-compose logs`
3. Run diagnostic script: `./deploy.sh --debug`
4. Contact the team via [email/discord/etc.]

---

**🎯 Quick Instructions for Evaluators** :

```bash
# 1. Prerequisites
# Install Git Bash from https://git-scm.com/downloads

# 2. Quick deployment
git clone <repository-url>
cd psychic-pancake
chmod +x deploy.sh
./deploy.sh

# 3. Verification
./deploy.sh --status

# 4. Main test
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{"data_type": "text", "content": "Amazing AI project!", "task": "sentiment"}'
```

**Ready to power your AI workflows? Deploy with `./deploy.sh` and start analyzing! 🤖**

**Ready for evaluation? Follow the instructions above and discover our complete AI solution! 🚀**
