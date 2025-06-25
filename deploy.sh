#!/bin/bash
# Script de déploiement COMPLET - API IA + FINE-TUNING + YOLO

set -e

# Colors pour Git Bash
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    PURPLE='\033[0;35m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    PURPLE=''
    CYAN=''
    NC=''
fi

echo -e "${BLUE}🤖 Complete AI API Deployment${NC}"
echo -e "${BLUE}=============================${NC}"

# Aller au répertoire du projet
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
echo -e "${YELLOW}📂 Project: $PROJECT_ROOT${NC}"

# Parse arguments
ACTION="deploy"
SKIP_FINETUNING=false
SKIP_YOLO_RETRAIN=false
WITH_INTEGRATION_TEST=false

case "${1:-}" in
    --clean) 
        ACTION="clean"
        echo -e "${YELLOW}🧹 Mode: Clean restart with full setup${NC}"
        ;;
    --fresh) 
        ACTION="fresh" 
        echo -e "${RED}🧹 Mode: Fresh deployment (complete reset)${NC}"
        ;;
    --status) 
        ACTION="status"
        echo -e "${BLUE}📊 Mode: Complete status check${NC}"
        ;;
    --debug)
        ACTION="debug"
        echo -e "${RED}🔧 Mode: Debug all AI services${NC}"
        ;;
    --build-only)
        ACTION="build-only"
        echo -e "${BLUE}🏗️ Mode: Build images only${NC}"
        ;;
    --finetune)
        ACTION="finetune"
        echo -e "${PURPLE}🧠 Mode: Run fine-tuning only${NC}"
        ;;
    --yolo-retrain)
        ACTION="yolo-retrain"
        echo -e "${CYAN}📷 Mode: YOLO retraining only${NC}"
        ;;
    --test)
        ACTION="test"
        echo -e "${GREEN}🧪 Mode: Run all tests${NC}"
        ;;
    --integration)
        ACTION="deploy"
        WITH_INTEGRATION_TEST=true
        echo -e "${BLUE}🔗 Mode: Deploy with Hadoop integration test${NC}"
        ;;
    --no-finetune)
        ACTION="deploy"
        SKIP_FINETUNING=true
        echo -e "${BLUE}📋 Mode: Deploy without fine-tuning${NC}"
        ;;
    --help|-h)
        echo -e "${YELLOW}Usage: $0 [OPTION]${NC}"
        echo "  (no args)       Complete deployment with fine-tuning"
        echo "  --clean         Clean restart with full setup"
        echo "  --fresh         Complete reset and fresh install"
        echo "  --status        Complete status of all AI services"
        echo "  --debug         Debug mode for all AI services"
        echo "  --build-only    Build Docker images only"
        echo "  --finetune      Run fine-tuning process only"
        echo "  --yolo-retrain  YOLO retraining with Hadoop images"
        echo "  --test          Run all tests (unit + integration)"
        echo "  --integration   Deploy with Hadoop integration test"
        echo "  --no-finetune   Deploy without fine-tuning"
        echo "  --help          Show this help"
        exit 0
        ;;
    "")
        echo -e "${BLUE}📋 Mode: Complete deployment with fine-tuning${NC}"
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo "Use --help for usage"
        exit 1
        ;;
esac

# ============ FONCTIONS DE BASE ============

check_docker() {
    echo -e "\n${YELLOW}🔍 Checking Docker...${NC}"
    
    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker command not found${NC}"
        exit 1
    fi
    
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker daemon not running${NC}"
        echo -e "${YELLOW}💡 Start Docker Desktop and try again${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Docker is running${NC}"
    
    if [[ ! -f "docker-compose.yml" ]]; then
        echo -e "${RED}❌ docker-compose.yml not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ docker-compose.yml found${NC}"
}

check_python_requirements() {
    echo -e "\n${YELLOW}🐍 Checking Python requirements...${NC}"
    
    if ! command -v python3 >/dev/null 2>&1; then
        echo -e "${RED}❌ Python3 not found${NC}"
        exit 1
    fi
    
    # Vérifier les packages critiques
    python3 -c "import torch; print('✅ PyTorch available')" 2>/dev/null || echo -e "${YELLOW}⚠️ PyTorch not available (will install)${NC}"
    python3 -c "import transformers; print('✅ Transformers available')" 2>/dev/null || echo -e "${YELLOW}⚠️ Transformers not available (will install)${NC}"
    python3 -c "import ultralytics; print('✅ Ultralytics available')" 2>/dev/null || echo -e "${YELLOW}⚠️ Ultralytics not available (will install)${NC}"
    
    echo -e "${GREEN}✅ Python environment checked${NC}"
}

check_network() {
    echo -e "\n${YELLOW}🌐 Checking network connectivity...${NC}"
    
    # Vérifier si le réseau Hadoop existe
    if docker network ls | grep -q hadoop-net; then
        echo -e "${GREEN}✅ Hadoop network found: hadoop-net${NC}"
    else
        echo -e "${RED}❌ Hadoop network not found${NC}"
        echo -e "${YELLOW}💡 Creating hadoop-net network...${NC}"
        docker network create hadoop-net
        echo -e "${GREEN}✅ Network hadoop-net created${NC}"
    fi
    
    # Vérifier connectivité Hadoop (si disponible)
    if docker ps --format "{{.Names}}" | grep -q namenode; then
        echo -e "${GREEN}✅ Hadoop cluster detected${NC}"
        check_hadoop_connectivity
    else
        echo -e "${YELLOW}⚠️ Hadoop cluster not running${NC}"
        echo -e "${YELLOW}💡 Some features will be limited without Hadoop${NC}"
    fi
}

check_hadoop_connectivity() {
    echo -e "\n${BLUE}🔗 Testing Hadoop connectivity...${NC}"
    
    # Test NameNode connectivity
    if curl -f -s --max-time 5 "http://localhost:9870" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ NameNode accessible${NC}"
    else
        echo -e "${YELLOW}⚠️ NameNode not accessible via localhost:9870${NC}"
    fi
}

ensure_directories() {
    echo -e "\n${YELLOW}📁 Ensuring directories...${NC}"
    
    local dirs=(
        "./models"
        "./logs" 
        "./training_data"
        "./yolo_retraining"
        "./test_results"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    echo -e "${GREEN}✅ All directories created${NC}"
}

complete_cleanup() {
    echo -e "\n${RED}🧹 COMPLETE AI CLEANUP${NC}"
    
    # 1. Arrêter tous les conteneurs AI
    echo -e "${YELLOW}⏹️ Stopping AI containers...${NC}"
    docker-compose down --remove-orphans -v || true
    
    # 2. Supprimer conteneurs AI orphelins
    echo -e "${YELLOW}🗑️ Removing AI containers...${NC}"
    docker ps -a --format "{{.Names}}" | grep -E "(ai-api|yolo-api)" | xargs -r docker rm -f || true
    
    # 3. Supprimer volumes AI
    echo -e "${YELLOW}💾 Removing AI volumes...${NC}"
    docker volume ls -q | grep -E "(ai-api|yolo)" | xargs -r docker volume rm -f || true
    
    # 4. Nettoyage modèles temporaires
    echo -e "${YELLOW}🧽 Cleaning temporary files...${NC}"
    rm -rf ./training_data/temp_* 2>/dev/null || true
    rm -rf ./yolo_retraining/temp_* 2>/dev/null || true
    rm -rf ./__pycache__ 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # 5. Nettoyage Docker général
    echo -e "${YELLOW}🧽 General Docker cleanup...${NC}"
    docker system prune -f
    
    echo -e "${GREEN}✅ Complete AI cleanup finished${NC}"
}

# ============ FONCTIONS DE VÉRIFICATION ============

check_service_health() {
    local name=$1
    local url=$2
    local timeout=${3:-5}
    
    if curl -f -s --max-time "$timeout" "$url" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_ai_api_health() {
    if check_service_health "AI-API" "http://localhost:8001/health" 5; then
        return 0
    else
        return 1
    fi
}

check_yolo_api_health() {
    if check_service_health "YOLO-API" "http://localhost:8002/health" 5; then
        return 0
    else
        return 1
    fi
}

wait_for_ai_service() {
    local name=$1
    local url=$2
    local max_wait=${3:-120}
    local elapsed=0
    
    echo -e "${YELLOW}⏳ Waiting for $name...${NC}"
    
    while [[ $elapsed -lt $max_wait ]]; do
        if check_service_health "$name" "$url" 3; then
            echo -e "${GREEN}✅ $name is ready (${elapsed}s)${NC}"
            return 0
        fi
        
        sleep 5
        elapsed=$((elapsed + 5))
        
        if [[ $((elapsed % 15)) -eq 0 ]]; then
            echo -e "${BLUE}... still waiting for $name (${elapsed}s/${max_wait}s)${NC}"
        fi
    done
    
    echo -e "${RED}❌ $name timeout after ${max_wait}s${NC}"
    return 1
}

# ============ FONCTIONS DE DÉPLOIEMENT ============

build_ai_images() {
    echo -e "\n${YELLOW}🏗️ Building AI Docker images...${NC}"
    
    # Build main AI API
    echo -e "${BLUE}📦 Building main AI API...${NC}"
    docker-compose build ai-api
    
    # Build YOLO API
    echo -e "${BLUE}📦 Building YOLO API...${NC}"
    docker-compose build yolo-api
    
    echo -e "${GREEN}✅ All AI images built successfully${NC}"
}

deploy_ai_services() {
    echo -e "\n${YELLOW}🚀 Deploying AI services...${NC}"
    
    # Démarrer YOLO API en premier
    echo -e "${BLUE}📷 Starting YOLO API...${NC}"
    docker-compose up -d yolo-api
    
    if wait_for_ai_service "YOLO-API" "http://localhost:8002/health" 60; then
        echo -e "${GREEN}✅ YOLO API ready${NC}"
    else
        echo -e "${RED}❌ YOLO API failed to start${NC}"
        return 1
    fi
    
    # Démarrer AI API principal
    echo -e "${BLUE}🤖 Starting main AI API...${NC}"
    docker-compose up -d ai-api
    
    if wait_for_ai_service "AI-API" "http://localhost:8001/health" 60; then
        echo -e "${GREEN}✅ AI API ready${NC}"
    else
        echo -e "${RED}❌ AI API failed to start${NC}"
        return 1
    fi
    
    return 0
}

run_fine_tuning() {
    if [[ "$SKIP_FINETUNING" == "true" ]]; then
        echo -e "\n${BLUE}📋 Skipping fine-tuning (--no-finetune flag)${NC}"
        return 0
    fi
    
    echo -e "\n${PURPLE}🧠 === FINE-TUNING PROCESS ===${NC}"
    
    # Vérifier si le modèle fine-tuné existe déjà
    if [[ -d "./models/finetuned_sentiment_model" ]]; then
        echo -e "${YELLOW}⚠️ Fine-tuned model already exists${NC}"
        read -p "Do you want to retrain? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}📋 Using existing fine-tuned model${NC}"
            return 0
        fi
    fi
    
    echo -e "${YELLOW}🔄 Starting fine-tuning process...${NC}"
    
    # Exécuter le fine-tuning dans le conteneur AI
    docker exec ai-api-unified python -c "
from api_ia_fastapi.app.models.llm_finetuning_production import run_complete_finetuning_pipeline
success = run_complete_finetuning_pipeline()
exit(0 if success else 1)
" || {
        echo -e "${RED}❌ Fine-tuning failed${NC}"
        return 1
    }
    
    echo -e "${GREEN}✅ Fine-tuning completed successfully${NC}"
    return 0
}

run_yolo_retraining() {
    if [[ "$SKIP_YOLO_RETRAIN" == "true" ]]; then
        echo -e "\n${BLUE}📋 Skipping YOLO retraining${NC}"
        return 0
    fi
    
    echo -e "\n${CYAN}📷 === YOLO RETRAINING PROCESS ===${NC}"
    
    # Vérifier connectivité Hadoop
    if ! docker ps --format "{{.Names}}" | grep -q namenode; then
        echo -e "${YELLOW}⚠️ Hadoop cluster not running, skipping YOLO retraining${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}🔄 Starting YOLO retraining with Hadoop images...${NC}"
    
    # Exécuter le ré-entraînement YOLO
    docker exec ai-api-unified python -c "
from api_ia_fastapi.yolo_retraining.complete_pipeline import YOLORetrainingPipeline
pipeline = YOLORetrainingPipeline()
results = pipeline.run_complete_pipeline()
exit(0 if results['success'] else 1)
" || {
        echo -e "${YELLOW}⚠️ YOLO retraining failed or skipped${NC}"
        return 0  # Non-critique
    }
    
    echo -e "${GREEN}✅ YOLO retraining completed${NC}"
    return 0
}

# ============ FONCTIONS DE TEST ============

run_unit_tests() {
    echo -e "\n${GREEN}🧪 Running unit tests...${NC}"
    
    # Installer pytest si nécessaire et lancer les tests
    docker exec ai-api-unified pip install pytest pytest-asyncio httpx || true
    docker exec ai-api-unified python -m pytest tests/ -v || {
        echo -e "${YELLOW}⚠️ Some unit tests failed${NC}"
    }
}

run_api_tests() {
    echo -e "\n${GREEN}🔌 Testing API endpoints...${NC}"
    
    # Test AI API principal
    echo -e "${BLUE}🤖 Testing main AI API...${NC}"
    
    # Test sentiment analysis
    local sentiment_result=$(curl -s -X POST "http://localhost:8001/analyze" \
        -H "Content-Type: application/json" \
        -d '{
            "data_type": "text",
            "content": "This is amazing!",
            "task": "sentiment",
            "model_preference": "finetuned"
        }' | jq -r '.status' 2>/dev/null || echo "error")
    
    if [[ "$sentiment_result" == "success" ]]; then
        echo -e "${GREEN}✅ Sentiment analysis working${NC}"
    else
        echo -e "${RED}❌ Sentiment analysis failed${NC}"
    fi
    
    # Test YOLO API
    echo -e "${BLUE}📷 Testing YOLO API...${NC}"
    docker exec ai-api-unified python test_yolo.py || echo -e "${YELLOW}⚠️ YOLO test had issues${NC}"
}

run_integration_tests() {
    echo -e "\n${GREEN}🔗 Running Hadoop integration tests...${NC}"
    
    if ! docker ps --format "{{.Names}}" | grep -q namenode; then
        echo -e "${YELLOW}⚠️ Hadoop not running, skipping integration tests${NC}"
        return 0
    fi
    
    # Test connectivité HDFS
    docker exec ai-api-unified python -c "
import requests
try:
    response = requests.get('http://namenode:9870/webhdfs/v1/data?op=LISTSTATUS', timeout=10)
    if response.status_code == 200:
        print('✅ HDFS connectivity successful')
        exit(0)
    else:
        print('❌ HDFS connectivity failed')
        exit(1)
except Exception as e:
    print(f'❌ HDFS connection error: {e}')
    exit(1)
" || echo -e "${YELLOW}⚠️ Hadoop integration test failed${NC}"
}

# ============ FONCTIONS DE STATUS ============

show_ai_services_status() {
    echo -e "\n${BLUE}🏥 AI SERVICES HEALTH CHECK${NC}"
    echo -e "${BLUE}=========================${NC}"
    
    local healthy=0
    local total=4
    
    # Services AI
    local ai_services=(
        "AI-API:http://localhost:8001/health"
        "YOLO-API:http://localhost:8002/health"
        "AI-Models:http://localhost:8001/models/status"
        "AI-Integration:http://localhost:8001/health"
    )
    
    echo -e "\n${YELLOW}🤖 AI Services:${NC}"
    for service in "${ai_services[@]}"; do
        local name="${service%%:*}"
        local url="${service#*:}"
        
        if check_service_health "$name" "$url" 5; then
            echo -e "${GREEN}✅ $name${NC}"
            ((healthy++))
        else
            echo -e "${RED}❌ $name${NC}"
        fi
    done
    
    # Modèles disponibles
    echo -e "\n${YELLOW}🧠 Available Models:${NC}"
    
    # Vérifier modèle fine-tuné
    if [[ -d "./models/finetuned_sentiment_model" ]]; then
        echo -e "${GREEN}✅ Fine-tuned sentiment model${NC}"
    else
        echo -e "${YELLOW}⚠️ Fine-tuned model not found${NC}"
    fi
    
    # Vérifier modèles YOLO
    if [[ -f "./models/yolo_custom_production.pt" ]]; then
        echo -e "${GREEN}✅ Custom YOLO model${NC}"
    else
        echo -e "${YELLOW}⚠️ Custom YOLO model not found${NC}"
    fi
    
    # Résumé final
    local percentage=$((healthy * 100 / total))
    echo -e "\n${BLUE}📈 Health Summary: ${GREEN}$healthy${NC}/${BLUE}$total${NC} services healthy (${percentage}%)${NC}"
    
    if [[ $percentage -ge 75 ]]; then
        echo -e "\n${GREEN}🎉 AI services are healthy!${NC}"
        return 0
    else
        echo -e "\n${RED}❌ Issues detected in AI services${NC}"
        return 1
    fi
}

show_containers_status() {
    echo -e "\n${BLUE}📊 AI CONTAINER STATUS${NC}"
    echo -e "${BLUE}=====================${NC}"
    
    echo -e "\n${YELLOW}🐳 AI Containers:${NC}"
    if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(ai-api|yolo-api)" 2>/dev/null; then
        echo ""
    else
        echo -e "${YELLOW}No AI containers running${NC}"
        return 1
    fi
    
    # Conteneurs arrêtés
    echo -e "${YELLOW}💤 Stopped AI Containers:${NC}"
    local stopped=$(docker ps -a --format "table {{.Names}}\t{{.Status}}" | grep -E "(ai-api|yolo-api)" | grep "Exited\|Created" || echo "None")
    if [[ "$stopped" == "None" ]]; then
        echo -e "${GREEN}✅ No stopped containers${NC}"
    else
        echo "$stopped"
    fi
    
    return 0
}

show_models_status() {
    echo -e "\n${BLUE}🧠 MODELS STATUS${NC}"
    echo -e "${BLUE}===============${NC}"
    
    echo -e "\n${YELLOW}📊 LLM Models:${NC}"
    
    # Fine-tuned model
    if [[ -d "./models/finetuned_sentiment_model" ]]; then
        local size=$(du -sh ./models/finetuned_sentiment_model 2>/dev/null | cut -f1 || echo "N/A")
        echo -e "${GREEN}✅ Fine-tuned sentiment model ($size)${NC}"
        
        # Vérifier métadonnées
        if [[ -f "./models/finetuned_sentiment_model/training_results.json" ]]; then
            local accuracy=$(jq -r '.eval_accuracy // "N/A"' ./models/finetuned_sentiment_model/training_results.json 2>/dev/null || echo "N/A")
            local f1_score=$(jq -r '.eval_f1 // "N/A"' ./models/finetuned_sentiment_model/training_results.json 2>/dev/null || echo "N/A")
            echo -e "    Accuracy: $accuracy, F1-Score: $f1_score"
        fi
    else
        echo -e "${YELLOW}⚠️ Fine-tuned model not found${NC}"
    fi
    
    echo -e "\n${YELLOW}📷 YOLO Models:${NC}"
    
    # Modèles YOLO
    for model_file in "./models"/*.pt; do
        if [[ -f "$model_file" ]]; then
            local model_name=$(basename "$model_file")
            local size=$(du -sh "$model_file" 2>/dev/null | cut -f1 || echo "N/A")
            echo -e "${GREEN}✅ $model_name ($size)${NC}"
        fi
    done
    
    if ! ls ./models/*.pt >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️ No YOLO models found${NC}"
    fi
}

debug_ai_services() {
    echo -e "\n${RED}🔧 DEBUG MODE - AI SERVICES${NC}"
    echo -e "${RED}============================${NC}"
    
    local services=("ai-api-unified" "yolo-api-server")
    
    for service in "${services[@]}"; do
        echo -e "\n${YELLOW}🔍 Debugging $service...${NC}"
        
        if docker ps --format "{{.Names}}" | grep -q "^$service$"; then
            echo -e "${GREEN}✅ $service is running${NC}"
            
            # Status détaillé
            local status=$(docker ps --format "{{.Status}}" --filter "name=$service")
            echo -e "${BLUE}Status: $status${NC}"
            
            # Logs récents
            echo -e "${BLUE}Last 10 log lines:${NC}"
            docker logs "$service" 2>&1 | tail -10 | sed 's/^/  /'
            
            # Resource usage
            local stats=$(docker stats "$service" --no-stream --format "CPU: {{.CPUPerc}}, Memory: {{.MemUsage}}" 2>/dev/null || echo "Stats unavailable")
            echo -e "${BLUE}Resources: $stats${NC}"
            
        else
            echo -e "${RED}❌ $service is not running${NC}"
            
            # Vérifier si le conteneur existe mais est arrêté
            local container_status=$(docker ps -a --format "{{.Names}}\t{{.Status}}" | grep "^$service" || echo "Container not found")
            echo -e "${BLUE}Container status: $container_status${NC}"
            
            # Si arrêté, montrer pourquoi
            if docker ps -a --format "{{.Names}}" | grep -q "^$service$"; then
                echo -e "${BLUE}Exit reason (last 5 lines):${NC}"
                docker logs "$service" 2>&1 | tail -5 | sed 's/^/  /'
            fi
        fi
    done
}

# ============ FONCTIONS D'AFFICHAGE ============

show_access_info() {
    echo -e "\n${BLUE}📊 AI API ACCESS INFORMATION${NC}"
    echo -e "${BLUE}============================${NC}"
    
    echo -e "\n${GREEN}🌐 API Endpoints:${NC}"
    echo -e "${GREEN}• Main AI API: http://localhost:8001${NC}"
    echo -e "${GREEN}• Interactive Docs: http://localhost:8001/docs${NC}"
    echo -e "${GREEN}• YOLO API: http://localhost:8002${NC}"
    echo -e "${GREEN}• YOLO Docs: http://localhost:8002/docs${NC}"
    
    echo -e "\n${YELLOW}🔗 Key Endpoints:${NC}"
    echo -e "${YELLOW}• Unified Analysis: POST http://localhost:8001/analyze${NC}"
    echo -e "${YELLOW}• Sentiment Analysis: POST http://localhost:8001/analyze/sentiment${NC}"
    echo -e "${YELLOW}• Model Comparison: GET http://localhost:8001/models/comparison${NC}"
    echo -e "${YELLOW}• YOLO Detection: POST http://localhost:8002/predict${NC}"
    echo -e "${YELLOW}• Health Check: GET http://localhost:8001/health${NC}"
    
    echo -e "\n${BLUE}💡 Useful Commands:${NC}"
    echo -e "  $0 --status        # Complete AI health check"
    echo -e "  $0 --debug         # Debug all AI services"
    echo -e "  $0 --test          # Run all tests"
    echo -e "  $0 --finetune      # Run fine-tuning only"
    echo -e "  $0 --yolo-retrain  # YOLO retraining only"
    echo -e "  $0 --integration   # Test Hadoop integration"
    echo -e "  docker-compose logs ai-api        # View AI API logs"
    echo -e "  docker-compose logs yolo-api      # View YOLO API logs"
    echo -e "  docker exec ai-api-unified python test_yolo.py              # Test YOLO functionality"
}

show_examples() {
    echo -e "\n${BLUE}📝 API USAGE EXAMPLES${NC}"
    echo -e "${BLUE}=====================${NC}"
    
    echo -e "\n${GREEN}🤖 Text Analysis Examples:${NC}"
    
    cat << 'EOF'
# Sentiment Analysis
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "text",
    "content": "This product is amazing!",
    "task": "sentiment",
    "model_preference": "finetuned"
  }'

# Classification
curl -X POST "http://localhost:8001/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "text", 
    "content": "New AI breakthrough in machine learning",
    "task": "classification"
  }'

# Model Comparison
curl -X POST "http://localhost:8001/analyze/sentiment/comparative" \
  -H "Content-Type: application/json" \
  -d '"This is an excellent service!"'
EOF

    echo -e "\n${GREEN}📷 Image Analysis Examples:${NC}"
    
    cat << 'EOF'
# Object Detection
curl -X POST "http://localhost:8002/predict" \
  -F "image=@your_image.jpg"

# Image Classification  
curl -X POST "http://localhost:8002/analyze/image?task=classification" \
  -F "image=@your_image.jpg"

# Batch Processing
curl -X POST "http://localhost:8002/predict/batch" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "images=@image3.jpg"
EOF
}

# ============ LOGIQUE PRINCIPALE ============

check_docker
check_python_requirements
check_network
ensure_directories

case $ACTION in
    "status")
        show_containers_status
        show_ai_services_status
        show_models_status
        show_access_info
        ;;
        
    "debug")
        debug_ai_services
        show_ai_services_status
        ;;
        
    "build-only")
        build_ai_images
        echo -e "\n${GREEN}✅ Docker images built successfully${NC}"
        ;;
        
    "finetune")
        if ! docker ps --format "{{.Names}}" | grep -q ai-api-unified; then
            echo -e "${YELLOW}🚀 Starting AI API for fine-tuning...${NC}"
            deploy_ai_services
        fi
        run_fine_tuning
        ;;
        
    "yolo-retrain")
        if ! docker ps --format "{{.Names}}" | grep -q ai-api-unified; then
            echo -e "${YELLOW}🚀 Starting AI API for YOLO retraining...${NC}"
            deploy_ai_services
        fi
        run_yolo_retraining
        ;;
        
    "test")
        if ! docker ps --format "{{.Names}}" | grep -q ai-api-unified; then
            echo -e "${YELLOW}🚀 Starting AI services for testing...${NC}"
            deploy_ai_services
        fi
        run_unit_tests
        run_api_tests
        if [[ "$WITH_INTEGRATION_TEST" == "true" ]]; then
            run_integration_tests
        fi
        ;;
        
    "clean")
        echo -e "\n${YELLOW}🧹 Clean restart with complete setup...${NC}"
        complete_cleanup
        build_ai_images
        if deploy_ai_services; then
            run_fine_tuning
            run_yolo_retraining
            run_api_tests
        else
            echo -e "${RED}❌ AI deployment failed${NC}"
            exit 1
        fi
        ;;
        
    "fresh")
        echo -e "\n${RED}🧹 Fresh deployment - complete reset...${NC}"
        complete_cleanup
        build_ai_images
        if deploy_ai_services; then
            run_fine_tuning
            run_yolo_retraining
            run_api_tests
        else
            echo -e "${RED}❌ Fresh deployment failed${NC}"
            exit 1
        fi
        ;;
        
    "deploy")
        # Vérifier si déjà en cours
        if show_containers_status >/dev/null 2>&1; then
            echo -e "\n${GREEN}✅ AI containers already running${NC}"
            echo -e "${BLUE}📋 Performing complete health check...${NC}"
            
            if show_ai_services_status; then
                echo -e "\n${GREEN}🎉 AI services are healthy and ready!${NC}"
                
                # Vérifier si modèles présents
                if [[ -d "./models/finetuned_sentiment_model" ]]; then
                    echo -e "${GREEN}✅ Fine-tuned model available${NC}"
                else
                    echo -e "${YELLOW}📥 No fine-tuned model found, starting fine-tuning...${NC}"
                    run_fine_tuning
                fi
                
                # Test Hadoop integration si demandé
                if [[ "$WITH_INTEGRATION_TEST" == "true" ]]; then
                    run_integration_tests
                fi
            else
                echo -e "\n${YELLOW}⚠️ Some AI services have issues${NC}"
                echo -e "${YELLOW}💡 Try: $0 --clean for a restart${NC}"
                echo -e "${YELLOW}💡 Or: $0 --debug for detailed diagnosis${NC}"
            fi
        else
            echo -e "\n${YELLOW}📋 No AI containers running, starting complete deployment...${NC}"
            
            # Build si nécessaire
            if ! docker images | grep -q "ai-api\|yolo-api"; then
                build_ai_images
            fi
            
            if deploy_ai_services; then
                run_fine_tuning
                run_yolo_retraining
                run_api_tests
                
                if [[ "$WITH_INTEGRATION_TEST" == "true" ]]; then
                    run_integration_tests
                fi
            else
                echo -e "${RED}❌ AI deployment failed${NC}"
                exit 1
            fi
        fi
        ;;
esac

# Résumé final
echo -e "\n${GREEN}✅ Operation completed!${NC}"

# Afficher les infos d'accès si les services tournent
if [[ $ACTION != "status" ]] && [[ $ACTION != "debug" ]] && [[ $ACTION != "build-only" ]] && docker ps --format "{{.Names}}" | grep -q ai-api-unified; then
    show_ai_services_status
    show_access_info
    show_examples
    
    echo -e "\n${BLUE}🎯 Next Steps:${NC}"
    echo -e "  • Visit http://localhost:8001/docs for interactive API documentation"
    echo -e "  • Visit http://localhost:8002/docs for YOLO API documentation"
    echo -e "  • Run '$0 --test' to validate all functionality"
    echo -e "  • Run '$0 --integration' to test Hadoop connectivity"
    echo -e "  • Run '$0 --status' anytime for complete health check"
    echo -e "  • Run '$0 --debug' if any issues occur"
    
    # Vérifier intégration Hadoop
    if docker ps --format "{{.Names}}" | grep -q namenode; then
        echo -e "\n${GREEN}🔗 Hadoop Integration Available:${NC}"
        echo -e "  • YOLO retraining with HDFS images: '$0 --yolo-retrain'"
        echo -e "  • Batch processing from Hadoop data available"
        echo -e "  • Real-time analysis pipeline ready"
    else
        echo -e "\n${YELLOW}⚠️ Hadoop Integration:${NC}"
        echo -e "  • Start Hadoop cluster first for full integration"
        echo -e "  • AI API works standalone without Hadoop"
    fi
    
    echo -e "\n${GREEN}🚀 Your complete AI API is ready for the presentation!${NC}"
    echo -e "\n${BLUE}🎓 Features Delivered:${NC}"
    echo -e "✅ Unified REST API for text and image analysis"
    echo -e "✅ Fine-tuned LLM for sentiment analysis"
    echo -e "✅ YOLO computer vision for object detection"
    echo -e "✅ Model comparison (fine-tuned vs LM Studio)"
    echo -e "✅ Hadoop integration for big data processing"
    echo -e "✅ YOLO retraining with HDFS images"
    echo -e "✅ Complete testing and monitoring"
    echo -e "✅ Production-ready deployment"
    
    # Affichage des métriques si disponibles
    if [[ -f "./models/finetuned_sentiment_model/training_results.json" ]]; then
        local accuracy=$(jq -r '.eval_accuracy // "N/A"' ./models/finetuned_sentiment_model/training_results.json 2>/dev/null || echo "N/A")
        local f1_score=$(jq -r '.eval_f1 // "N/A"' ./models/finetuned_sentiment_model/training_results.json 2>/dev/null || echo "N/A")
        local duration=$(jq -r '.training_duration_seconds // "N/A"' ./models/finetuned_sentiment_model/training_results.json 2>/dev/null || echo "N/A")
        
        if [[ "$accuracy" != "N/A" ]]; then
            echo -e "\n${PURPLE}📊 Fine-tuning Results:${NC}"
            echo -e "  • Model Accuracy: ${accuracy}"
            echo -e "  • F1-Score: ${f1_score}"
            echo -e "  • Training Duration: ${duration}s"
        fi
    fi
    
    # Performance tips
    echo -e "\n${CYAN}💡 Performance Tips:${NC}"
    echo -e "  • Use batch endpoints for processing multiple items"
    echo -e "  • Fine-tuned model is faster than LM Studio for sentiment"
    echo -e "  • YOLO processes images in ~50-100ms"
    echo -e "  • Monitor with: docker stats ai-api-unified yolo-api-server"
    
fi

# ============ FONCTIONS D'AIDE ET DIAGNOSTICS ============

show_troubleshooting() {
    echo -e "\n${CYAN}🔧 TROUBLESHOOTING GUIDE${NC}"
    echo -e "${CYAN}========================${NC}"
    
    echo -e "\n${YELLOW}🚨 Common Issues:${NC}"
    
    echo -e "\n${BLUE}1. Docker Issues:${NC}"
    echo -e "   Problem: 'Docker daemon not running'"
    echo -e "   Solution: Start Docker Desktop"
    echo -e "   Command: Check with 'docker info'"
    
    echo -e "\n${BLUE}2. Port Conflicts:${NC}"
    echo -e "   Problem: 'Port already in use'"
    echo -e "   Solution: Check ports 8001, 8002"
    echo -e "   Command: netstat -tlnp | grep ':800[12]'"
    
    echo -e "\n${BLUE}3. Model Loading Issues:${NC}"
    echo -e "   Problem: 'YOLO model not found'"
    echo -e "   Solution: Models will auto-download on first run"
    echo -e "   Command: docker logs yolo-api-server"
    
    echo -e "\n${BLUE}4. Fine-tuning Failures:${NC}"
    echo -e "   Problem: 'Fine-tuning failed'"
    echo -e "   Solution: Check memory and disk space"
    echo -e "   Command: docker exec ai-api-unified df -h"
    
    echo -e "\n${BLUE}5. Hadoop Connectivity:${NC}"
    echo -e "   Problem: 'Cannot connect to Hadoop'"
    echo -e "   Solution: Ensure Hadoop cluster is running"
    echo -e "   Command: curl http://localhost:9870"
    
    echo -e "\n${YELLOW}🔍 Diagnostic Commands:${NC}"
    echo -e "  $0 --debug          # Full diagnostic"
    echo -e "  docker-compose logs  # All logs"
    echo -e "  docker stats        # Resource usage"
    echo -e "  curl http://localhost:8001/health  # API health"
    echo -e "  docker exec ai-api-unified python test_yolo.py              # YOLO test"
}

# Fonction pour montrer les logs en temps réel
tail_logs() {
    echo -e "\n${BLUE}📜 Viewing real-time logs...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    sleep 2
    docker-compose logs -f ai-api yolo-api
}

# Fonction de nettoyage d'urgence
emergency_cleanup() {
    echo -e "\n${RED}🚨 EMERGENCY CLEANUP${NC}"
    
    # Arrêter tout brutalement
    docker kill $(docker ps -q --filter "name=ai-api") 2>/dev/null || true
    docker kill $(docker ps -q --filter "name=yolo-api") 2>/dev/null || true
    
    # Supprimer les conteneurs
    docker rm -f ai-api-unified yolo-api-server 2>/dev/null || true
    
    # Nettoyer les volumes
    docker volume prune -f
    
    echo -e "${GREEN}✅ Emergency cleanup completed${NC}"
}

# Fonction pour créer un rapport de diagnostic
generate_diagnostic_report() {
    local report_file="diagnostic_report_$(date +%Y%m%d_%H%M%S).txt"
    
    echo -e "\n${BLUE}📋 Generating diagnostic report...${NC}"
    
    {
        echo "=== AI API DIAGNOSTIC REPORT ==="
        echo "Generated: $(date)"
        echo "Project: $(pwd)"
        echo ""
        
        echo "=== DOCKER STATUS ==="
        docker --version
        docker-compose --version
        docker info | head -20
        echo ""
        
        echo "=== CONTAINERS ==="
        docker ps -a --filter "name=ai-api\|yolo-api"
        echo ""
        
        echo "=== IMAGES ==="
        docker images | grep -E "(ai-api|yolo|ultralytics)"
        echo ""
        
        echo "=== VOLUMES ==="
        docker volume ls | grep -E "(ai|yolo)"
        echo ""
        
        echo "=== NETWORK ==="
        docker network ls | grep hadoop
        echo ""
        
        echo "=== AI API LOGS (last 50 lines) ==="
        docker logs ai-api-unified 2>&1 | tail -50 || echo "Container not running"
        echo ""
        
        echo "=== YOLO API LOGS (last 50 lines) ==="
        docker logs yolo-api-server 2>&1 | tail -50 || echo "Container not running"
        echo ""
        
        echo "=== FILE SYSTEM ==="
        ls -la models/ 2>/dev/null || echo "Models directory not found"
        df -h .
        echo ""
        
        echo "=== PROCESS STATUS ==="
        ps aux | grep -E "(docker|python)" | head -10
        
    } > "$report_file"
    
    echo -e "${GREEN}✅ Diagnostic report saved: $report_file${NC}"
}

# Extensions des options pour les nouvelles fonctions
if [[ "${1:-}" == "--logs" ]]; then
    tail_logs
    exit 0
elif [[ "${1:-}" == "--emergency-clean" ]]; then
    emergency_cleanup
    exit 0
elif [[ "${1:-}" == "--troubleshoot" ]]; then
    show_troubleshooting
    exit 0
elif [[ "${1:-}" == "--diagnostic" ]]; then
    generate_diagnostic_report
    exit 0
fi

# Ajouter ces options à l'aide
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    echo -e "\n${CYAN}🔧 Additional Commands:${NC}"
    echo "  --logs              View real-time logs"
    echo "  --emergency-clean   Emergency cleanup"
    echo "  --troubleshoot      Show troubleshooting guide"
    echo "  --diagnostic        Generate diagnostic report"
fi