# AI API - Text and Image Intelligence API

This repository contains the AI component of the *Hadoop & AI Project*, designed to provide a unified API for processing textual and image data using state-of-the-art deep learning models.

## 🔍 Overview

The goal of this project is to develop a RESTful API, powered by FastAPI, that exposes:

- A **Large Language Model (LLM)** for text classification, summarization, or sentiment analysis.
- A **YOLO-based vision model** for image detection and classification.

Both models are accessible via a **single API endpoint** to facilitate integration with the Hadoop infrastructure.

## 🧠 Features

- REST API with FastAPI
- LLM integration (fine-tuned GPT/BERT)
- YOLO image classification and detection
- Supports real-time data input from Hadoop or external sources
- JSON responses optimized for database storage
- Docker-ready & CI/CD compatible

## 🏗️ Architecture

```plaintext
             ┌────────────┐
             │   Client   │
             └────┬───────┘
                  │
         ┌────────▼────────┐
         │    FastAPI      │
         └────┬────┬───────┘
              │    │
        ┌─────▼┐ ┌─▼────────┐
        │ LLM  │ │ YOLOv8   │
        └──────┘ └──────────┘
📦 Setup Instructions
Clone the repository


git clone https://github.com/data-mining-ia-89/ai-api.git
cd ai-api
Install dependencies


pip install -r requirements.txt
Run the API


uvicorn main:app --reload
Access the docs

Visit http://localhost:8000/docs

🧪 Endpoints
Endpoint	Method	Description
/analyze/text	POST	Analyze or classify text
/analyze/image	POST	Analyze or detect objects in image

🤖 Models
LLM: Fine-tuned version of BERT/GPT depending on selected task.

YOLO: Re-trained on custom dataset from web-scraped images.

🔁 Model Retraining
The YOLO model can be re-trained using images from the Hadoop HDFS. Images are preprocessed (resized, normalized) and annotated (manually or semi-automatically) before fine-tuning.

Training reference: Ultralytics Docs

⚙️ DevOps & Deployment
Dockerized app with ready-to-deploy Dockerfile

CI/CD ready via GitHub Actions

Automated image building & deployment on push to main

📁 Folder Structure

api_ia_fastapi/
│
├── app/
│   ├── models/            # ML models and scripts
│   ├── routes/            # API routes
│   ├── utils/             # Preprocessing, image conversion
│   └── main.py            # FastAPI app
│
├── Dockerfile
├── requirements.txt
└── README.md
📊 Performance & Metrics
Execution time and resource usage logged and benchmarked

Monitoring and error tracking integrated (Prometheus / Sentry optional)