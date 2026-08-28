# PrathGPT Deployment Guide

PrathGPT is an open-source **agentic AI chatbot** built with **Python, FastAPI, LangGraph, LangChain, Mistral AI, Google Gemini Embeddings, Tavily, ChromaDB, and SQLite**.

It supports real-time streaming chat, document uploads, retrieval-augmented generation (RAG), web search, stock prices, live weather, conversation memory, human-in-the-loop interaction, and a responsive web UI.

---

## Features

* Chat with an AI agent powered by Mistral AI models (`mistral-small`, `mistral-large`, `codestral`, etc.)
* Stream responses in real time with Server-Sent Events (SSE) and live Markdown rendering
* Upload documents such as PDF, DOCX, TXT, MD, PY, and CSV
* Use uploaded files as isolated context through RAG (ChromaDB + Gemini Embeddings)
* Search the web with Tavily for current information
* Retrieve stock prices and live weather reports
* Store and recall conversation history & long-term memory
* Responsive FastAPI-based web interface with voice dictation
* Docker-ready deployment
* AWS CI/CD support using GitHub Actions, ECR, and EC2

---

## Project Overview

This project combines:

* **FastAPI** for the backend server and API endpoints
* **Jinja2** for rendering the frontend UI
* **LangGraph** for ReAct agent orchestration and checkpointing
* **LangChain** for tools, messages, and RAG workflow
* **Mistral AI** as the primary LLM provider
* **Google Gemini** for text embeddings (`gemini-embedding-001`)
* **Tavily** for web search
* **ChromaDB** for vector search over uploaded documents
* **SQLite / SQLAlchemy** for conversation and memory persistence
* **Docker** for containerized deployment

---

## Prerequisites

Make sure you have the following installed:

* Python 3.11
* pip or conda
* Git
* Mistral API key (`MISTRAL_API_KEY`)
* Google API key (`GOOGLE_API_KEY` for embeddings)
* Tavily API key (`TAVILY_API_KEY` for web search)
* OpenWeather API key (`OPENWEATHER_API_KEY` - optional for weather tool)
* AlphaVantage API key (`ALPHAVANTAGE_API_KEY` - optional for stock tool)

Optional for deployment:

* Docker
* AWS account (IAM User with ECR & EC2 access)
* Amazon ECR repository
* Amazon EC2 instance (Ubuntu)
* GitHub Actions self-hosted runner

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PrathmeshRanjan/PrathGPT.git
```

### 2. Navigate to the project directory

```bash
cd PrathGPT
```

### 3. Create a virtual environment

Using conda:

```bash
conda create -n prathgpt python=3.11 -y
conda activate prathgpt
```

Or using python venv:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root directory:

```env
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key

LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=prathgpt
```

---

## Run Locally

Start the FastAPI app:

```bash
python app.py
```

The app will be available at:

```text
http://127.0.0.1:8080
```

---

## Project Structure

```text
PrathGPT/
│
├── app.py                  # FastAPI server and streaming chat endpoints
├── agent.py                # LangGraph agent setup and model configurations
├── database.py             # Conversation, messages, and long-term memory ORM
├── rag.py                  # Document parsing, chunking, and ChromaDB vector search
├── tools.py                # Tools (Tavily search, weather, stock, calculator, memory)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment configuration template
├── Dockerfile              # Docker image configuration
├── .dockerignore           # Docker build exclusions
├── .gitignore              # Git ignored files (databases, secrets, uploads)
│
├── template/
│   └── index.html          # Frontend UI with streaming & voice dictation
│
├── uploads/                # Local uploaded document storage
├── data/                   # SQLite database (langgraph checkpoints & memory)
└── chroma_db/              # ChromaDB vector database storage
```

---

## Docker Deployment

### 1. Build the Docker image

```bash
docker build -t prathgpt .
```

### 2. Run the Docker container

```bash
docker run -d \
  --name prathgpt \
  --restart always \
  -p 8080:8080 \
  --env-file .env \
  prathgpt
```

The app will be available at:

```text
http://localhost:8080
```

---

## AWS CI/CD Deployment with GitHub Actions

This project is configured for automated deployment to AWS using:

* GitHub Actions
* Amazon ECR
* Amazon EC2
* Docker
* GitHub self-hosted runner

---

### 1. Create an IAM User

Create an IAM user for deployment in the AWS Console with programmatic access and attach the following policies:

```text
AmazonEC2ContainerRegistryFullAccess
AmazonEC2FullAccess
```

---

### 2. Create an ECR Repository

Create an Amazon ECR repository (e.g. named `prathgpt`).

For GitHub Secrets, only save the repository name:

```text
ECR_REPO=prathgpt
```

---

### 3. Create and Configure an EC2 Instance

1. Launch an Ubuntu 22.04 / 24.04 EC2 instance (`t3.medium` or larger recommended for embeddings & ChromaDB).
2. Configure the Security Group with the following inbound rules:
   * **Custom TCP**: Port `8080` (Source: `0.0.0.0/0`)
   * **SSH**: Port `22` (Source: Your IP)

---

### 4. Install Docker on EC2

Connect to your EC2 instance via SSH and run:

```bash
sudo apt-get update -y && sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
docker --version
```

---

### 5. Configure EC2 as a GitHub Self-Hosted Runner

In your GitHub repository:
1. Navigate to **Settings → Actions → Runners → New self-hosted runner**.
2. Select **Linux** architecture and run the provided registration commands on your EC2 instance.
3. Install and run as a systemd service:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

---

## GitHub Secrets Checklist

Add the following secrets to your GitHub repository (**Settings → Secrets and variables → Actions**):

| Secret Name | Description | Example |
| :--- | :--- | :--- |
| `AWS_ACCESS_KEY_ID` | IAM User Access Key | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | IAM User Secret Key | `wJalr...` |
| `AWS_DEFAULT_REGION` | AWS Region | `us-east-1` |
| `ECR_REPO` | ECR repository name | `prathgpt` |
| `MISTRAL_API_KEY` | Mistral AI API key | `...` |
| `GOOGLE_API_KEY` | Google API key (for Gemini Embeddings) | `...` |
| `TAVILY_API_KEY` | Tavily Web Search API key | `tvly-...` |
| `OPENWEATHER_API_KEY` | OpenWeather API key | `...` |
| `ALPHAVANTAGE_API_KEY` | AlphaVantage API key | `...` |
| `LANGSMITH_TRACING` | Enable/disable LangSmith tracing | `false` |
| `LANGSMITH_ENDPOINT` | LangSmith Endpoint | `https://api.smith.langchain.com` |
| `LANGSMITH_API_KEY` | LangSmith API Key (Optional) | `...` |
| `LANGSMITH_PROJECT` | LangSmith Project Name | `prathgpt` |

---

## GitHub Actions Workflow

The automated workflow is located at:

```text
.github/workflows/cicd.yaml
```

Every push to `main` will automatically:
1. Build the Docker image on GitHub Actions.
2. Push the image to Amazon ECR.
3. Trigger the self-hosted runner on EC2.
4. Pull the latest image from ECR.
5. Stop and remove the old container.
6. Launch the new container with injected environment secrets on port 8080.
7. Prune stale images.