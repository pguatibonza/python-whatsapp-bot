# Application Deployment Guide

This repository contains two services packaged as Podman containers:

1. **LangGraph Service** (FastAPI AI agent exposing `/chat` endpoint)
2. **Quart App** (handles WhatsApp webhooks and forwards messages to LangGraph)

Both services can be built and run locally using Podman. Follow the steps below to get everything up and running.

---

## Prerequisites

* Podman installed on your machine
* A project root containing:

  * `langgraph-service/` directory
  * `quart-app/` directory
  * `.env` file with your environment variables

## Environment Variables

Create a `.env` file in the project root with the following variables:

```dotenv
# Shared variables
APP_SECRET
ACCESS_TOKEN
APP_ID
APP_SECRET
RECIPIENT_WAID
VERSION=v19.0
PHONE_NUMBER_ID
VERIFY_TOKEN
LANGGRAPH_URL=http://langgraph-service:8000


# Quart App-specific
PORT=8080

# LangGraph Service-specific
OPENAI_API_KEY=<your_openai_api_key>
# ...any other variables your FastAPI app requires
```

> **Important:** Do **not** surround values with quotes (`"`) in your `.env` file.

---

## Project Structure

```
project-root/
├── langgraph/
│   ├── __pycache__/
│   ├── graphrag_reduced/
│   ├── secrets/
│   ├── src/
│   ├── yamls/
│   ├── .env
│   ├── config.py
│   ├── Dockerfile
│   ├── main.py
│   ├── models.py
│   ├── README.md
│   └── requirements.txt
├── meta_app/
│   ├── app/
│   │   └── utils/whatsapp_utils.py
│   ├── .env
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── run.py
│   ├── .gitignore
│   ├── README.md
│   └── supabase_setup.txt
└── README.md
```

---

## 1) Build the Images

### a) LangGraph Service

```bash
cd langgraph
podman build -t langgraph-service .
```

### b) Quart App

```bash
cd meta_-app
podman build -t meta_app .
```

---

## 2) Setup Networking

Create a user-defined Podman network to allow containers to communicate by name:

```bash
podman network create meta-net
```

---

## 3) Run the Containers

### a) LangGraph Service

```bash
podman run -d \
  --name langgraph-service \
  --env-file .env \
  --network meta-net \
  -p 8000:8000 \
  langgraph-service
```

### b) Quart App

```bash
podman run -d \
  --name meta_app \
  --env-file .env \
  --network meta-net \
  -p 8080:8080 \
  meta_app
```

Now:

* LangGraph API is at `http://localhost:8000/chat`
* Quart App (WhatsApp webhook) is at `http://localhost:8080` (adjust your webhook route if different)

---

## 4) Testing

#### Test LangGraph Directly

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"wa_id":"test","message":"Hola"}'
```

