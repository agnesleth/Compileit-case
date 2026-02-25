# Compileit Backend (FastAPI + LangGraph)

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## 2. Configure environment

Edit `backend/.env` and set at least:

```bash
OPENAI_API_KEY=your_key_here
```

## 3. Run ingestion

```bash
python backend/ingest.py
```

This crawls `compileit.com`, chunks content, and stores embeddings in Chroma.

## 4. Start API server

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## 5. Connect frontend proxy

In the frontend `.env.local`, set one of:

```bash
BACKEND_CHAT_URL=http://localhost:8000/api/chat
# or
# BACKEND_BASE_URL=http://localhost:8000
```
