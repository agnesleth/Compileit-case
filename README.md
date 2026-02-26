# Compileit Case

Frontend (Next.js + Vercel AI SDK) + backend (FastAPI + LangGraph + Chroma) for a Swedish Compileit assistant.

## Prerequisites

- Node.js 20+
- Python 3.9+
- OpenAI API key

## 1. Install dependencies

Frontend:

```bash
npm install
```

Backend (recommended in virtual environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## 2. Configure environment variables

Backend:

```bash
cp backend/.env.example backend/.env
```

Set at least:

```bash
OPENAI_API_KEY=your_key_here
```

Frontend:

Create `.env.local` with:

```bash
BACKEND_CHAT_URL=http://127.0.0.1:8000/api/chat
```

Alternative:

```bash
# BACKEND_BASE_URL=http://127.0.0.1:8000
```

## 3. Build the vector index (required before first run)

```bash
.venv/bin/python backend/ingest.py
```

## 4. Start the app

Terminal 1 (backend):

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 (frontend):

```bash
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Open:

- Local: `http://localhost:3000`
- Same Wi-Fi device: `http://<your-laptop-ip>:3000`

## 5. Useful checks

```bash
npm run lint
npm run build
curl -i http://127.0.0.1:8000/health
```

## Why this design

- A Next.js `/api/chat` proxy decouples frontend from backend deployment and hides backend URL details from the client.
- Corporate-first retrieval improves precision by prioritizing business-relevant Compileit pages over noisy or recruiting/legal content.
- Conservative confidence gating avoids hallucinations by returning "Jag vet inte" when context quality is weak.
- Source citations are passed through and rendered in UI so answers stay verifiable and grounded.
