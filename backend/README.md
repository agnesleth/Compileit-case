# Compileit Backend (FastAPI + LangGraph)

This backend uses a corporate-first RAG pipeline:
- index only `compileit.com` content (not careers pages)
- filter noisy/legal/recruiting boilerplate
- rank retrieval candidates with vector score + lexical overlap + section boosts
- keep conservative low-confidence behavior (`Jag vet inte ...`)

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## 2. Configure environment

```bash
cp backend/.env.example backend/.env
```

Then set at least:

```bash
OPENAI_API_KEY=your_key_here
```

Optional debug:

```bash
DEBUG_RETRIEVAL=1
```

## 3. Reindex data (required after ingestion/retrieval changes)

```bash
python backend/ingest.py
```

## 4. Start API server

Use non-reload mode to avoid broad file watcher issues:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -i http://127.0.0.1:8000/health
```

## 5. Connect frontend proxy

In frontend `.env.local`:

```bash
BACKEND_CHAT_URL=http://127.0.0.1:8000/api/chat
```

## Validation checklist

1. Query `Vilka branscher jobbar ni med?` should return mainly corporate/service/case sources.
2. Query `Vad erbjuder ni för typer av AI-tjänster?` should cite service-related pages.
3. Weak/underspecified query should trigger conservative clarification response.
4. Streaming/citations should still work in the frontend unchanged.
