# Project overview

Use this guide to build the Python backend for a RAG (Retrieval-Augmented Generation) chat application. The system answers questions in Swedish about the company Compileit (compileit.com).

# Feature requirements

- Tech stack: Python, FastAPI, LangGraph, LangChain, ChromaDB (for local vector storage), and OpenAI.
- Ingestion: Create a script (`ingest.py`) that crawls compileit.com, chunks the text, creates embeddings using OpenAI, and stores them in ChromaDB.
- Orchestration & Tool Calling: Use LangGraph as the core framework for the agent. The agent should have access to a custom tool (`@tool`) that performs similarity searches in the ChromaDB vector store.
- State Management: The LangGraph state must keep track of the conversation history so the user can ask follow-up questions in the same session.
- System Prompt Rules:
  1. ALWAYS answer in Swedish.
  2. Keep answers concise and accurate.
  3. Hallucination guardrail: If the requested information is not found in the retrieved context, the agent MUST explicitly state that it does not know the answer.
- Streaming: The FastAPI application must have an endpoint (e.g., `/api/chat`) that streams the LLM's response tokens back to the frontend in real-time, formatted to be compatible with the Vercel AI SDK.

# Expected File structure

backend/
├── main.py          # FastAPI application and streaming endpoint
├── agent.py         # LangGraph logic, state management, and tools
├── ingest.py        # Web scraper and ChromaDB indexer
├── requirements.txt # Dependencies (fastapi, uvicorn, langchain, langgraph, chromadb, openai, etc.)
└── .env             # Environment variables (OPENAI_API_KEY)

# Rules

- Ensure the LangGraph implementation is modular and easy to read.
- Use `StreamingResponse` from FastAPI to handle the token streaming.
- Document the code clearly, especially the graph/agent logic, to make it easy to explain during a technical interview.

