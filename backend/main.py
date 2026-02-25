"""
FastAPI server exposing a Vercel-AI-SDK-compatible streaming chat endpoint.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .agent import stream_answer_tokens

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="Compileit RAG Backend", version="0.1.0")


def sse_data(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def as_ui_stream(
    *,
    ui_messages: list[dict[str, Any]],
    session_id: str,
) -> Iterator[str]:
    message_id = f"msg_{uuid.uuid4().hex}"
    text_id = f"text_{uuid.uuid4().hex}"

    try:
        sources, token_stream = stream_answer_tokens(
            ui_messages=ui_messages,
            session_id=session_id,
        )

        yield sse_data({"type": "start", "messageId": message_id})

        for source in sources:
            url = source.get("url", "").strip()
            if not url:
                continue
            title = source.get("title", "").strip() or url
            yield sse_data(
                {
                    "type": "source-url",
                    "sourceId": url,
                    "url": url,
                    "title": title,
                }
            )

        yield sse_data({"type": "text-start", "id": text_id})
        for delta in token_stream:
            if delta:
                yield sse_data({"type": "text-delta", "id": text_id, "delta": delta})

        yield sse_data({"type": "text-end", "id": text_id})
        yield sse_data({"type": "finish-step"})
        yield sse_data({"type": "finish"})
        yield sse_data("[DONE]")
    except Exception as exc:  # noqa: BLE001
        # The UI protocol expects structured error parts in-stream.
        yield sse_data({"type": "error", "errorText": f"Backend-fel: {str(exc)}"})
        yield sse_data({"type": "finish"})
        yield sse_data("[DONE]")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: Request):
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Ogiltig JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body måste vara ett objekt.")

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="Request body måste innehålla 'messages'.")

    if not messages:
        return JSONResponse({"error": "messages får inte vara tom."}, status_code=400)

    session_id = str(payload.get("id") or uuid.uuid4().hex)

    headers = {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "x-vercel-ai-ui-message-stream": "v1",
    }

    return StreamingResponse(
        as_ui_stream(ui_messages=messages, session_id=session_id),
        headers=headers,
        media_type="text/event-stream",
    )
