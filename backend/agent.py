"""
LangGraph-based RAG agent for Compileit Q&A.

This module contains:
- vector store access and retrieval tool
- a small LangGraph orchestration pipeline
- chat response streaming helpers used by FastAPI
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


SYSTEM_PROMPT = """Du är Compileits AI-assistent.

Regler du ALLTID ska följa:
1. Svara alltid på svenska.
2. Håll svaren korta, konkreta och korrekta.
3. Använd endast information som finns i den hämtade kontexten.
4. Om kontexten inte räcker för att besvara frågan, säg tydligt att du inte vet.
"""

FALLBACK_NO_CONTEXT_RESPONSE = (
    "Jag vet tyvärr inte svaret baserat på informationen jag har hämtat från compileit.com."
)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "compileit_docs")
DEFAULT_K = 4


@dataclass
class SourceItem:
    title: str
    url: str
    snippet: str


class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]
    search_query: str
    context: str
    sources: list[dict[str, str]]


def _get_chroma_persist_dir() -> str:
    return os.getenv("CHROMA_PERSIST_DIR", "./backend/chroma_db")


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    persist_directory = _get_chroma_persist_dir()
    if not os.path.isdir(persist_directory):
        raise RuntimeError(
            f"Chroma-katalogen hittades inte: {persist_directory}. Kör ingest.py först."
        )

    return Chroma(
        collection_name=DEFAULT_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )


@lru_cache(maxsize=1)
def get_chat_model(streaming: bool = False) -> ChatOpenAI:
    # Cache separate model clients for streaming/non-streaming calls.
    return ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, streaming=streaming)


def _extract_message_text(message: dict[str, Any]) -> str:
    parts = message.get("parts")
    if isinstance(parts, list):
        texts: list[str] = []
        for part in parts:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                text = part["text"].strip()
                if text:
                    texts.append(text)
        if texts:
            return "\n".join(texts).strip()

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    return ""


def ui_messages_to_langchain_messages(ui_messages: Sequence[dict[str, Any]]) -> list[BaseMessage]:
    lc_messages: list[BaseMessage] = []

    for message in ui_messages:
        role = message.get("role")
        text = _extract_message_text(message)
        if not text:
            continue

        if role == "user":
            lc_messages.append(HumanMessage(content=text))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=text))
        elif role == "system":
            lc_messages.append(SystemMessage(content=text))

    return lc_messages


def _messages_to_text(messages: Sequence[BaseMessage], max_messages: int = 8) -> str:
    lines: list[str] = []
    for message in messages[-max_messages:]:
        role = "Användare"
        if isinstance(message, AIMessage):
            role = "Assistent"
        elif isinstance(message, SystemMessage):
            role = "System"

        content = str(message.content).strip()
        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines).strip()


def _last_user_message(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            text = str(message.content).strip()
            if text:
                return text
    return ""


def _sanitize_title(title: str, url: str) -> str:
    cleaned = (title or "").strip()
    if cleaned:
        return cleaned
    return url.replace("https://", "").replace("http://", "").rstrip("/")


@tool
def search_compileit_knowledge(question: str) -> dict[str, Any]:
    """
    Search Compileit's indexed website content via Chroma similarity search.
    """
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(question, k=DEFAULT_K)

    sources: list[SourceItem] = []
    context_chunks: list[str] = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        url = str(metadata.get("source", "")).strip()
        if not url:
            continue

        title = _sanitize_title(str(metadata.get("title", "")), url)
        snippet = " ".join(str(doc.page_content).split())
        snippet = snippet[:700].strip()

        sources.append(SourceItem(title=title, url=url, snippet=snippet))
        context_chunks.append(f"[Källa {index}] {title}\nURL: {url}\nInnehåll: {snippet}")

    return {
        "context": "\n\n".join(context_chunks).strip(),
        "sources": [{"title": source.title, "url": source.url} for source in sources],
    }


def _rewrite_search_query(messages: Sequence[BaseMessage]) -> str:
    latest_question = _last_user_message(messages)
    if not latest_question:
        return ""

    if len(messages) <= 1:
        return latest_question

    history = _messages_to_text(messages)
    prompt = (
        "Du hjälper till att skapa en sökfråga för RAG.\n"
        "Skriv om senaste användarfrågan till en fristående sökfråga på svenska.\n"
        "Behåll namn, fakta och sammanhang från historiken vid behov.\n"
        "Returnera endast sökfrågan, ingen förklaring.\n\n"
        f"Historik:\n{history}\n\n"
        f"Senaste fråga:\n{latest_question}"
    )

    response = get_chat_model(streaming=False).invoke([HumanMessage(content=prompt)])
    rewritten = str(response.content).strip()
    return rewritten or latest_question


def _build_context_graph():
    graph = StateGraph(AgentState)

    def rewrite_node(state: AgentState) -> AgentState:
        query = _rewrite_search_query(state.get("messages", []))
        return {"search_query": query}

    def retrieve_node(state: AgentState) -> AgentState:
        search_query = state.get("search_query", "").strip() or _last_user_message(
            state.get("messages", [])
        )
        if not search_query:
            return {"context": "", "sources": []}

        tool_result = search_compileit_knowledge.invoke({"question": search_query})
        context = str(tool_result.get("context", "")).strip()
        raw_sources = tool_result.get("sources", [])

        sources: list[dict[str, str]] = []
        if isinstance(raw_sources, list):
            for source in raw_sources:
                if not isinstance(source, dict):
                    continue
                title = str(source.get("title", "")).strip()
                url = str(source.get("url", "")).strip()
                if url:
                    sources.append({"title": title or url, "url": url})

        return {"context": context, "sources": sources}

    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", END)

    return graph.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def get_context_graph():
    return _build_context_graph()


def _chunk_to_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)

    return ""


def _format_sources_block(sources: Sequence[dict[str, str]]) -> str:
    lines: list[str] = []
    for idx, source in enumerate(sources, start=1):
        title = source.get("title", "").strip() or source.get("url", "").strip()
        url = source.get("url", "").strip()
        if url:
            lines.append(f"{idx}. {title} ({url})")
    return "\n".join(lines).strip()


def prepare_context(
    ui_messages: Sequence[dict[str, Any]], session_id: str
) -> tuple[list[BaseMessage], str, list[dict[str, str]], str]:
    lc_messages = ui_messages_to_langchain_messages(ui_messages)
    if not lc_messages:
        raise ValueError("Inga giltiga meddelanden skickades till agenten.")

    graph = get_context_graph()
    result = graph.invoke(
        {"messages": lc_messages},
        config={"configurable": {"thread_id": session_id}},
    )

    search_query = str(result.get("search_query", "")).strip()
    context = str(result.get("context", "")).strip()
    sources = result.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    parsed_sources: list[dict[str, str]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        title = str(source.get("title", "")).strip()
        url = str(source.get("url", "")).strip()
        if url:
            parsed_sources.append({"title": title or url, "url": url})

    return lc_messages, context, parsed_sources, search_query


def stream_answer_tokens(
    ui_messages: Sequence[dict[str, Any]],
    session_id: str,
) -> tuple[list[dict[str, str]], Iterator[str]]:
    lc_messages, context, sources, search_query = prepare_context(ui_messages, session_id)

    if not context:
        return sources, iter([FALLBACK_NO_CONTEXT_RESPONSE])

    sources_block = _format_sources_block(sources)
    system_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Sökfråga: {search_query or _last_user_message(lc_messages)}\n\n"
        f"Tillgänglig kontext från compileit.com:\n{context}\n\n"
        f"Källor:\n{sources_block if sources_block else 'Inga explicita källor.'}"
    )

    prompt_messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *lc_messages]
    model = get_chat_model(streaming=True)

    def token_iterator() -> Iterator[str]:
        for chunk in model.stream(prompt_messages):
            text = _chunk_to_text(chunk)
            if text:
                yield text

    return sources, token_iterator()
