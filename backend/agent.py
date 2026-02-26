"""
LangGraph-based RAG agent for Compileit Q&A.

This module contains:
- vector store access and retrieval tool
- a small LangGraph orchestration pipeline
- chat response streaming helpers used by FastAPI
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

SYSTEM_PROMPT = """Du är Compileits AI-assistent.

Regler du ALLTID ska följa:
1. Svara alltid på svenska.
2. Håll svaren korta, konkreta och korrekta.
3. Använd endast information som finns i den hämtade kontexten.
4. Om kontexten inte räcker för att besvara frågan, säg tydligt att du inte vet.
5. Om frågan gäller kontakt eller kontor, lista alla kontaktvägar och orter som nämns i kontexten.
"""

FALLBACK_NO_CONTEXT_RESPONSE = (
    "Jag vet inte säkert baserat på informationen jag har hittat om Compileit.\n"
    "Menar du branscher hos kunderna, eller vilka typer av uppdrag och tjänster vi jobbar med?"
)
LOW_CONFIDENCE_RESPONSE_TEMPLATE = (
    "Jag vet inte säkert utifrån tillgänglig information för frågan: \"{query}\".\n"
    "Kan du förtydliga vad du menar?\n"
    "Exempel: branscher hos kunder, typer av uppdrag eller vilka tjänster Compileit erbjuder."
)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "compileit_docs")
DEFAULT_CANDIDATE_K = int(os.getenv("RETRIEVAL_CANDIDATE_K", "24"))
DEFAULT_FINAL_K = int(os.getenv("RETRIEVAL_FINAL_K", "6"))
DEBUG_RETRIEVAL = os.getenv("DEBUG_RETRIEVAL", "0").lower() in ("1", "true", "yes")
PRINT_RETRIEVAL_REPORT = os.getenv("PRINT_RETRIEVAL_REPORT", "1").lower() in (
    "1",
    "true",
    "yes",
)

SECTION_BOOSTS: dict[str, float] = {
    "contact": 0.22,
    "services": 0.20,
    "cases": 0.18,
    "about": 0.08,
    "news": 0.03,
    "general": 0.0,
}
SWEDISH_STOPWORDS = {
    "och",
    "att",
    "det",
    "som",
    "en",
    "ett",
    "den",
    "de",
    "i",
    "på",
    "av",
    "till",
    "för",
    "med",
    "om",
    "hur",
    "vad",
    "vilka",
    "vilken",
    "är",
    "har",
    "ni",
    "er",
    "vi",
    "man",
    "från",
}
GENERIC_TEXT_PATTERNS = (
    "webbplatsen använder cookies",
    "välj vilka cookies du vill godkänna",
    "integritetspolicy",
    "cookie policy",
    "jobbar du redan på compileit",
)
BUSINESS_INTENT_BRANSCH_PATTERNS = (r"\bbransch\w*", r"\bindustri\w*")
BRANSCH_EXPANSION_TERMS = ("kunder", "uppdrag", "case", "verksamhet")
CONTACT_INTENT_PATTERNS = (
    r"\bkontakt\w*",
    r"\bmail\w*",
    r"\bepost\w*",
    r"\be-?post\w*",
    r"\btelefon\w*",
    r"\btel\b",
    r"\bkontor\w*",
    r"\badress\w*",
    r"\bsitter\b",
)
CONTACT_EXPANSION_TERMS = (
    "kontakt",
    "epost",
    "email",
    "telefon",
    "kontor",
    "adress",
    "stockholm",
    "kalmar",
    "skövde",
)
OUT_OF_SCOPE_STEMS = (
    "väd",
    "temperat",
    "prognos",
    "match",
    "sport",
    "aktie",
    "bitcoin",
    "krypto",
)
COMPILEIT_RELEVANCE_STEMS = (
    "compileit",
    "tjänst",
    "uppdrag",
    "case",
    "kund",
    "bransch",
    "ai",
    "apputveck",
    "webbutveck",
    "kontakt",
    "mail",
    "telefon",
    "kontor",
    "adress",
    "stockhol",
    "kalmar",
    "skövd",
    "säker",
    "sekret",
    "karri",
    "nyhet",
)


@dataclass
class RetrievedCandidate:
    title: str
    url: str
    section: str
    path: str
    content_type: str
    snippet: str
    vector_score: float
    lexical_score: float
    section_boost: float
    intent_boost: float
    final_score: float


class SourceItem(TypedDict):
    title: str
    url: str


class AgentState(TypedDict, total=False):
    messages: list[BaseMessage]
    search_query: str
    expanded_query: str
    context: str
    sources: list[SourceItem]
    low_confidence: bool


def debug_log(message: str, *args: object) -> None:
    if DEBUG_RETRIEVAL:
        logger.info(message, *args)


def _truncate_text(text: str, limit: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def _log_retrieval_report(
    question: str,
    expanded_query: str,
    selected_candidates: Sequence[RetrievedCandidate],
    low_confidence: bool,
) -> None:
    if not PRINT_RETRIEVAL_REPORT:
        return

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 88)
    lines.append("RETRIEVAL REPORT")
    lines.append("=" * 88)
    lines.append(f"Query          : {question}")
    lines.append(f"Expanded query : {expanded_query}")
    lines.append(f"Low confidence : {low_confidence}")
    lines.append(f"Selected hits  : {len(selected_candidates)}")
    lines.append("-" * 88)

    if not selected_candidates:
        lines.append("No candidates selected.")
    else:
        for idx, candidate in enumerate(selected_candidates, start=1):
            lines.append(
                f"[{idx}] section={candidate.section} "
                f"type={candidate.content_type} "
                f"final={candidate.final_score:.3f} "
                f"vector={candidate.vector_score:.3f} "
                f"lexical={candidate.lexical_score:.3f} "
                f"intent={candidate.intent_boost:.3f}"
            )
            lines.append(f"    title : {candidate.title}")
            lines.append(f"    url   : {candidate.url}")
            lines.append(f"    path  : {candidate.path}")
            lines.append(f"    ctx   : {_truncate_text(candidate.snippet)}")
            lines.append("-" * 88)

    lines.append("=" * 88)
    logger.info("\n%s", "\n".join(lines))


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


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9åäö]+", text.lower())
    return [token for token in tokens if token not in SWEDISH_STOPWORDS and len(token) > 2]


def _clamp_score(score: float) -> float:
    if score < 0:
        return 0.0
    if score > 1:
        return 1.0
    return score


def _lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0

    text_tokens = set(_tokenize(text))
    overlap = query_tokens.intersection(text_tokens)
    return len(overlap) / len(query_tokens)


def _is_generic_chunk(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in GENERIC_TEXT_PATTERNS)


def _tokens_match_stems(tokens: set[str], stems: Sequence[str]) -> bool:
    for token in tokens:
        for stem in stems:
            if token.startswith(stem):
                return True
    return False


def _is_contact_intent(query: str) -> bool:
    lowered = query.lower()
    return any(re.search(pattern, lowered) for pattern in CONTACT_INTENT_PATTERNS)


def _expand_search_query(query: str) -> str:
    expanded_terms: list[str] = []
    for pattern in BUSINESS_INTENT_BRANSCH_PATTERNS:
        if re.search(pattern, query.lower()):
            expanded_terms.extend(BRANSCH_EXPANSION_TERMS)
            break

    if _is_contact_intent(query):
        expanded_terms.extend(CONTACT_EXPANSION_TERMS)

    if not expanded_terms:
        return query

    query_tokens = set(_tokenize(query))
    deduped_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in expanded_terms:
        if term in query_tokens or term in seen_terms:
            continue
        seen_terms.add(term)
        deduped_terms.append(term)

    if not deduped_terms:
        return query

    expansion = " ".join(deduped_terms)
    return f"{query} {expansion}".strip()


def _build_candidate(
    doc: Document,
    vector_score: float,
    query_for_lexical: str,
    *,
    contact_intent: bool,
) -> RetrievedCandidate | None:
    metadata = doc.metadata or {}
    url = str(metadata.get("source", "")).strip()
    if not url:
        return None

    domain = str(metadata.get("domain", "")).strip().lower()
    if domain and domain != "compileit.com":
        return None

    path = str(metadata.get("path", "")).strip().lower()
    content_type = str(metadata.get("content_type", "page")).strip().lower() or "page"
    section = str(metadata.get("section", "general")).strip().lower() or "general"
    if content_type == "contact" or path.startswith("/kontakt"):
        section = "contact"

    section_boost = SECTION_BOOSTS.get(section, 0.0)
    intent_boost = 0.0
    if contact_intent:
        if content_type == "contact" or path.startswith("/kontakt"):
            intent_boost = 0.22
        elif section in {"about", "general"}:
            intent_boost = 0.04

    snippet = " ".join(str(doc.page_content).split())
    snippet_limit = 1200 if content_type == "contact" else 700
    snippet = snippet[:snippet_limit].strip()
    lexical_score = _lexical_overlap_score(query_for_lexical, snippet)
    final_score = (0.75 * vector_score) + (0.25 * lexical_score) + section_boost + intent_boost

    return RetrievedCandidate(
        title=_sanitize_title(str(metadata.get("title", "")), url),
        url=url,
        section=section,
        path=path or "/",
        content_type=content_type,
        snippet=snippet,
        vector_score=vector_score,
        lexical_score=lexical_score,
        section_boost=section_boost,
        intent_boost=intent_boost,
        final_score=final_score,
    )


def _retrieve_ranked_candidates(question: str) -> tuple[str, list[RetrievedCandidate]]:
    expanded_query = _expand_search_query(question)
    contact_intent = _is_contact_intent(question)
    vector_store = get_vector_store()

    try:
        doc_score_pairs = vector_store.similarity_search_with_relevance_scores(
            expanded_query,
            k=DEFAULT_CANDIDATE_K,
            filter={"domain": "compileit.com"},
        )
    except TypeError:
        doc_score_pairs = vector_store.similarity_search_with_relevance_scores(
            expanded_query,
            k=DEFAULT_CANDIDATE_K,
        )
    candidates: list[RetrievedCandidate] = []
    for doc, raw_score in doc_score_pairs:
        vector_score = _clamp_score(float(raw_score))
        candidate = _build_candidate(
            doc=doc,
            vector_score=vector_score,
            query_for_lexical=expanded_query,
            contact_intent=contact_intent,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda item: item.final_score, reverse=True)
    return expanded_query, candidates


def _is_low_confidence(
    question: str,
    selected_candidates: Sequence[RetrievedCandidate],
) -> bool:
    if not selected_candidates:
        return True

    top = selected_candidates[0]
    max_lexical = max(item.lexical_score for item in selected_candidates)
    sections = {item.section for item in selected_candidates}
    generic_count = sum(1 for item in selected_candidates if _is_generic_chunk(item.snippet))
    query_tokens = set(_tokenize(question))
    contact_intent = _is_contact_intent(question)
    has_contact_evidence = any(
        item.section == "contact"
        or item.content_type == "contact"
        or item.path.startswith("/kontakt")
        for item in selected_candidates
    )

    if top.final_score < 0.42:
        return True
    if max_lexical < 0.08:
        return True
    if generic_count >= max(2, len(selected_candidates) - 1):
        return True
    if sections == {"general"} and top.final_score < 0.55:
        return True
    if re.search(r"\bbransch\w*", question.lower()):
        if not any(item.section in {"services", "cases", "about"} for item in selected_candidates):
            return True
    if contact_intent:
        if not has_contact_evidence:
            return True
        if top.final_score < 0.50:
            return True
    if _tokens_match_stems(query_tokens, OUT_OF_SCOPE_STEMS):
        return True
    if not _tokens_match_stems(query_tokens, COMPILEIT_RELEVANCE_STEMS) and top.final_score < 0.70:
        return True

    return False


def _build_context_and_sources(
    candidates: Sequence[RetrievedCandidate],
) -> tuple[str, list[SourceItem], list[dict[str, float | str]]]:
    context_chunks: list[str] = []
    sources: list[SourceItem] = []
    seen_source_urls: set[str] = set()
    retrieval_rows: list[dict[str, float | str]] = []

    for index, candidate in enumerate(candidates, start=1):
        context_chunks.append(
            f"[Källa {index}] {candidate.title}\n"
            f"URL: {candidate.url}\n"
            f"Innehåll: {candidate.snippet}"
        )
        if candidate.url not in seen_source_urls:
            seen_source_urls.add(candidate.url)
            sources.append({"title": candidate.title, "url": candidate.url})
        retrieval_rows.append(
            {
                "title": candidate.title,
                "url": candidate.url,
                "section": candidate.section,
                "path": candidate.path,
                "content_type": candidate.content_type,
                "vector_score": candidate.vector_score,
                "lexical_score": candidate.lexical_score,
                "section_boost": candidate.section_boost,
                "intent_boost": candidate.intent_boost,
                "final_score": candidate.final_score,
            }
        )

    return "\n\n".join(context_chunks).strip(), sources, retrieval_rows


@tool
def search_compileit_knowledge(question: str) -> dict[str, Any]:
    """
    Search Compileit's indexed website content via ranked vector retrieval.
    """
    expanded_query, ranked_candidates = _retrieve_ranked_candidates(question)
    selected: list[RetrievedCandidate] = []
    per_url_count: dict[str, int] = {}
    for candidate in ranked_candidates:
        current_count = per_url_count.get(candidate.url, 0)
        if current_count >= 2:
            continue
        per_url_count[candidate.url] = current_count + 1
        selected.append(candidate)
        if len(selected) >= DEFAULT_FINAL_K:
            break

    if not selected:
        selected = ranked_candidates[:DEFAULT_FINAL_K]
    context, sources, retrieval_rows = _build_context_and_sources(selected)
    low_confidence = _is_low_confidence(question=question, selected_candidates=selected)

    debug_log("retrieval.query original=%s", question)
    debug_log("retrieval.query expanded=%s", expanded_query)
    debug_log("retrieval.count selected=%s total=%s", len(selected), len(ranked_candidates))
    for row in retrieval_rows[:6]:
        debug_log(
            "retrieval.hit section=%s type=%s final=%.3f vec=%.3f lex=%.3f intent=%.3f url=%s",
            row["section"],
            row["content_type"],
            float(row["final_score"]),
            float(row["vector_score"]),
            float(row["lexical_score"]),
            float(row["intent_boost"]),
            row["url"],
        )
    debug_log("retrieval.low_confidence=%s", low_confidence)
    _log_retrieval_report(
        question=question,
        expanded_query=expanded_query,
        selected_candidates=selected,
        low_confidence=low_confidence,
    )

    return {
        "context": context,
        "sources": sources,
        "retrieval": retrieval_rows,
        "expanded_query": expanded_query,
        "low_confidence": low_confidence,
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
            return {"context": "", "sources": [], "low_confidence": True}

        tool_result = search_compileit_knowledge.invoke({"question": search_query})
        context = str(tool_result.get("context", "")).strip()
        expanded_query = str(tool_result.get("expanded_query", search_query)).strip() or search_query
        raw_sources = tool_result.get("sources", [])
        low_confidence = bool(tool_result.get("low_confidence", False))

        sources: list[SourceItem] = []
        if isinstance(raw_sources, list):
            for source in raw_sources:
                if not isinstance(source, dict):
                    continue
                title = str(source.get("title", "")).strip()
                url = str(source.get("url", "")).strip()
                if url:
                    sources.append({"title": title or url, "url": url})

        return {
            "context": context,
            "sources": sources,
            "expanded_query": expanded_query,
            "low_confidence": low_confidence,
        }

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


def _format_sources_block(sources: Sequence[SourceItem]) -> str:
    lines: list[str] = []
    for idx, source in enumerate(sources, start=1):
        title = source["title"].strip() or source["url"].strip()
        url = source["url"].strip()
        if url:
            lines.append(f"{idx}. {title} ({url})")
    return "\n".join(lines).strip()


def _build_low_confidence_response(query: str) -> str:
    cleaned_query = query.strip() or "din fråga"
    return LOW_CONFIDENCE_RESPONSE_TEMPLATE.format(query=cleaned_query)


def prepare_context(
    ui_messages: Sequence[dict[str, Any]], session_id: str
) -> tuple[list[BaseMessage], str, list[SourceItem], str, bool]:
    lc_messages = ui_messages_to_langchain_messages(ui_messages)
    if not lc_messages:
        raise ValueError("Inga giltiga meddelanden skickades till agenten.")

    graph = get_context_graph()
    result = graph.invoke(
        {"messages": lc_messages},
        config={"configurable": {"thread_id": session_id}},
    )

    search_query = str(result.get("search_query", "")).strip()
    expanded_query = str(result.get("expanded_query", search_query)).strip() or search_query
    context = str(result.get("context", "")).strip()
    low_confidence = bool(result.get("low_confidence", False))
    sources_raw = result.get("sources", [])

    sources: list[SourceItem] = []
    if isinstance(sources_raw, list):
        for source in sources_raw:
            if not isinstance(source, dict):
                continue
            title = str(source.get("title", "")).strip()
            url = str(source.get("url", "")).strip()
            if url:
                sources.append({"title": title or url, "url": url})

    debug_log("prepare_context.search_query=%s", search_query)
    debug_log("prepare_context.expanded_query=%s", expanded_query)
    debug_log("prepare_context.low_confidence=%s", low_confidence)

    return lc_messages, context, sources, expanded_query, low_confidence


def stream_answer_tokens(
    ui_messages: Sequence[dict[str, Any]],
    session_id: str,
) -> tuple[list[SourceItem], Iterator[str]]:
    lc_messages, context, sources, expanded_query, low_confidence = prepare_context(
        ui_messages, session_id
    )

    if not context:
        return sources, iter([FALLBACK_NO_CONTEXT_RESPONSE])

    if low_confidence:
        return sources, iter([_build_low_confidence_response(expanded_query)])

    sources_block = _format_sources_block(sources)
    system_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Sökfråga: {expanded_query or _last_user_message(lc_messages)}\n\n"
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
