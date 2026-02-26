"""
Ingest script for compileit.com -> chunking -> embeddings -> Chroma vector store.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


DEFAULT_START_URL = os.getenv("COMPILEIT_START_URL", "https://compileit.com/")
DEFAULT_MAX_PAGES = int(os.getenv("COMPILEIT_MAX_PAGES", "0"))
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION_NAME", "compileit_docs")
DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./backend/chroma_db")
DEFAULT_ALLOWED_DOMAIN = os.getenv("COMPILEIT_ALLOWED_DOMAIN", "compileit.com")

ALLOWED_SCHEMES = {"http", "https"}
SKIP_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".pdf",
    ".zip",
    ".mp4",
    ".mp3",
}
PATH_DENYLIST_PREFIXES = (
    "/jobb",
    "/job",
    "/medarbetare",
    "/cookie",
    "/privacy",
    "/data_request",
    "/request_removal",
    "/connect",
    "/sso",
)
CONTACT_PATH_PREFIXES = ("/kontakt", "/kontakta")
BOILERPLATE_PATTERNS = (
    "webbplatsen använder cookies",
    "välj vilka cookies du vill godkänna",
    "nödvändiga cookies",
    "integritetspolicy",
    "cookie policy",
    "privacy policy",
)


@dataclass
class PageData:
    url: str
    title: str
    text: str


def normalize_url(url: str) -> str:
    parsed = urlsplit(url)
    # Drop query + fragment to avoid duplicate crawl entries with tracking params.
    clean = parsed._replace(query="", fragment="")
    normalized = urlunsplit(clean)
    return normalized.rstrip("/")


def canonical_host(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def is_same_domain(url: str, root_domain: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False

    candidate = canonical_host(url)
    root = root_domain.lower()
    return candidate == root


def should_skip_url(url: str, root_domain: str) -> bool:
    lowered = url.lower()
    if any(lowered.endswith(ext) for ext in SKIP_EXTENSIONS):
        return True

    parsed = urlparse(url)
    if not is_same_domain(url, root_domain):
        return True

    path = (parsed.path or "/").lower()
    if any(path.startswith(prefix) for prefix in PATH_DENYLIST_PREFIXES):
        return True

    return False


def should_skip_text_block(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in BOILERPLATE_PATTERNS)


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def is_contact_path(path: str) -> bool:
    lowered = (path or "/").lower()
    return any(lowered.startswith(prefix) for prefix in CONTACT_PATH_PREFIXES)


def classify_section(path: str) -> str:
    lowered = path.lower()
    if is_contact_path(lowered):
        return "contact"
    if lowered.startswith("/vara-tjanster"):
        return "services"
    if lowered.startswith("/uppdrag"):
        return "cases"
    if lowered.startswith("/om-oss"):
        return "about"
    if lowered.startswith("/nyheter"):
        return "news"
    return "general"


def infer_content_type(path: str) -> str:
    lowered = path.lower()
    if is_contact_path(lowered):
        return "contact"
    if lowered.startswith("/uppdrag"):
        return "case"
    if lowered.startswith("/nyheter"):
        return "article"
    if lowered in ("", "/"):
        return "landing"
    if lowered.startswith("/vara-tjanster"):
        return "service"
    return "page"


def is_low_quality_chunk(text: str, *, content_type: str = "page") -> bool:
    cleaned = normalize_text(text)
    is_contact = content_type == "contact"
    min_chars = 70 if is_contact else 120
    if len(cleaned) < min_chars:
        return True

    lowered = cleaned.lower()
    if should_skip_text_block(lowered):
        return True

    tokens = re.findall(r"[a-z0-9åäö]+", lowered)
    min_tokens = 12 if is_contact else 20
    if len(tokens) < min_tokens:
        return True

    unique_ratio = len(set(tokens)) / len(tokens)
    min_unique_ratio = 0.20 if is_contact else 0.30
    if unique_ratio < min_unique_ratio:
        return True

    return False


def extract_clean_text(html: str, page_url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.extract()

    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())

    path = (urlparse(page_url).path or "/").lower()
    is_contact_page = is_contact_path(path)
    min_len = 6 if is_contact_page else 20

    text_blocks: list[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "address"]):
        text = normalize_text(element.get_text(" ", strip=True))
        if len(text) >= min_len and not should_skip_text_block(text):
            text_blocks.append(text)

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        lowered_href = href.lower()
        if lowered_href.startswith("mailto:"):
            email = href.split(":", 1)[1].split("?", 1)[0].strip().replace(" ", "")
            if email:
                text_blocks.append(f"E-post: {email}")
        elif lowered_href.startswith("tel:"):
            phone = href.split(":", 1)[1].split("?", 1)[0].strip().replace(" ", "")
            if phone:
                text_blocks.append(f"Telefon: {phone}")

    if is_contact_page:
        # Contact pages often keep address/city lines in div wrappers instead of <p>/<li>.
        for element in soup.find_all("div"):
            text = normalize_text(element.get_text(" ", strip=True))
            lowered = text.lower()
            if len(text) < 30 or len(text) > 900:
                continue
            if should_skip_text_block(text):
                continue
            if (
                "våra kontor" in lowered
                or "kontakta oss" in lowered
                or "@compileit.com" in lowered
            ):
                text_blocks.append(text)

    deduped_blocks: list[str] = []
    seen_blocks: set[str] = set()
    for block in text_blocks:
        if block in seen_blocks:
            continue
        seen_blocks.add(block)
        deduped_blocks.append(block)

    combined = "\n".join(deduped_blocks).strip()
    return title, combined


def extract_internal_links(html: str, page_url: str, root_domain: str) -> Iterable[str]:
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href:
            continue

        absolute = normalize_url(urljoin(page_url, href))
        if should_skip_url(absolute, root_domain):
            continue

        yield absolute


def crawl_compileit(start_url: str, max_pages: int) -> list[PageData]:
    normalized_start = normalize_url(start_url)
    root_domain = DEFAULT_ALLOWED_DOMAIN

    queue = deque([normalized_start])
    visited: set[str] = set()
    pages: list[PageData] = []

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "CompileitRAGIngest/1.0 (+https://compileit.com; "
                "for interview project indexing)"
            )
        }
    )

    unlimited = max_pages <= 0

    while queue and (unlimited or len(visited) < max_pages):
        url = queue.popleft()
        if url in visited:
            continue
        if should_skip_url(url, root_domain):
            continue

        visited.add(url)
        if unlimited:
            print(f"[crawl] {len(visited)}: {url}")
        else:
            print(f"[crawl] {len(visited)}/{max_pages}: {url}")

        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Kunde inte hämta {url}: {exc}")
            continue

        title, text = extract_clean_text(response.text, page_url=url)
        if text:
            pages.append(PageData(url=url, title=title, text=text))

        for link in extract_internal_links(response.text, url, root_domain):
            if link not in visited:
                queue.append(link)

        time.sleep(0.2)

    return pages


def pages_to_documents(pages: list[PageData]) -> list[Document]:
    docs: list[Document] = []
    for page in pages:
        parsed = urlparse(page.url)
        path = parsed.path or "/"
        docs.append(
            Document(
                page_content=page.text,
                metadata={
                    "source": page.url,
                    "title": page.title,
                    "domain": canonical_host(page.url),
                    "path": path,
                    "section": classify_section(path),
                    "content_type": infer_content_type(path),
                },
            )
        )
    return docs


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=180,
    )
    raw_chunks = splitter.split_documents(documents)
    chunks: list[Document] = []
    seen_texts: set[str] = set()

    for chunk in raw_chunks:
        normalized = normalize_text(chunk.page_content)
        content_type = str(chunk.metadata.get("content_type", "page")).strip().lower()
        if is_low_quality_chunk(normalized, content_type=content_type):
            continue
        if normalized in seen_texts:
            continue

        seen_texts.add(normalized)
        chunk.page_content = normalized
        chunks.append(chunk)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(idx)

    return chunks


def index_documents(chunks: list[Document], persist_dir: str, collection_name: str) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = OpenAIEmbeddings()

    # Recreate collection by deleting existing entries first.
    existing = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    try:
        existing.delete_collection()
    except Exception:
        pass

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest compileit.com into ChromaDB")
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Max antal sidor att crawla. 0 eller negativt = inga begränsningar.",
    )
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    args = parser.parse_args()

    print("[info] Startar crawl...")
    pages = crawl_compileit(start_url=args.start_url, max_pages=args.max_pages)
    if not pages:
        raise RuntimeError("Inga sidor kunde indexeras. Avbryter.")

    print(f"[info] Crawlat sidor: {len(pages)}")

    docs = pages_to_documents(pages)
    chunks = chunk_documents(docs)
    print(f"[info] Skapade text-chunks: {len(chunks)}")

    index_documents(
        chunks=chunks,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )
    print(
        f"[ok] Ingestion klar. Persistens: {args.persist_dir}, "
        f"collection: {args.collection}"
    )


if __name__ == "__main__":
    main()
