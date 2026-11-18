import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import ALLOWED_SYMBOLS, NEWS_BRONZE_PATH, CHROMA_DIR

load_dotenv()  # loads OPENAI_API_KEY from .env

# ---- settings (same embedding/chunking choices as the PubMed RAG) ----
EMBEDDING_MODEL = "text-embedding-3-small"   # 1536-dim, cheap, good for RAG
PERSIST_DIR = CHROMA_DIR              # local vector store location
COLLECTION_NAME = "financial_news"           # this corpus's collection
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 100


def _ts_to_date(unix_ts):
    """Finnhub gives a Unix timestamp (seconds) → readable ISO date string."""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d")


def build_documents():
    """
    Load every ticker's raw news JSON and turn each article into a LangChain
    Document: the text we embed is headline + summary; metadata carries the
    ticker (for symbol-filtered retrieval), source, url, and date.
    """
    docs = []
    for symbol in ALLOWED_SYMBOLS:
        path = os.path.join(NEWS_BRONZE_PATH, f"{symbol}.json")
        if not os.path.exists(path):
            print(f"  no news file for {symbol}, skipping")
            continue

        with open(path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        for a in articles:
            headline = a.get("headline", "").strip()
            summary = a.get("summary", "").strip()
            if not headline:
                continue   # skip anything with no headline (nothing to embed)

            # the text that gets embedded + retrieved
            content = f"{headline}. {summary}".strip()

            docs.append(Document(
                page_content=content,
                metadata={
                    "ticker": symbol,                       # enables ticker-filtered retrieval
                    "headline": headline,
                    "source": a.get("source", ""),
                    "url": a.get("url", ""),
                    "date": _ts_to_date(a.get("datetime", 0)),
                    "article_id": str(a.get("id", "")),     # for stable chunk ids
                },
            ))
    return docs


def build_index():
    # 1. build Documents from the raw news
    docs = build_documents()
    print(f"Built {len(docs)} article documents.")

    # 2. chunk them (articles are short, so most become a single chunk —
    #    we still run the splitter for consistency with the RAG pattern)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    # stable id per chunk so re-running UPDATES instead of duplicating
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata['ticker']}_{c.metadata['article_id']}_{i}"
    print(f"Produced {len(chunks)} chunks.")

    # 3. open (or create) the Chroma collection, cosine similarity
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedder,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 4. embed + store in batches (idempotent via chunk_id)
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        ids = [c.metadata["chunk_id"] for c in batch]
        vectorstore.add_documents(documents=batch, ids=ids)
        print(f"  stored {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"\nDone. Financial news index → {PERSIST_DIR} (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    build_index()