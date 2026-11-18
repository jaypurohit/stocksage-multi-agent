import os
import json
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import MEDICAL_DATA, CHROMA_DIR, EMBEDDING_MODEL, MEDICAL_COLLECTION

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 100


def build_documents():
    """Load PubMed abstracts and turn each into a Document (title + abstract
    as content; pmid/title/journal/year/source as metadata)."""
    with open(MEDICAL_DATA, "r", encoding="utf-8") as f:
        abstracts = json.load(f)

    docs = []
    for a in abstracts:
        title = a.get("title", "").strip()
        abstract = a.get("abstract", "").strip()
        if not abstract:
            continue
        content = f"{title}. {abstract}".strip()
        docs.append(Document(
            page_content=content,
            metadata={
                "pmid": a.get("pmid", ""),
                "title": title,
                "journal": a.get("journal", ""),
                "year": a.get("year", ""),
                "source": a.get("source", ""),
            },
        ))
    return docs


def build_index():
    docs = build_documents()
    print(f"Built {len(docs)} abstract documents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"pubmed_{c.metadata['pmid']}_{i}"
    print(f"Produced {len(chunks)} chunks.")

    vectorstore = Chroma(
        collection_name=MEDICAL_COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,          # SAME chroma_db, different collection
        collection_metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        ids = [c.metadata["chunk_id"] for c in batch]
        vectorstore.add_documents(documents=batch, ids=ids)
        print(f"  stored {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"\nDone. Medical index → {CHROMA_DIR} (collection: {MEDICAL_COLLECTION})")


if __name__ == "__main__":
    build_index()