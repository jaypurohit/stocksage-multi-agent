from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBEDDING_MODEL, NEWS_COLLECTION


def get_vectorstore():
    """Open the persistent financial-news Chroma collection."""
    return Chroma(
        collection_name=NEWS_COLLECTION,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def retrieve(vectorstore, query, k=4, ticker=None):
    """
    Top-k relevant news chunks by cosine similarity.
    If `ticker` is given, filter to ONLY that symbol's news (metadata filtering)
    — combining semantic search with a hard metadata constraint.
    """
    where = {"ticker": ticker} if ticker else None
    return vectorstore.similarity_search(query, k=k, filter=where)