from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBEDDING_MODEL

load_dotenv()
CLAUDE_MODEL = "claude-sonnet-5-5"


class RAGCore:
    """
    Reusable RAG: one Chroma collection + one grounding prompt. Both specialists
    use this — they differ ONLY by which collection and which domain prompt.
    That's the DRY refactor: the retrieve → ground → generate logic lives once.
    """
    def __init__(self, collection, system_prompt, k=4):
        self.k = k
        self.system_prompt = system_prompt
        self.vectorstore = Chroma(
            collection_name=collection,
            embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=CHROMA_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.llm = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024)

    def answer(self, query, metadata_filter=None):
        # retrieve (optionally filtered by metadata, e.g. a ticker)
        docs = self.vectorstore.similarity_search(query, k=self.k, filter=metadata_filter)
        context = self._format(docs)
        user_msg = (
            f"Sources:\n{context}\n\nQuestion: {query}\n\n"
            f"Answer using only the sources above, citing them with [number]."
        )
        resp = self.llm.invoke([("system", self.system_prompt), ("human", user_msg)])
        return resp.content, docs

    def _format(self, docs):
        """Numbered context block — works for both domains via metadata fallbacks."""
        blocks = []
        for i, d in enumerate(docs, 1):
            m = d.metadata
            label = m.get("headline") or m.get("title") or "source"
            tag = m.get("ticker") or m.get("pmid") or ""
            date = m.get("date") or m.get("year") or ""
            blocks.append(f"[{i}] ({tag} {date} {m.get('source','')}) {label}\n{d.page_content}")
        return "\n\n".join(blocks)