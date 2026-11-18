from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from rag.retriever import get_vectorstore, retrieve
from rag.prompt_builder import SYSTEM_PROMPT, build_user_message

load_dotenv()  # ANTHROPIC_API_KEY

CLAUDE_MODEL = "claude-sonnet-5-5"   # ← see note below; use a model your key supports


class FinancialRAG:
    """question → retrieve news → grounded prompt → Claude → cited answer."""

    def __init__(self, k=4):
        self.k = k
        self.vectorstore = get_vectorstore()
        self.llm = ChatAnthropic(model=CLAUDE_MODEL, max_tokens=1024)

    def answer(self, question, ticker=None):
        # 1. RETRIEVE (optionally filtered to one ticker)
        docs = retrieve(self.vectorstore, question, k=self.k, ticker=ticker)
        # 2. AUGMENT — build the grounded prompt
        user_message = build_user_message(question, docs)
        # 3. GENERATE the cited answer
        response = self.llm.invoke([
            ("system", SYSTEM_PROMPT),
            ("human", user_message),
        ])
        return response.content, docs


def print_sources(docs):
    print("\nSources:")
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        print(f"  [{i}] {m.get('ticker')} | {m.get('date')} | {m.get('headline')}")
        print(f"      {m.get('source')} — {m.get('url')}")


if __name__ == "__main__":
    rag = FinancialRAG(k=4)

    # a general query (searches all tickers)
    q1 = "What is happening with AI chip demand and memory shortages?"
    print("=" * 70, "\nQ:", q1)
    ans, docs = rag.answer(q1)
    print("\nA:", ans)
    print_sources(docs)

    # a ticker-FILTERED query (only NVDA news) — the metadata-filter feature
    q2 = "What's the latest news?"
    print("\n", "=" * 70, "\nQ (NVDA only):", q2)
    ans, docs = rag.answer(q2, ticker="NVDA")
    print("\nA:", ans)
    print_sources(docs)