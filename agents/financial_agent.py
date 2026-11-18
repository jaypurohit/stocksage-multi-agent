from rag.rag_core import RAGCore
from config import NEWS_COLLECTION

FINANCIAL_PROMPT = (
    "You are a careful financial news research assistant. Answer using ONLY the "
    "numbered news sources provided. Cite inline like [1]. If the sources don't "
    "cover it, say so. Do not use outside knowledge, and do not give investment "
    "advice — only report what the sources say."
)


class FinancialAgent:
    name = "financial"
    description = "Stock market news, tickers, companies, earnings, and financial questions"

    def __init__(self, k=4):
        self.core = RAGCore(NEWS_COLLECTION, FINANCIAL_PROMPT, k=k)

    def answer(self, query, ticker=None):
        metadata_filter = {"ticker": ticker} if ticker else None
        text, docs = self.core.answer(query, metadata_filter=metadata_filter)
        sources = [
            {"ticker": d.metadata.get("ticker"), "date": d.metadata.get("date"),
             "headline": d.metadata.get("headline"), "source": d.metadata.get("source")}
            for d in docs
        ]
        return {"specialist": self.name, "answer": text, "sources": sources}