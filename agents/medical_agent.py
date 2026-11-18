from rag.rag_core import RAGCore
from config import MEDICAL_COLLECTION

MEDICAL_PROMPT = (
    "You are a careful biomedical research assistant. Answer using ONLY the "
    "numbered sources provided. Cite inline like [1]. If the sources don't "
    "contain enough information, say so plainly instead of guessing. Do not use "
    "outside knowledge."
)


class MedicalAgent:
    name = "medical"
    description = "Biomedical and clinical research — diseases, treatments, drugs, diabetes"

    def __init__(self, k=4):
        self.core = RAGCore(MEDICAL_COLLECTION, MEDICAL_PROMPT, k=k)

    def answer(self, query, ticker=None):      # ticker ignored (uniform signature)
        text, docs = self.core.answer(query)
        sources = [
            {"pmid": d.metadata.get("pmid"), "year": d.metadata.get("year"),
             "title": d.metadata.get("title"), "source": d.metadata.get("source")}
            for d in docs
        ]
        return {"specialist": self.name, "answer": text, "sources": sources}