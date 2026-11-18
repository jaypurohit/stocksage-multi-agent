# The grounding instruction — this is what keeps answers trustworthy:
# answer ONLY from the provided news, cite sources, refuse if not covered,
# and (important for a financial assistant) do NOT give investment advice.
SYSTEM_PROMPT = (
    "You are a careful financial news research assistant. "
    "Answer the user's question using ONLY the numbered news sources provided. "
    "Cite the sources you use inline with bracketed numbers like [1] or [2]. "
    "If the sources do not contain enough information to answer, say so plainly "
    "instead of guessing. Do not use outside knowledge, and do not give "
    "investment or financial advice — only report what the sources say."
)


def format_context(docs):
    """Render retrieved news as a numbered list the model can cite by number."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        header = f"[{i}] ({m.get('ticker')}, {m.get('date')}, {m.get('source')}) {m.get('headline')}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_user_message(query, docs):
    context = format_context(docs)
    return (
        f"News sources:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the sources above, citing them with [number]."
    )