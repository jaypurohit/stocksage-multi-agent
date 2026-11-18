import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import MCPServer
from agents.medical_agent import MedicalAgent

mcp = MCPServer("pubmed-medical")
agent = MedicalAgent(k=4)


@mcp.tool()
def search_medical_research(question: str) -> str:
    """Answer a biomedical or clinical research question from PubMed abstracts,
    with citations back to source PMIDs.

    Args:
        question: the biomedical or clinical question to answer
    """
    result = agent.answer(question)
    sources = "\n".join(
        f"- PMID {s['pmid']} | {s['year']} | {s['title']}"
        for s in result["sources"]
    )
    return f"{result['answer']}\n\nSources:\n{sources}"


if __name__ == "__main__":
    mcp.run()