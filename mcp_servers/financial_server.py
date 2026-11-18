# make project modules importable no matter how this server is launched
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import MCPServer
from agents.financial_agent import FinancialAgent

mcp = MCPServer("stocksage-financial")
agent = FinancialAgent(k=4)   # built once, reused per call


@mcp.tool()
def search_financial_news(question: str, ticker: str = "") -> str:
    """Answer a stock-market or financial question from recent market news,
    with citations. Optionally scope the search to a single ticker.

    Args:
        question: the financial / market question to answer
        ticker: optional stock symbol like 'NVDA' to scope the news (empty = all tickers)
    """
    result = agent.answer(question, ticker=ticker or None)
    sources = "\n".join(
        f"- {s['ticker']} | {s['date']} | {s['headline']} ({s['source']})"
        for s in result["sources"]
    )
    return f"{result['answer']}\n\nSources:\n{sources}"


if __name__ == "__main__":
    mcp.run()