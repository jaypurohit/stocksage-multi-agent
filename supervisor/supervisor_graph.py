from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

from supervisor.state import SupervisorState
from agents.financial_agent import FinancialAgent
from agents.medical_agent import MedicalAgent

load_dotenv()


class Supervisor:
    """Multi-agent orchestrator: route → dispatch → synthesize, over two
    specialist RAG agents (medical + financial)."""

    def __init__(self):
        self.specialists = {
            "financial": FinancialAgent(),
            "medical": MedicalAgent(),
        }
        # a small, cheap LLM call decides the route
        self.router_llm = ChatAnthropic(model="claude-sonnet-5-5", max_tokens=20)
        self.graph = self._build()

    # ---------- NODES ----------

    def _route(self, state: SupervisorState) -> SupervisorState:
        """Real routing: ask the LLM which specialist fits the query.
        (A cheap embedding classifier with this as fallback is the next
        optimization — LLM routing is the clean first version.)"""
        options = "\n".join(f"- {n}: {a.description}" for n, a in self.specialists.items())
        prompt = (
            f"Route this question to the single best specialist.\n\n{options}\n\n"
            f"Question: {state['query']}\n\n"
            f"Reply with ONLY the specialist name, one word."
        )
        choice = self.router_llm.invoke([("human", prompt)]).content.strip().lower()
        if choice not in self.specialists:        # guard: default if unclear
            choice = "financial"
        print(f"🧭 [ROUTE] → {choice}")
        return {**state, "route": [choice]}

    def _dispatch(self, state: SupervisorState) -> SupervisorState:
        answers = []
        for name in state["route"]:
            print(f"🔧 [DISPATCH] calling {name} specialist")
            answers.append(self.specialists[name].answer(state["query"], ticker=state.get("ticker")))
        return {**state, "answers": answers}

    def _synthesize(self, state: SupervisorState) -> SupervisorState:
        if len(state["answers"]) == 1:
            final = state["answers"][0]["answer"]
        else:
            final = "\n\n".join(f"[{a['specialist']}] {a['answer']}" for a in state["answers"])
        return {**state, "final_answer": final}

    # ---------- WIRING ----------

    def _build(self):
        g = StateGraph(SupervisorState)
        g.add_node("route", self._route)
        g.add_node("dispatch", self._dispatch)
        g.add_node("synthesize", self._synthesize)
        g.set_entry_point("route")
        g.add_edge("route", "dispatch")
        g.add_edge("dispatch", "synthesize")
        g.add_edge("synthesize", END)
        return g.compile()

    def answer(self, query, ticker=None):
        initial = {"query": query, "route": [], "ticker": ticker, "answers": [], "final_answer": ""}
        final = self.graph.invoke(initial)
        return final["final_answer"], final["answers"]


if __name__ == "__main__":
    sup = Supervisor()

    tests = [
        "What are common treatments for type 2 diabetes?",      # → medical
        "What's happening with AI chip demand and memory?",      # → financial
    ]
    for q in tests:
        print("=" * 70, "\nQ:", q)
        ans, answers = sup.answer(q)
        print("A:", ans[:400], "...\n")