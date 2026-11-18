from typing import TypedDict, Optional


class SupervisorState(TypedDict):
    """State that flows through the supervisor graph. Each node reads and updates it."""
    query: str                 # the user's question
    route: list[str]           # which specialist(s) to use — a LIST, so fan-out
                               #   to multiple specialists later is no rewrite
    ticker: Optional[str]      # optional ticker hint for the financial specialist
    answers: list[dict]        # each specialist's result: {specialist, answer, sources}
    final_answer: str          # the synthesized answer