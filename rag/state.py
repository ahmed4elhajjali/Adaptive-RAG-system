"""Shared state passed between every node in the graph."""
from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    question: str               # the user's question
    generation: str             # latest generated answer
    documents: List[Document]   # retrieved and/or web-searched context
    route: str                  # "vectorstore" | "web_search"
    needs_web_search: bool      # set by grade_documents when nothing relevant remains
    web_searched: bool          # True once web_search ran (prevents search loops)
    retries: int                # number of generation attempts so far
    grounded: str               # "yes" | "no" - answer supported by documents?
    useful: str                 # "yes" | "no" - answer addresses the question?
    steps: List[dict]           # execution trace: [{"node": ..., "detail": ...}]
