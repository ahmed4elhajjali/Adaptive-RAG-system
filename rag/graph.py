"""Wires the nodes into a compiled LangGraph workflow."""
from langgraph.graph import END, START, StateGraph

from config import Settings
from rag.knowledge_base import build_vectorstore
from rag.nodes import RAGNodes
from rag.state import GraphState


def build_graph(settings: Settings):
    n = RAGNodes(settings, build_vectorstore(settings))
    g = StateGraph(GraphState)

    g.add_node("route_question", n.route_question)
    g.add_node("retrieve_documents", n.retrieve_documents)
    g.add_node("grade_documents", n.grade_documents)
    g.add_node("web_search", n.web_search)
    g.add_node("generate_answer", n.generate_answer)
    g.add_node("check_answer", n.check_answer)

    g.add_edge(START, "route_question")
    g.add_conditional_edges("route_question", n.route_decision,
                            {"vectorstore": "retrieve_documents", "web_search": "web_search"})
    g.add_edge("retrieve_documents", "grade_documents")
    g.add_conditional_edges("grade_documents", n.decide_to_generate,
                            {"web_search": "web_search", "generate_answer": "generate_answer"})
    g.add_edge("web_search", "generate_answer")
    g.add_edge("generate_answer", "check_answer")
    g.add_conditional_edges("check_answer", n.decide_to_finish,
                            {"end": END, "regenerate": "generate_answer", "web_search": "web_search"})
    return g.compile()
