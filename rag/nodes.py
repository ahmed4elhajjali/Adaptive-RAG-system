"""The six graph nodes plus the routing decisions, bundled in one class."""
import logging
from typing import Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from config import Settings, require_openai_key
from rag.state import GraphState

log = logging.getLogger(__name__)


# ---- Structured-output schemas for the LLM graders/router -------------------
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search"] = Field(
        description="Where to look up the answer.")


class GradeDocument(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Is the document relevant?")


class GradeGrounded(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Is the answer supported by the documents?")


class GradeUseful(BaseModel):
    binary_score: Literal["yes", "no"] = Field(description="Does the answer address the question?")


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs) or "(no context)"


class RAGNodes:
    def __init__(self, settings: Settings, vectorstore):
        self.s = settings
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
        llm = ChatOpenAI(
            model=settings.chat_model,
            temperature=0,
            api_key=require_openai_key(),
            base_url=settings.openai_base_url,
        )

        self.router = ChatPromptTemplate.from_messages([
            ("system", "You route user questions. The vectorstore holds documents about: {topics}. "
                       "Use 'vectorstore' for questions on those topics, otherwise use 'web_search' "
                       "(current events, other domains)."),
            ("human", "{question}"),
        ]) | llm.with_structured_output(RouteQuery)

        self.doc_grader = ChatPromptTemplate.from_messages([
            ("system", "Grade whether the document is relevant to the question (keyword or semantic "
                       "match). Answer yes or no."),
            ("human", "Document:\n{document}\n\nQuestion: {question}"),
        ]) | llm.with_structured_output(GradeDocument)

        self.generator = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using ONLY the context below. If the context is not "
                       "enough, say you don't know. Be concise (max 4 sentences).\n\nContext:\n{context}"),
            ("human", "{question}"),
        ]) | llm | StrOutputParser()

        self.grounded_grader = ChatPromptTemplate.from_messages([
            ("system", "Is the answer grounded in / supported by the facts? Answer yes or no."),
            ("human", "Facts:\n{documents}\n\nAnswer: {generation}"),
        ]) | llm.with_structured_output(GradeGrounded)

        self.useful_grader = ChatPromptTemplate.from_messages([
            ("system", "Does the answer resolve the question? Answer yes or no."),
            ("human", "Question: {question}\n\nAnswer: {generation}"),
        ]) | llm.with_structured_output(GradeUseful)

        self.tavily = None
        if settings.tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily = TavilyClient(api_key=settings.tavily_api_key)
            except ImportError:
                log.warning("tavily-python not installed; using mock web search.")

    @staticmethod
    def _step(state: GraphState, node: str, detail: str) -> list[dict]:
        return list(state.get("steps", [])) + [{"node": node, "detail": detail}]

    # ---- Node 1: route_question ------------------------------------------
    def route_question(self, state: GraphState) -> dict:
        result = self.router.invoke({"question": state["question"], "topics": self.s.kb_topics})
        return {"route": result.datasource,
                "steps": self._step(state, "route_question", f"Routed to {result.datasource}")}

    def route_decision(self, state: GraphState) -> str:
        return state["route"]

    # ---- Node 2: retrieve_documents --------------------------------------
    def retrieve_documents(self, state: GraphState) -> dict:
        docs = self.retriever.invoke(state["question"])
        return {"documents": docs,
                "steps": self._step(state, "retrieve_documents", f"Retrieved {len(docs)} documents")}

    # ---- Node 3: grade_documents -----------------------------------------
    def grade_documents(self, state: GraphState) -> dict:
        kept = []
        for doc in state["documents"]:
            grade = self.doc_grader.invoke({"question": state["question"], "document": doc.page_content})
            if grade.binary_score == "yes":
                kept.append(doc)
        total = len(state["documents"])
        return {"documents": kept,
                "needs_web_search": len(kept) == 0,
                "steps": self._step(state, "grade_documents", f"{len(kept)} of {total} documents relevant")}

    def decide_to_generate(self, state: GraphState) -> str:
        return "web_search" if state.get("needs_web_search") else "generate_answer"

    # ---- Node 4: web_search ----------------------------------------------
    def web_search(self, state: GraphState) -> dict:
        question = state["question"]
        try:
            if not self.tavily:
                raise RuntimeError("no Tavily key")
            hits = self.tavily.search(query=question, max_results=3).get("results", [])
            new_docs = [Document(page_content=h["content"], metadata={"source": h["url"]}) for h in hits]
            detail = f"Tavily returned {len(new_docs)} results"
        except Exception as exc:  # any failure -> mock fallback keeps the graph running
            log.info("Using mock web search (%s)", exc)
            new_docs = [Document(
                page_content=f"[MOCK SEARCH] No live web access is configured. Query was: '{question}'. "
                             "Set TAVILY_API_KEY for real results.",
                metadata={"source": "mock-search"})]
            detail = "Mock search used (set TAVILY_API_KEY for live results)"
        docs = list(state.get("documents", [])) + new_docs
        return {"documents": docs, "web_searched": True, "needs_web_search": False,
                "steps": self._step(state, "web_search", detail)}

    # ---- Node 5: generate_answer -----------------------------------------
    def generate_answer(self, state: GraphState) -> dict:
        answer = self.generator.invoke({"context": _format_docs(state.get("documents", [])),
                                        "question": state["question"]})
        attempt = state.get("retries", 0) + 1
        return {"generation": answer, "retries": attempt,
                "steps": self._step(state, "generate_answer", f"Attempt {attempt}")}

    # ---- Node 6: check_answer --------------------------------------------
    def check_answer(self, state: GraphState) -> dict:
        grounded = self.grounded_grader.invoke({
            "documents": _format_docs(state.get("documents", [])),
            "generation": state["generation"]}).binary_score
        useful = "no"
        if grounded == "yes":
            useful = self.useful_grader.invoke({
                "question": state["question"], "generation": state["generation"]}).binary_score
        return {"grounded": grounded, "useful": useful,
                "steps": self._step(state, "check_answer", f"grounded={grounded}, useful={useful}")}

    # ---- decide_to_finish (conditional edge, retry-limited) ---------------
    def decide_to_finish(self, state: GraphState) -> Literal["end", "regenerate", "web_search"]:
        out_of_retries = state.get("retries", 0) > self.s.max_retries
        if state["grounded"] == "yes":
            if state["useful"] == "yes" or out_of_retries:
                return "end"
            return "end" if state.get("web_searched") else "web_search"
        return "end" if out_of_retries else "regenerate"
