"""Builds the in-memory vector store from local files or built-in samples."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings


SAMPLE_DOCS = [
    (
        "langgraph",
        "LangGraph is a library for building stateful, multi-step LLM applications as graphs. "
        "Nodes are Python functions that read and update a shared state; edges define what runs next."
    ),
    (
        "adaptive-rag",
        "Adaptive RAG combines query routing, document grading and self-correction. "
        "A router chooses between a local vector store and web search."
    ),
    (
        "rag",
        "Retrieval-Augmented Generation (RAG) grounds an LLM's answer in external documents. "
        "Documents are embedded, stored in a vector store, retrieved by similarity to the question, "
        "and inserted into the prompt as context."
    ),
    (
        "vector-store",
        "A vector store indexes embeddings so semantically similar text can be found quickly. "
        "InMemoryVectorStore keeps everything in RAM."
    ),
    (
        "hallucination",
        "Hallucination grading checks whether an answer is supported by retrieved documents."
    ),
    (
        "flask",
        "Flask is a lightweight Python web framework."
    ),
    (
        "langchain",
        "LangChain provides building blocks for LLM applications including chat models, "
        "retrievers, prompts and structured output."
    ),
]


def _load_local_docs(kb_dir: str) -> list[Document]:
    root = Path(kb_dir)

    if not root.is_dir():
        return []

    docs = []

    for path in sorted(root.rglob("*")):

        if path.suffix.lower() in {".txt", ".md"} and path.is_file():

            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name
                        }
                    )
                )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    return splitter.split_documents(docs)


class FastEmbedEmbeddings(Embeddings):
    """
    Local embeddings using FastEmbed.
    No API key and no external embeddings API.
    """

    def __init__(self, model_name: str):

        from fastembed import TextEmbedding

        print(
            f"[Embeddings] Loading FastEmbed model: {model_name}"
        )

        self._model = TextEmbedding(
            model_name=model_name
        )

        print(
            "[Embeddings] FastEmbed model loaded successfully."
        )

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        vectors = self._model.embed(texts)

        return [
            vector.tolist()
            for vector in vectors
        ]

    def embed_query(
        self,
        text: str
    ) -> list[float]:

        vector = next(
            iter(
                self._model.embed([text])
            )
        )

        return vector.tolist()


def _build_embeddings(
    settings: Settings
) -> Embeddings:

    provider = getattr(
        settings,
        "embedding_provider",
        "fastembed"
    )

    model = getattr(
        settings,
        "embedding_model",
        "BAAI/bge-small-en-v1.5"
    )

    print(
        f"[Embeddings] Provider: {provider}"
    )

    print(
        f"[Embeddings] Model: {model}"
    )

    if provider.lower() == "fastembed":

        return FastEmbedEmbeddings(
            model
        )

    raise ValueError(
        f"Unsupported embedding provider: {provider}. "
        f"Use 'fastembed'."
    )


def build_vectorstore(
    settings: Settings
) -> InMemoryVectorStore:

    docs = _load_local_docs(
        settings.kb_dir
    )

    if not docs:

        docs = [
            Document(
                page_content=text,
                metadata={
                    "source": name
                }
            )
            for name, text in SAMPLE_DOCS
        ]

    print(
        f"[VectorStore] Documents: {len(docs)}"
    )

    embeddings = _build_embeddings(
        settings
    )

    print(
        "[VectorStore] Building vector store..."
    )

    vectorstore = InMemoryVectorStore.from_documents(
        docs,
        embeddings
    )

    print(
        "[VectorStore] Vector store ready."
    )

    return vectorstore