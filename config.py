"""Environment-driven configuration. Nothing is hard-coded except defaults."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def require_openai_key() -> str:
    try:
        return os.environ["OPENAI_API_KEY"]
    except KeyError as exc:
        raise ConfigError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        ) from exc


@dataclass(frozen=True)
class Settings:
    chat_model: str
    embedding_model: str
    openai_base_url: str | None
    tavily_api_key: str | None
    top_k: int
    max_retries: int
    recursion_limit: int
    kb_dir: str
    kb_topics: str
    host: str
    port: int
    debug: bool


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        chat_model=env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_base_url=env("OPENAI_BASE_URL") or None,
        tavily_api_key=env("TAVILY_API_KEY") or None,
        top_k=int(env("RETRIEVER_TOP_K", "4")),
        max_retries=int(env("MAX_GENERATION_RETRIES", "2")),
        recursion_limit=int(env("RECURSION_LIMIT", "25")),
        kb_dir=env("KB_DIR", "knowledge_base"),
        kb_topics=env("KB_TOPICS", "LangGraph, LangChain, RAG, vector stores, Flask"),
        host=env("FLASK_HOST", "127.0.0.1"),
        port=int(env("FLASK_PORT", "5000")),
        debug=env("FLASK_DEBUG", "0") == "1",
    )
