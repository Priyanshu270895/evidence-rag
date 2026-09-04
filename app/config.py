from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_local_files_only: bool = True
    database_path: Path = Path("data/evidence_rag.db")
    upload_dir: Path = Path("data/uploads")
    chunk_size: int = 900
    chunk_overlap: int = 120
    max_upload_bytes: int = 25 * 1024 * 1024
    top_k: int = 5
    log_level: str = "INFO"
    ollama_timeout_seconds: float = 120.0
    ollama_max_retries: int = 1
    ollama_retry_backoff_seconds: float = 0.5
    grounding_min_support_score: float = 0.2
    vector_backend: Literal["sqlite", "qdrant"] = "sqlite"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "evidence_rag_chunks"
    qdrant_timeout_seconds: float = 10.0


settings = Settings()
