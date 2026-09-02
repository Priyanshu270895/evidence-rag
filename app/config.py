from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:1.5b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    database_path: Path = Path("data/evidence_rag.db")
    top_k: int = 5


settings = Settings()
